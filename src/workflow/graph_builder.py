# src/workflow/graph_builder.py
"""
LangGraph workflow для системы оценки рисков ИИ-агентов
Создает и настраивает мультиагентный граф для полного цикла оценки
"""

import asyncio
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph

from ..models.risk_models import (
    WorkflowState, RiskType, ProcessingStatus, AgentRiskAssessment,
    AgentProfile, AgentTaskResult
)
from ..agents.profiler_agent import create_profiler_agent, create_profiler_node_function
from ..agents.evaluator_agents import (
    create_all_evaluator_agents, create_evaluator_nodes_for_langgraph_safe,
    extract_risk_evaluations_from_results, calculate_overall_risk_score,
    get_highest_risk_areas, create_critic_node_function_fixed
)
from ..agents.critic_agent import (
    create_critic_agent, create_quality_check_router
)
from ..agents.evaluator_agents import create_critic_node_function_fixed
from ..utils.logger import get_langgraph_logger, log_graph_node, log_conditional_edge_func
from ..models.database import get_db_manager


class RiskAssessmentWorkflow:
    """
    Основной workflow для оценки рисков ИИ-агентов
    
    Граф включает:
    1. Профилирование агента
    2. Параллельная оценка 6 типов рисков  
    3. Критический анализ качества
    4. Повторы при необходимости
    5. Финализация результата
    """
    
    def __init__(
        self,
        llm_base_url: str = "http://127.0.0.1:1234",
        llm_model: str = "qwen3-4b",
        quality_threshold: float = 7.0,
        max_retries: int = 3
    ):
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.quality_threshold = quality_threshold
        self.max_retries = max_retries
        
        # Создаем агентов
        self.profiler = create_profiler_agent(llm_base_url, llm_model)
        self.evaluators = create_all_evaluator_agents(llm_base_url, llm_model)
        self.critic = create_critic_agent(llm_base_url, llm_model, quality_threshold)
        
        # Логгер для LangGraph
        self.graph_logger = get_langgraph_logger()
        
        # Создаем граф
        self.graph = self._build_graph()
    
    def _build_graph(self) -> CompiledGraph:
        """Построение LangGraph workflow"""
        
        # Создаем StateGraph
        workflow = StateGraph(WorkflowState)
        
        # Добавляем узлы
        self._add_nodes(workflow)
        
        # Добавляем рёбра
        self._add_edges(workflow)
        
        # Устанавливаем точки входа и выхода
        workflow.set_entry_point("initialization")
        workflow.set_finish_point("finalization")
        
        return workflow.compile()
    
    def _add_nodes(self, workflow: StateGraph):
        """ИСПРАВЛЕННОЕ добавление узлов с батчированными оценщиками"""
        
        # 1. Инициализация
        workflow.add_node("initialization", self._initialization_node)
        
        # 2. Профилирование
        profiler_node = create_profiler_node_function(self.profiler)
        workflow.add_node("profiling", log_graph_node("profiling")(profiler_node))
        
        # 3. Подготовка к оценке
        workflow.add_node("evaluation_preparation", self._evaluation_preparation_node)
        
        # 4. НОВЫЕ БАТЧИРОВАННЫЕ УЗЛЫ ОЦЕНКИ
        workflow.add_node("batch_1_evaluation", self._batch_1_evaluation_node)
        workflow.add_node("batch_2_evaluation", self._batch_2_evaluation_node) 
        workflow.add_node("batch_3_evaluation", self._batch_3_evaluation_node)
        
        # 5. Сбор результатов
        workflow.add_node("evaluation_collection", self._evaluation_collection_node)
        
        # 6. Критический анализ
        critic_node = create_critic_node_function_fixed(self.critic)
        workflow.add_node("critic_analysis", log_graph_node("critic_analysis")(critic_node))
        
        # 7. Проверка качества
        workflow.add_node("quality_check", self._quality_check_node)
        
        # 8. Повторная оценка
        workflow.add_node("retry_evaluation", self._retry_evaluation_node)
        
        # 9. Финализация
        workflow.add_node("finalization", self._finalization_node)
        
        # 10. Обработка ошибок
        workflow.add_node("error_handling", self._error_handling_node)
    
    def _add_edges(self, workflow: StateGraph):
        """ИСПРАВЛЕННОЕ добавление рёбер с батчированием агентов"""
        
        # Последовательные переходы
        workflow.add_edge("initialization", "profiling")
        workflow.add_edge("profiling", "evaluation_preparation")
        
        # НОВОЕ: Батчированная оценка рисков (по 2 агента за раз)
        
        # Батч 1: Этические и социальные риски
        workflow.add_edge("evaluation_preparation", "batch_1_evaluation")
        
        # Батч 2: Безопасность и стабильность  
        workflow.add_edge("batch_1_evaluation", "batch_2_evaluation")
        
        # Батч 3: Автономность и регуляторные
        workflow.add_edge("batch_2_evaluation", "batch_3_evaluation")
        
        # После всех батчей - сбор результатов
        workflow.add_edge("batch_3_evaluation", "evaluation_collection")
        
        # Основной поток остается прежним
        workflow.add_edge("evaluation_collection", "critic_analysis")
        workflow.add_edge("critic_analysis", "quality_check")
        
        # Условные переходы из проверки качества
        workflow.add_conditional_edges(
            "quality_check",
            log_conditional_edge_func("quality_check_router")(self._quality_check_router),
            {
                "retry": "retry_evaluation",
                "finalize": "finalization",
                "error": "error_handling"
            }
        )
        
        # Из повторной оценки обратно к подготовке
        workflow.add_edge("retry_evaluation", "evaluation_preparation")
        
        # Завершение
        workflow.add_edge("finalization", END)
        workflow.add_edge("error_handling", END)
    
    # ===============================
    # Узлы графа
    # ===============================
    
    @log_graph_node("batch_1_evaluation")
    async def _batch_1_evaluation_node(self, state: WorkflowState) -> WorkflowState:
        """Батч 1: Этические и социальные риски (параллельно)"""
        
        assessment_id = state["assessment_id"]
        agent_profile = state.get("agent_profile", {})
        
        self.graph_logger.log_workflow_step(
            assessment_id, 
            "batch_1_evaluation", 
            "Запуск Батча 1: этические + социальные риски"
        )
        
        # Подготавливаем входные данные
        input_data = {"agent_profile": agent_profile}
        
        try:
            # Запускаем 2 агента параллельно
            ethical_task = self.evaluators[RiskType.ETHICAL].run(input_data, assessment_id)
            social_task = self.evaluators[RiskType.SOCIAL].run(input_data, assessment_id)
            
            # Ждем завершения обоих
            ethical_result, social_result = await asyncio.gather(
                ethical_task, social_task, return_exceptions=True
            )
            
            # Обрабатываем результаты
            if not isinstance(ethical_result, Exception):
                state.set_evaluation_result("ethical", ethical_result)
                self.graph_logger.log_workflow_step(
                    assessment_id, "batch_1_ethical", 
                    f"Этические риски: {ethical_result.status}"
                )
            else:
                state.set_evaluation_result("ethical", self._create_error_result(
                    "ethical_risk_evaluator", str(ethical_result)
                ))
            
            if not isinstance(social_result, Exception):
                state.set_evaluation_result("social", social_result)
                self.graph_logger.log_workflow_step(
                    assessment_id, "batch_1_social",
                    f"Социальные риски: {social_result.status}"
                )
            else:
                state.set_evaluation_result("social", self._create_error_result(
                    "social_risk_evaluator", str(social_result)
                ))
            
            # Небольшая пауза перед следующим батчем
            await asyncio.sleep(2)
            
        except Exception as e:
            self.graph_logger.log_workflow_step(
                assessment_id, "batch_1_error", f"Ошибка батча 1: {e}"
            )
            
            # Создаем error результаты для обоих типов
            state.set_evaluation_result("ethical", self._create_error_result(
                "ethical_risk_evaluator", f"Ошибка батча: {e}"
            ))
            state.set_evaluation_result("social", self._create_error_result(
                "social_risk_evaluator", f"Ошибка батча: {e}"
            ))
        
        state["current_step"] = "batch_2_evaluation"
        return state

    @log_graph_node("batch_2_evaluation") 
    async def _batch_2_evaluation_node(self, state: WorkflowState) -> WorkflowState:
        """Батч 2: Безопасность и стабильность (параллельно)"""
        
        assessment_id = state["assessment_id"]
        agent_profile = state.get("agent_profile", {})
        
        self.graph_logger.log_workflow_step(
            assessment_id, 
            "batch_2_evaluation", 
            "Запуск Батча 2: безопасность + стабильность"
        )
        
        input_data = {"agent_profile": agent_profile}
        
        try:
            # Запускаем 2 агента параллельно
            security_task = self.evaluators[RiskType.SECURITY].run(input_data, assessment_id)
            stability_task = self.evaluators[RiskType.STABILITY].run(input_data, assessment_id)
            
            security_result, stability_result = await asyncio.gather(
                security_task, stability_task, return_exceptions=True
            )
            
            # Обрабатываем результаты
            if not isinstance(security_result, Exception):
                state.set_evaluation_result("security", security_result)
                self.graph_logger.log_workflow_step(
                    assessment_id, "batch_2_security",
                    f"Безопасность: {security_result.status}"
                )
            else:
                state.set_evaluation_result("security", self._create_error_result(
                    "security_risk_evaluator", str(security_result)
                ))
            
            if not isinstance(stability_result, Exception):
                state.set_evaluation_result("stability", stability_result)
                self.graph_logger.log_workflow_step(
                    assessment_id, "batch_2_stability",
                    f"Стабильность: {stability_result.status}"
                )
            else:
                state.set_evaluation_result("stability", self._create_error_result(
                    "stability_risk_evaluator", str(stability_result)
                ))
            
            await asyncio.sleep(2)  # Пауза
            
        except Exception as e:
            self.graph_logger.log_workflow_step(
                assessment_id, "batch_2_error", f"Ошибка батча 2: {e}"
            )
            
            state.set_evaluation_result("security", self._create_error_result(
                "security_risk_evaluator", f"Ошибка батча: {e}"
            ))
            state.set_evaluation_result("stability", self._create_error_result(
                "stability_risk_evaluator", f"Ошибка батча: {e}"
            ))
        
        state["current_step"] = "batch_3_evaluation"
        return state

    @log_graph_node("batch_3_evaluation")
    async def _batch_3_evaluation_node(self, state: WorkflowState) -> WorkflowState:
        """Батч 3: Автономность и регуляторные риски (параллельно)"""
        
        assessment_id = state["assessment_id"]
        agent_profile = state.get("agent_profile", {})
        
        self.graph_logger.log_workflow_step(
            assessment_id, 
            "batch_3_evaluation", 
            "Запуск Батча 3: автономность + регуляторные риски"
        )
        
        input_data = {"agent_profile": agent_profile}
        
        try:
            # Запускаем 2 агента параллельно
            autonomy_task = self.evaluators[RiskType.AUTONOMY].run(input_data, assessment_id)
            regulatory_task = self.evaluators[RiskType.REGULATORY].run(input_data, assessment_id)
            
            autonomy_result, regulatory_result = await asyncio.gather(
                autonomy_task, regulatory_task, return_exceptions=True
            )
            
            # Обрабатываем результаты
            if not isinstance(autonomy_result, Exception):
                state.set_evaluation_result("autonomy", autonomy_result)
                self.graph_logger.log_workflow_step(
                    assessment_id, "batch_3_autonomy",
                    f"Автономность: {autonomy_result.status}"
                )
            else:
                state.set_evaluation_result("autonomy", self._create_error_result(
                    "autonomy_risk_evaluator", str(autonomy_result)
                ))
            
            if not isinstance(regulatory_result, Exception):
                state.set_evaluation_result("regulatory", regulatory_result)
                self.graph_logger.log_workflow_step(
                    assessment_id, "batch_3_regulatory",
                    f"Регуляторные: {regulatory_result.status}"
                )
            else:
                state.set_evaluation_result("regulatory", self._create_error_result(
                    "regulatory_risk_evaluator", str(regulatory_result)
                ))
            
            await asyncio.sleep(2)  # Финальная пауза
            
        except Exception as e:
            self.graph_logger.log_workflow_step(
                assessment_id, "batch_3_error", f"Ошибка батча 3: {e}"
            )
            
            state.set_evaluation_result("autonomy", self._create_error_result(
                "autonomy_risk_evaluator", f"Ошибка батча: {e}"
            ))
            state.set_evaluation_result("regulatory", self._create_error_result(
                "regulatory_risk_evaluator", f"Ошибка батча: {e}"
            ))
        
        state["current_step"] = "evaluation_collection"
        return state

    # ===============================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ===============================

    def _create_error_result(self, agent_name: str, error_message: str) -> Dict[str, Any]:
        """Создает стандартизированный результат ошибки"""
        return {
            "status": "failed",
            "result_data": None,
            "agent_name": agent_name,
            "error_message": error_message,
            "execution_time_seconds": 0.0,
            "start_time": datetime.now(),
            "end_time": datetime.now()
        }

    @log_graph_node("evaluation_collection")
    async def _evaluation_collection_node(self, state: WorkflowState) -> WorkflowState:
        """ИСПРАВЛЕННЫЙ сбор результатов батчированной оценки"""
        assessment_id = state["assessment_id"]
        
        # Получаем сводку по результатам оценки
        evaluation_summary = state.get_evaluation_summary()
        
        self.graph_logger.log_workflow_step(
            assessment_id,
            "evaluation_collection",
            f"Сбор результатов: {evaluation_summary['successful_evaluations']}/{evaluation_summary['total_evaluations']} успешно"
        )
        
        # Логируем детальную информацию
        successful_evaluations = state.get_successful_evaluations()
        failed_evaluations = state.get_failed_evaluations()
        
        if successful_evaluations:
            self.graph_logger.log_workflow_step(
                assessment_id, 
                "successful_evaluations",
                f"Успешные оценки: {list(successful_evaluations.keys())}"
            )
        
        if failed_evaluations:
            self.graph_logger.log_workflow_step(
                assessment_id,
                "failed_evaluations", 
                f"Неудачные оценки: {list(failed_evaluations.keys())}"
            )
        
        # Проверяем критический порог успешности 
        success_rate = evaluation_summary["success_rate"]
        if success_rate < 0.5:  # Менее 50% успешных оценок
            state["current_step"] = "error"
            state["error_message"] = f"Критически низкий процент успешных оценок: {success_rate:.1%}"
            return state
        
        # Переходим к критическому анализу
        state["current_step"] = "critic_analysis"
        return state

    @log_graph_node("initialization")
    async def _initialization_node(self, state: WorkflowState) -> WorkflowState:
        """Инициализация workflow"""
        assessment_id = state.get("assessment_id") or f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.graph_logger.log_graph_start(assessment_id, "risk_assessment_workflow")
        
        # Валидация входных данных
        if not state.get("source_files"):
            state["current_step"] = "error"
            state["error_message"] = "Не предоставлены файлы для анализа"
            return state
        
        # Инициализация состояния
        state.update({
            "assessment_id": assessment_id,
            "current_step": "profiling",
            "retry_count": {},
            "max_retries": self.max_retries,
            "quality_threshold": self.quality_threshold,
            "evaluation_results": {},
            "critic_results": {},
            "start_time": datetime.now()
        })
        
        return state
    
    @log_graph_node("evaluation_preparation")
    async def _evaluation_preparation_node(self, state: WorkflowState) -> WorkflowState:
        """Подготовка к параллельной оценке рисков"""
        assessment_id = state["assessment_id"]
        
        # Проверяем наличие профиля агента
        if not state.get("agent_profile"):
            state["current_step"] = "error"
            state["error_message"] = "Отсутствует профиль агента после профилирования"
            return state
        
        self.graph_logger.log_workflow_step(
            assessment_id, 
            "evaluation_preparation", 
            "Подготовка к параллельной оценке 6 типов рисков"
        )
        
        # Очищаем предыдущие результаты при повторах
        state["evaluation_results"] = {}
        state["current_step"] = "parallel_evaluation"
        
        return state
    
    
    
    @log_graph_node("quality_check")
    async def _quality_check_node(self, state: WorkflowState) -> WorkflowState:
        """Проверка качества и принятие решения о следующих шагах - ИСПРАВЛЕННАЯ"""
        assessment_id = state["assessment_id"]
        critic_results = state.get("critic_results", {})
        retry_count = state.get("retry_count", {})
        max_retries = state.get("max_retries", 3)
        
        # Определяем, какие риски нуждаются в повторной оценке
        retry_needed = []
        quality_scores = []
        
        for risk_type, critic_result in critic_results.items():
            # ИСПРАВЛЕНИЕ: Проверяем что critic_result это dict, а не AgentTaskResult объект
            if isinstance(critic_result, dict):
                if (critic_result.get("status") == "completed" and 
                    critic_result.get("result_data") and 
                    "critic_evaluation" in critic_result["result_data"]):
                    
                    critic_eval = critic_result["result_data"]["critic_evaluation"]
                    quality_scores.append(critic_eval["quality_score"])
                    
                    # Проверяем, нужен ли повтор
                    if not critic_eval["is_acceptable"]:
                        current_retries = retry_count.get(risk_type, 0)
                        if current_retries < max_retries:
                            retry_needed.append(risk_type)
            else:
                # Если это не dict, возможно это объект с атрибутами
                try:
                    if (hasattr(critic_result, 'status') and 
                        critic_result.status == "completed" and 
                        hasattr(critic_result, 'result_data') and 
                        critic_result.result_data and 
                        "critic_evaluation" in critic_result.result_data):
                        
                        critic_eval = critic_result.result_data["critic_evaluation"]
                        quality_scores.append(critic_eval["quality_score"])
                        
                        if not critic_eval["is_acceptable"]:
                            current_retries = retry_count.get(risk_type, 0)
                            if current_retries < max_retries:
                                retry_needed.append(risk_type)
                except Exception as e:
                    self.graph_logger.log_workflow_step(
                        assessment_id,
                        "quality_check_warning",
                        f"Ошибка обработки critic_result для {risk_type}: {e}"
                    )
        
        # Логируем результаты проверки качества
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 5.0
        self.graph_logger.log_quality_check(
            assessment_id, 
            "overall", 
            avg_quality, 
            self.quality_threshold
        )
        
        # Обновляем состояние
        state["retry_needed"] = retry_needed
        state["average_quality"] = avg_quality
        
        # Определяем следующий шаг (на данном этапе всегда переходим к финализации)
        state["current_step"] = "finalization"
        
        return state
    
    @log_graph_node("retry_evaluation")
    async def _retry_evaluation_node(self, state: WorkflowState) -> WorkflowState:
        """ИСПРАВЛЕННАЯ повторная оценка с селективным перезапуском"""
        assessment_id = state["assessment_id"]
        retry_needed = state.get("retry_needed", [])
        retry_count = state.get("retry_count", {})
        
        if not retry_needed:
            # Если нет рисков для повтора, переходим к финализации
            state["current_step"] = "finalization"
            return state
        
        self.graph_logger.log_workflow_step(
            assessment_id,
            "retry_evaluation",
            f"Повторная оценка для: {retry_needed}"
        )
        
        # Обновляем счетчики повторов
        for risk_type in retry_needed:
            risk_key = risk_type if isinstance(risk_type, str) else risk_type.value
            retry_count[risk_key] = retry_count.get(risk_key, 0) + 1
            
            self.graph_logger.log_retry_logic(
                assessment_id,
                f"{risk_key}_evaluator", 
                retry_count[risk_key],
                self.max_retries
            )
        
        state["retry_count"] = retry_count
        
        # СЕЛЕКТИВНЫЙ ПЕРЕЗАПУСК только нужных агентов
        agent_profile = state.get("agent_profile", {})
        input_data = {"agent_profile": agent_profile}
        
        retry_tasks = []
        for risk_type in retry_needed:
            risk_key = risk_type if isinstance(risk_type, str) else risk_type.value
            
            # Маппинг строковых имен к RiskType enum
            risk_type_mapping = {
                "ethical": RiskType.ETHICAL,
                "stability": RiskType.STABILITY,
                "security": RiskType.SECURITY,
                "autonomy": RiskType.AUTONOMY,
                "regulatory": RiskType.REGULATORY,
                "social": RiskType.SOCIAL
            }
            
            risk_enum = risk_type_mapping.get(risk_key)
            if risk_enum and risk_enum in self.evaluators:
                self.graph_logger.log_workflow_step(
                    assessment_id, f"retry_{risk_key}",
                    f"Повторный запуск оценщика {risk_key}"
                )
                
                task = self.evaluators[risk_enum].run(input_data, assessment_id)
                retry_tasks.append((risk_key, task))
        
        # Выполняем повторные оценки
        if retry_tasks:
            try:
                # Запускаем с небольшими интервалами чтобы не перегружать LLM
                for i, (risk_key, task) in enumerate(retry_tasks):
                    if i > 0:
                        await asyncio.sleep(3)  # Пауза между запросами
                    
                    try:
                        result = await task
                        state.set_evaluation_result(risk_key, result)
                        
                        self.graph_logger.log_workflow_step(
                            assessment_id, f"retry_{risk_key}_completed",
                            f"Повторная оценка {risk_key}: {result.status}"
                        )
                        
                    except Exception as e:
                        # Если повтор тоже не удался
                        error_result = self._create_error_result(
                            f"{risk_key}_risk_evaluator", 
                            f"Ошибка повторной оценки: {e}"
                        )
                        state.set_evaluation_result(risk_key, error_result)
                        
                        self.graph_logger.log_workflow_step(
                            assessment_id, f"retry_{risk_key}_failed",
                            f"Повторная оценка {risk_key} не удалась: {e}"
                        )
            
            except Exception as e:
                self.graph_logger.log_workflow_step(
                    assessment_id, "retry_critical_error",
                    f"Критическая ошибка при повторных оценках: {e}"
                )
        
        # Очищаем список повторов и переходим обратно к критику
        state["retry_needed"] = []
        state["current_step"] = "critic_analysis"
        
        return state
    
    @log_graph_node("finalization")
    async def _finalization_node(self, state: WorkflowState) -> WorkflowState:
        """ПОЛНОСТЬЮ ИСПРАВЛЕННАЯ финализация с решением ошибки result_data"""
        assessment_id = state["assessment_id"]
        start_time = state.get("start_time", datetime.now())
        
        try:
            # Шаг 1: Собираем профиль агента
            agent_profile_data = state.get("agent_profile", {})
            
            # Создаем объект AgentProfile с защитой от ошибок
            try:
                agent_profile = AgentProfile(**agent_profile_data)
            except Exception as e:
                self.graph_logger.log_workflow_step(
                    assessment_id, 
                    "finalization_warning", 
                    f"Ошибка создания AgentProfile: {e}"
                )
                # Создаем минимальный профиль
                agent_profile = AgentProfile(
                    name=agent_profile_data.get("name", "Unknown"),
                    description=agent_profile_data.get("description", "Не указано"),
                    agent_type=agent_profile_data.get("agent_type", "other"),
                    llm_model=agent_profile_data.get("llm_model", "unknown"),
                    autonomy_level=agent_profile_data.get("autonomy_level", "supervised"),
                    target_audience=agent_profile_data.get("target_audience", "Неизвестно")
                )
            
            # Шаг 2: ИСПРАВЛЕННАЯ обработка результатов оценки
            evaluation_results = state.get_successful_evaluations()
            
            # Преобразуем в формат для extract_risk_evaluations_from_results
            formatted_evaluation_results = {}
            for risk_type, result in evaluation_results.items():
                # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Обеспечиваем правильную структуру
                if isinstance(result, dict):
                    # Если это уже dict, создаем объект с нужными атрибутами
                    formatted_result = type('AgentTaskResult', (), {
                        'status': ProcessingStatus.COMPLETED,
                        'result_data': result.get("result_data", {}),
                        'agent_name': result.get("agent_name", "unknown")
                    })()
                    
                    # Маппинг строковых имен к RiskType enum
                    risk_type_mapping = {
                        "ethical": RiskType.ETHICAL,
                        "stability": RiskType.STABILITY,
                        "security": RiskType.SECURITY,
                        "autonomy": RiskType.AUTONOMY,
                        "regulatory": RiskType.REGULATORY,
                        "social": RiskType.SOCIAL
                    }
                    
                    risk_enum = risk_type_mapping.get(risk_type)
                    if risk_enum:
                        formatted_evaluation_results[risk_enum] = formatted_result
            
            # Шаг 3: Извлекаем оценки рисков с максимальной защитой от ошибок
            risk_evaluations = {}
            try:
                if formatted_evaluation_results:
                    from ..agents.evaluator_agents import extract_risk_evaluations_from_results
                    risk_evaluations = extract_risk_evaluations_from_results(formatted_evaluation_results)
            except Exception as e:
                self.graph_logger.log_workflow_step(
                    assessment_id,
                    "finalization_warning", 
                    f"Ошибка извлечения оценок рисков: {e}"
                )
                # Создаем fallback оценки из сырых данных
                risk_evaluations = self._create_fallback_risk_evaluations(evaluation_results)
            
            # Шаг 4: Рассчитываем общие метрики с защитой от ошибок
            if risk_evaluations:
                try:
                    from ..agents.evaluator_agents import calculate_overall_risk_score, get_highest_risk_areas
                    overall_score, overall_level = calculate_overall_risk_score(risk_evaluations)
                    highest_risk_areas = get_highest_risk_areas(risk_evaluations, threshold=10)
                except Exception as e:
                    self.graph_logger.log_workflow_step(
                        assessment_id,
                        "finalization_warning",
                        f"Ошибка расчета метрик: {e}"
                    )
                    # Fallback расчет
                    overall_score, overall_level = self._calculate_fallback_metrics(evaluation_results)
                    highest_risk_areas = []
            else:
                # Если нет оценок, используем минимальные значения
                overall_score, overall_level = 6, "low" 
                highest_risk_areas = []
            
            # Шаг 5: Собираем рекомендации
            all_recommendations = []
            for risk_eval in risk_evaluations.values():
                if hasattr(risk_eval, 'recommendations') and risk_eval.recommendations:
                    all_recommendations.extend(risk_eval.recommendations)
            
            # Если нет рекомендаций из оценок, извлекаем из сырых данных
            if not all_recommendations:
                all_recommendations = self._extract_recommendations_from_raw_results(evaluation_results)
            
            # Дедуплицируем и берем топ рекомендации
            unique_recommendations = list(dict.fromkeys(all_recommendations))[:10]
            
            # Если нет рекомендаций, добавляем базовые
            if not unique_recommendations:
                unique_recommendations = [
                    "Регулярно проводить мониторинг работы агента",
                    "Обеспечить человеческий надзор за принятием решений",
                    "Документировать все изменения в конфигурации",
                    "Проводить периодические оценки рисков",
                    "Внедрить систему логирования всех действий агента"
                ]
            
            # Шаг 6: Создаем итоговую оценку
            processing_time = (datetime.now() - start_time).total_seconds()
            
            final_assessment_data = {
                "agent_profile": agent_profile.dict() if hasattr(agent_profile, 'dict') else agent_profile_data,
                "assessment_id": assessment_id,
                "risk_evaluations": {
                    str(k.value if hasattr(k, 'value') else k): v.dict() if hasattr(v, 'dict') else v 
                    for k, v in risk_evaluations.items()
                },
                "overall_risk_score": overall_score,
                "overall_risk_level": overall_level,
                "highest_risk_areas": [
                    str(area.value if hasattr(area, 'value') else area) 
                    for area in highest_risk_areas
                ],
                "priority_recommendations": unique_recommendations,
                "suggested_guardrails": self._generate_guardrails(overall_level, highest_risk_areas),
                "processing_time_seconds": processing_time,
                "quality_checks_passed": len(state.get("retry_needed", [])) == 0,
                "evaluation_summary": state.get_evaluation_summary()
            }
            
            # Шаг 7: Сохранение в базу данных (с защитой от ошибок)
            try:
                db_manager = await get_db_manager()
                profile_id = await db_manager.save_agent_profile(agent_profile)
                saved_assessment_id = assessment_id
            except Exception as e:
                self.graph_logger.log_workflow_step(
                    assessment_id,
                    "finalization_warning",
                    f"Ошибка сохранения в БД: {e}"
                )
                profile_id = None
                saved_assessment_id = assessment_id
            
            # Шаг 8: Обновляем состояние
            state.update({
                "final_assessment": final_assessment_data,
                "saved_assessment_id": saved_assessment_id,
                "profile_id": profile_id,
                "current_step": "completed",
                "processing_time": processing_time
            })
            
            # Шаг 9: Логируем завершение
            successful_evaluations = len(evaluation_results)
            total_possible = 6  # 6 типов рисков
            
            self.graph_logger.log_graph_completion(
                assessment_id, 
                processing_time, 
                successful_evaluations
            )
            
            self.graph_logger.log_workflow_step(
                assessment_id,
                "finalization_success",
                f"Финализация завершена: {successful_evaluations}/{total_possible} оценок, "
                f"общий риск: {overall_level} ({overall_score}), время: {processing_time:.2f}с"
            )
            
            return state
            
        except Exception as e:
            # Критическая ошибка финализации
            self.graph_logger.log_workflow_step(
                assessment_id,
                "finalization_critical_error",
                f"Критическая ошибка финализации: {e}"
            )
            
            # Создаем минимальный fallback результат
            fallback_assessment = {
                "assessment_id": assessment_id,
                "overall_risk_score": 12,
                "overall_risk_level": "medium",
                "priority_recommendations": [
                    "Провести повторную оценку рисков",
                    "Проверить качество входных данных",
                    "Обратиться к специалисту по безопасности ИИ"
                ],
                "error_message": f"Ошибка финализации: {str(e)}",
                "processing_time_seconds": (datetime.now() - start_time).total_seconds()
            }
            
            state.update({
                "final_assessment": fallback_assessment,
                "current_step": "error",
                "error_message": f"Ошибка финализации: {str(e)}"
            })
            
            return state
    
    # ===============================
# ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ ФИНАЛИЗАЦИИ
# ===============================

    def _create_fallback_risk_evaluations(self, evaluation_results: Dict[str, Any]) -> Dict[Any, Any]:
        """Создает fallback оценки рисков из сырых результатов"""
        from ..models.risk_models import RiskEvaluation, RiskType
        
        fallback_evaluations = {}
        
        risk_type_mapping = {
            "ethical": RiskType.ETHICAL,
            "stability": RiskType.STABILITY,
            "security": RiskType.SECURITY,
            "autonomy": RiskType.AUTONOMY,
            "regulatory": RiskType.REGULATORY,
            "social": RiskType.SOCIAL
        }
        
        for risk_type_str, result in evaluation_results.items():
            risk_type_enum = risk_type_mapping.get(risk_type_str)
            if risk_type_enum:
                try:
                    result_data = result.get("result_data", {})
                    risk_eval_data = result_data.get("risk_evaluation", {})
                    
                    if risk_eval_data:
                        # Пытаемся создать RiskEvaluation из данных
                        risk_evaluation = RiskEvaluation(
                            risk_type=risk_type_enum,
                            evaluator_agent=result.get("agent_name", "unknown"),
                            probability_score=risk_eval_data.get("probability_score", 3),
                            impact_score=risk_eval_data.get("impact_score", 3),
                            probability_reasoning=risk_eval_data.get("probability_reasoning", "Не указано"),
                            impact_reasoning=risk_eval_data.get("impact_reasoning", "Не указано"),
                            recommendations=risk_eval_data.get("recommendations", []),
                            confidence_level=risk_eval_data.get("confidence_level", 0.7)
                        )
                        fallback_evaluations[risk_type_enum] = risk_evaluation
                except Exception:
                    # Если не удалось создать, создаем минимальную оценку
                    fallback_evaluations[risk_type_enum] = RiskEvaluation(
                        risk_type=risk_type_enum,
                        evaluator_agent="fallback",
                        probability_score=3,
                        impact_score=3,
                        probability_reasoning="Fallback оценка из-за ошибки парсинга",
                        impact_reasoning="Fallback оценка из-за ошибки парсинга",
                        recommendations=["Провести повторную оценку"],
                        confidence_level=0.3
                    )
        
        return fallback_evaluations

    def _calculate_fallback_metrics(self, evaluation_results: Dict[str, Any]) -> tuple[int, str]:
        """Рассчитывает fallback метрики риска"""
        
        if not evaluation_results:
            return 6, "low"
        
        try:
            total_scores = []
            for result in evaluation_results.values():
                result_data = result.get("result_data", {})
                risk_eval = result_data.get("risk_evaluation", {})
                total_score = risk_eval.get("total_score")
                
                if total_score and isinstance(total_score, (int, float)):
                    total_scores.append(int(total_score))
            
            if total_scores:
                avg_score = sum(total_scores) // len(total_scores)
                max_score = max(total_scores)
                
                # Используем максимальный балл для определения уровня
                if max_score <= 6:
                    return max_score, "low"
                elif max_score <= 14:
                    return max_score, "medium"
                else:
                    return max_score, "high"
            
        except Exception:
            pass
        
        # Fallback - средний риск
        return 9, "medium"

    def _extract_recommendations_from_raw_results(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Извлекает рекомендации из сырых результатов оценки"""
        
        recommendations = []
        
        for result in evaluation_results.values():
            try:
                result_data = result.get("result_data", {})
                risk_eval = result_data.get("risk_evaluation", {})
                risk_recommendations = risk_eval.get("recommendations", [])
                
                if isinstance(risk_recommendations, list):
                    recommendations.extend(risk_recommendations)
            except Exception:
                continue
        
        return recommendations

    def _generate_guardrails(self, risk_level: str, highest_risk_areas: List[Any]) -> List[str]:
        """Генерирует рекомендации по защитным мерам на основе уровня риска"""
        
        base_guardrails = [
            "Внедрить систему логирования всех действий агента",
            "Настроить мониторинг аномального поведения",
            "Обеспечить регулярное обновление системы безопасности"
        ]
        
        if risk_level == "high":
            base_guardrails.extend([
                "Требовать человеческое подтверждение для критических решений",
                "Ограничить доступ к чувствительным данным",
                "Проводить ежедневный аудит действий агента",
                "Внедрить автоматическое отключение при аномалиях"
            ])
        elif risk_level == "medium":
            base_guardrails.extend([
                "Настроить уведомления о подозрительной активности",
                "Проводить еженедельный мониторинг",
                "Ограничить права доступа агента"
            ])
        
        # Добавляем специфичные меры на основе областей высокого риска
        for risk_area in highest_risk_areas:
            area_str = str(risk_area.value if hasattr(risk_area, 'value') else risk_area)
            
            if area_str == "ethical":
                base_guardrails.append("Внедрить фильтры этического контента")
            elif area_str == "security":
                base_guardrails.append("Усилить меры кибербезопасности")
            elif area_str == "autonomy":
                base_guardrails.append("Снизить уровень автономности агента")
        
        return list(dict.fromkeys(base_guardrails))  # Удаляем дубликаты

    @log_graph_node("error_handling")
    async def _error_handling_node(self, state: WorkflowState) -> WorkflowState:
        """Обработка ошибок"""
        assessment_id = state["assessment_id"]
        error_message = state.get("error_message", "Неизвестная ошибка")
        
        self.graph_logger.log_workflow_step(
            assessment_id,
            "error_handling",
            f"Обработка ошибки: {error_message}"
        )
        
        # Логируем в базу данных
        try:
            db_manager = await get_db_manager()
            await db_manager.log_processing_step(
                assessment_id=assessment_id,
                agent_name="workflow",
                task_type="error_handling",
                status=ProcessingStatus.FAILED,
                error_message=error_message
            )
        except Exception:
            pass  # Не критично если не удалось залогировать
        
        state["current_step"] = "failed"
        return state
    
    # ===============================
    # Условные переходы
    # ===============================
    
    @log_conditional_edge_func("quality_check_router")
    def _quality_check_router(self, state: WorkflowState) -> Literal["retry", "finalize", "error"]:
        """Маршрутизация после проверки качества"""
        
        current_step = state.get("current_step")
        retry_needed = state.get("retry_needed", [])
        error_message = state.get("error_message")
        
        if error_message or current_step == "error":
            return "error"
        elif retry_needed:
            return "retry"
        else:
            return "finalize"
    
    # ===============================
    # Публичные методы
    # ===============================
    
    async def run_assessment(
        self,
        source_files: List[str],
        agent_name: Optional[str] = None,
        assessment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Запуск полной оценки рисков ИИ-агента
        
        Args:
            source_files: Список файлов/папок для анализа
            agent_name: Предварительное имя агента
            assessment_id: ID оценки (генерируется автоматически если не указан)
            
        Returns:
            Результат оценки рисков
        """
        
        # Инициализируем начальное состояние
        initial_state = WorkflowState(
            source_files=source_files,
            preliminary_agent_name=agent_name or "Unknown_Agent",
            assessment_id=assessment_id
        )
        
        # Запускаем граф
        try:
            final_state = await self.graph.ainvoke(initial_state.dict())
            
            return {
                "success": True,
                "assessment_id": final_state.get("assessment_id"),
                "final_assessment": final_state.get("final_assessment"),
                "processing_time": final_state.get("processing_time"),
                "current_step": final_state.get("current_step"),
                "saved_assessment_id": final_state.get("saved_assessment_id")
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "assessment_id": initial_state.assessment_id
            }
    
    async def get_assessment_status(self, assessment_id: str) -> Dict[str, Any]:
        """Получение статуса оценки"""
        try:
            db_manager = await get_db_manager()
            logs = await db_manager.get_processing_logs(assessment_id)
            
            return {
                "assessment_id": assessment_id,
                "logs": logs,
                "total_steps": len(logs)
            }
        except Exception as e:
            return {
                "assessment_id": assessment_id,
                "error": str(e)
            }
    
    def get_graph_visualization(self) -> str:
        """Получение текстового представления графа"""
        # TODO: Реализовать визуализацию графа
        return "Граф оценки рисков ИИ-агентов (визуализация в разработке)"


# ===============================
# Фабрики и утилиты
# ===============================

def create_risk_assessment_workflow(
    llm_base_url: str = "http://127.0.0.1:1234",
    llm_model: str = "qwen3-4b",
    quality_threshold: float = 7.0,
    max_retries: int = 3
) -> RiskAssessmentWorkflow:
    """
    Создание workflow для оценки рисков
    
    Args:
        llm_base_url: URL LLM сервера
        llm_model: Модель LLM
        quality_threshold: Порог качества для критика
        max_retries: Максимум повторов
        
    Returns:
        Настроенный workflow
    """
    return RiskAssessmentWorkflow(
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        quality_threshold=quality_threshold,
        max_retries=max_retries
    )


def create_workflow_from_env() -> RiskAssessmentWorkflow:
    """Создание workflow из переменных окружения"""
    import os
    
    return create_risk_assessment_workflow(
        llm_base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234"),
        llm_model=os.getenv("LLM_MODEL", "qwen3-4b"),
        quality_threshold=float(os.getenv("QUALITY_THRESHOLD", "7.0")),
        max_retries=int(os.getenv("MAX_RETRY_COUNT", "3"))
    )


# Экспорт
__all__ = [
    "RiskAssessmentWorkflow",
    "create_risk_assessment_workflow", 
    "create_workflow_from_env"
]