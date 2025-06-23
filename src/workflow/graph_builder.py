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
        """Добавление узлов в граф"""
        
        # 1. Инициализация
        workflow.add_node("initialization", self._initialization_node)
        
        # 2. Профилирование
        profiler_node = create_profiler_node_function(self.profiler)
        workflow.add_node("profiling", log_graph_node("profiling")(profiler_node))
        
        # 3. Подготовка к оценке
        workflow.add_node("evaluation_preparation", self._evaluation_preparation_node)
        
        # 4. ИСПРАВЛЕННАЯ параллельная оценка рисков - используем безопасные узлы
        evaluator_nodes = create_evaluator_nodes_for_langgraph_safe(self.evaluators)
        for node_name, node_func in evaluator_nodes.items():
            workflow.add_node(node_name, log_graph_node(node_name)(node_func))
        
        # 5. Сбор результатов оценки
        workflow.add_node("evaluation_collection", self._evaluation_collection_node)
        
        # 6. Критический анализ
        critic_node = create_critic_node_function_fixed(self.critic)
        workflow.add_node("critic_analysis", log_graph_node("critic_analysis")(critic_node))
        
        # 7. Проверка качества и решение о повторах
        workflow.add_node("quality_check", self._quality_check_node)
        
        # 8. Повторная оценка (при необходимости)
        workflow.add_node("retry_evaluation", self._retry_evaluation_node)
        
        # 9. Финализация результата
        workflow.add_node("finalization", self._finalization_node)
        
        # 10. Обработка ошибок
        workflow.add_node("error_handling", self._error_handling_node)
    
    def _add_edges(self, workflow: StateGraph):
        """Добавление рёбер между узлами"""
        
        # Последовательные переходы
        workflow.add_edge("initialization", "profiling")
        workflow.add_edge("profiling", "evaluation_preparation")
        
        # Параллельная оценка рисков
        risk_evaluator_nodes = [
            "ethical_evaluator_node",
            "stability_evaluator_node", 
            "security_evaluator_node",
            "autonomy_evaluator_node",
            "regulatory_evaluator_node",
            "social_evaluator_node"
        ]
        
        # Из подготовки к параллельной оценке
        for node_name in risk_evaluator_nodes:
            workflow.add_edge("evaluation_preparation", node_name)
        
        # Из параллельной оценки к сбору результатов
        for node_name in risk_evaluator_nodes:
            workflow.add_edge(node_name, "evaluation_collection")
        
        # Основной поток
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
        
        # Из повторной оценки обратно к критику
        workflow.add_edge("retry_evaluation", "critic_analysis")
        
        # Завершение
        workflow.add_edge("finalization", END)
        workflow.add_edge("error_handling", END)
    
    # ===============================
    # Узлы графа
    # ===============================
    
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
    
    @log_graph_node("evaluation_collection")
    async def _evaluation_collection_node(self, state: WorkflowState) -> WorkflowState:
        """Сбор результатов параллельной оценки в единую структуру"""
        assessment_id = state["assessment_id"]
        
        # Собираем результаты из отдельных полей
        evaluation_results = state.get_evaluation_results()
        
        # Проверяем, что все оценки завершены
        expected_risk_types = ["ethical", "stability", "security", "autonomy", "regulatory", "social"]
        completed_evaluations = []
        failed_evaluations = []
        
        for risk_type in expected_risk_types:
            result = evaluation_results.get(risk_type)
            if result and result.get("status") == "completed":
                completed_evaluations.append(risk_type)
            else:
                failed_evaluations.append(risk_type)
        
        self.graph_logger.log_workflow_step(
            assessment_id,
            "evaluation_collection",
            f"Завершено: {len(completed_evaluations)}, Ошибки: {len(failed_evaluations)}"
        )
        
        # Если есть критические ошибки, переходим к обработке ошибок
        if len(failed_evaluations) > len(expected_risk_types) // 2:
            state.current_step = "error"
            state.error_message = f"Слишком много неудачных оценок: {failed_evaluations}"
            return state
        
        # Сохраняем собранные результаты для совместимости с остальным кодом
        state.update({"evaluation_results": evaluation_results})
        state.current_step = "critic_analysis"
        
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
        """Повторная оценка для рисков с низким качеством"""
        assessment_id = state["assessment_id"]
        retry_needed = state.get("retry_needed", [])
        retry_count = state.get("retry_count", {})
        
        # Обновляем счетчики повторов
        for risk_type in retry_needed:
            risk_key = risk_type.value
            retry_count[risk_key] = retry_count.get(risk_key, 0) + 1
            
            self.graph_logger.log_retry_logic(
                assessment_id,
                f"{risk_type.value}_evaluator", 
                retry_count[risk_key],
                self.max_retries
            )
        
        state["retry_count"] = retry_count
        
        # Запускаем повторную оценку только для нужных рисков
        # Для простоты, перезапускаем весь блок оценки
        # В более сложной реализации можно селективно перезапускать агентов
        state["current_step"] = "evaluation_preparation"
        
        return state
    
    @log_graph_node("finalization")
    async def _finalization_node(self, state: WorkflowState) -> WorkflowState:
        """Финализация результата и сохранение в БД - ОБНОВЛЕННАЯ ВЕРСИЯ"""
        assessment_id = state["assessment_id"]
        start_time = state.get("start_time", datetime.now())
        
        try:
            # Собираем итоговый результат
            agent_profile_data = state.get("agent_profile", {})
            critic_results = state.get("critic_results", {})
            
            # Получаем результаты оценки из нового формата
            evaluation_results = state.get_evaluation_results()
            
            # Преобразуем в старый формат для совместимости
            formatted_evaluation_results = {}
            for risk_type, result in evaluation_results.items():
                if result and result.get("status") == "completed":
                    formatted_evaluation_results[risk_type] = result
            
            # Создаем объект AgentProfile
            try:
                agent_profile = AgentProfile(**agent_profile_data)
            except Exception as e:
                self.graph_logger.log_workflow_step(
                    assessment_id, 
                    "finalization_error", 
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
            
            # Извлекаем оценки рисков с защитой от ошибок
            try:
                from ..agents.evaluator_agents import extract_risk_evaluations_from_results
                risk_evaluations = extract_risk_evaluations_from_results(formatted_evaluation_results)
            except Exception as e:
                self.graph_logger.log_workflow_step(
                    assessment_id,
                    "finalization_warning", 
                    f"Ошибка извлечения оценок: {e}"
                )
                risk_evaluations = {}
            
            # Рассчитываем общие метрики с защитой от ошибок
            if risk_evaluations:
                try:
                    from ..agents.evaluator_agents import calculate_overall_risk_score, get_highest_risk_areas
                    overall_score, overall_level = calculate_overall_risk_score(risk_evaluations)
                    highest_risk_areas = get_highest_risk_areas(risk_evaluations)
                except Exception as e:
                    self.graph_logger.log_workflow_step(
                        assessment_id,
                        "finalization_warning",
                        f"Ошибка расчета метрик: {e}"
                    )
                    overall_score, overall_level = 12, "medium"  # Дефолтные значения
                    highest_risk_areas = []
            else:
                overall_score, overall_level = 6, "low"  # Минимальный риск если нет оценок
                highest_risk_areas = []
            
            # Собираем рекомендации
            all_recommendations = []
            for risk_eval in risk_evaluations.values():
                if hasattr(risk_eval, 'recommendations'):
                    all_recommendations.extend(risk_eval.recommendations)
            
            # Дедуплицируем и берем топ рекомендации
            unique_recommendations = list(dict.fromkeys(all_recommendations))[:10]
            
            # Если нет рекомендаций, добавляем базовые
            if not unique_recommendations:
                unique_recommendations = [
                    "Регулярно проводить мониторинг работы агента",
                    "Обеспечить человеческий надзор за принятием решений",
                    "Документировать все изменения в конфигурации"
                ]
            
            # Создаем итоговую оценку
            processing_time = (datetime.now() - start_time).total_seconds()
            
            final_assessment_data = {
                "agent_profile": agent_profile.dict(),
                "assessment_id": assessment_id,
                "risk_evaluations": {k: v.dict() for k, v in risk_evaluations.items()},
                "overall_risk_score": overall_score,
                "overall_risk_level": overall_level,
                "highest_risk_areas": highest_risk_areas,
                "priority_recommendations": unique_recommendations,
                "suggested_guardrails": [],  # Можно добавить логику извлечения
                "processing_time_seconds": processing_time,
                "quality_checks_passed": len(state.get("retry_needed", [])) == 0
            }
            
            # Пытаемся сохранить в базу данных
            try:
                db_manager = await get_db_manager()
                profile_id = await db_manager.save_agent_profile(agent_profile)
                # Для сохранения assessment нужно создать объект, но упростим
                saved_assessment_id = assessment_id  # Упрощаем пока
            except Exception as e:
                self.graph_logger.log_workflow_step(
                    assessment_id,
                    "finalization_warning",
                    f"Ошибка сохранения в БД: {e}"
                )
                profile_id = None
                saved_assessment_id = assessment_id
            
            # Обновляем состояние
            state.update({
                "final_assessment": final_assessment_data,
                "saved_assessment_id": saved_assessment_id,
                "profile_id": profile_id,
                "current_step": "completed",
                "processing_time": processing_time
            })
            
            # Логируем завершение
            total_evaluations = len([r for r in evaluation_results.values() if r])
            self.graph_logger.log_graph_completion(assessment_id, processing_time, total_evaluations)
            
            return state
            
        except Exception as e:
            self.graph_logger.log_workflow_step(
                assessment_id,
                "finalization_error",
                f"Критическая ошибка финализации: {e}"
            )
            
            state.update({
                "current_step": "error",
                "error_message": f"Ошибка финализации: {str(e)}"
            })
            return state
    
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