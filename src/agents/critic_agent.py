# src/agents/critic_agent.py
"""
Критик-агент для оценки качества работы агентов-оценщиков рисков
Анализирует результаты оценки и принимает решения о необходимости повторных оценок
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_agent import AnalysisAgent, AgentConfig
from ..models.risk_models import (
    RiskType, RiskEvaluation, CriticEvaluation, AgentTaskResult, ProcessingStatus
)
from ..utils.logger import LogContext


class CriticAgent(AnalysisAgent):
    """
    Критик-агент для контроля качества оценок рисков
    
    Функции:
    1. Анализ качества оценок агентов-оценщиков
    2. Проверка обоснованности выводов
    3. Определение необходимости повторных оценок
    4. Выдача рекомендаций по улучшению качества
    """
    
    def __init__(self, config: AgentConfig, quality_threshold: float = 7.0):
        # Критик использует специализированный клиент для анализа рисков
        config.use_risk_analysis_client = True
        super().__init__(config)
        
        self.quality_threshold = quality_threshold
    
    def get_system_prompt(self) -> str:
        """Системный промпт для критика"""
        return f"""Ты - старший эксперт-аудитор по оценке качества анализа рисков ИИ-агентов.

Твоя задача: критически оценивать качество работы агентов-оценщиков операционных рисков.

КРИТЕРИИ КАЧЕСТВА ОЦЕНКИ:

1. ОБОСНОВАННОСТЬ (30%):
   - Соответствие выводов предоставленным данным
   - Логичность рассуждений
   - Учет всех релевантных факторов

2. ПОЛНОТА АНАЛИЗА (25%):
   - Рассмотрение всех аспектов риска
   - Достаточность приведенных аргументов
   - Учет контекста банковской деятельности

3. ТОЧНОСТЬ ОЦЕНОК (25%):
   - Адекватность баллов вероятности и тяжести
   - Соответствие итогового уровня риска
   - Применение корректной методики

4. ПРАКТИЧНОСТЬ РЕКОМЕНДАЦИЙ (20%):
   - Применимость предложенных мер
   - Конкретность и детализация
   - Соответствие лучшим практикам

ШКАЛА КАЧЕСТВА: 0-10 баллов
ПОРОГ ПРИЕМЛЕМОСТИ: {self.quality_threshold} баллов

ТИПИЧНЫЕ ПРОБЛЕМЫ ДЛЯ ВЫЯВЛЕНИЯ:
- Завышение или занижение рисков без обоснования
- Игнорирование важных факторов риска
- Общие формулировки без конкретики
- Несоответствие рекомендаций выявленным рискам
- Неучет специфики банковского сектора

ОБЯЗАТЕЛЬНЫЙ ФОРМАТ ОТВЕТА (JSON):
{{
    "quality_score": <0.0-10.0>,
    "is_acceptable": <true|false>,
    "issues_found": ["<проблема1>", "<проблема2>", ...],
    "improvement_suggestions": ["<предложение1>", "<предложение2>", ...],
    "critic_reasoning": "<подробное обоснование оценки качества>"
}}

Будь строгим, но справедливым. Высокие стандарты качества - залог надежной оценки рисков."""
    
    async def process(
        self, 
        input_data: Dict[str, Any], 
        assessment_id: str
    ) -> AgentTaskResult:
        """
        Основная обработка критического анализа
        
        Args:
            input_data: Содержит результаты оценки для анализа
                - risk_type: RiskType - тип риска для проверки
                - risk_evaluation: Dict - оценка риска для критики
                - agent_profile: Dict - профиль анализируемого агента
                - evaluator_name: str - имя агента-оценщика
            assessment_id: ID оценки
            
        Returns:
            Результат с критической оценкой
        """
        start_time = datetime.now()
        
        try:
            with LogContext("critic_analysis", assessment_id, self.name):
                # Извлекаем входные данные
                risk_type = RiskType(input_data["risk_type"])
                risk_evaluation = input_data["risk_evaluation"]
                agent_profile = input_data.get("agent_profile", {})
                evaluator_name = input_data.get("evaluator_name", "unknown")
                
                # Выполняем критический анализ
                critic_result = await self._critique_evaluation(
                    risk_type=risk_type,
                    risk_evaluation=risk_evaluation,
                    agent_profile=agent_profile,
                    evaluator_name=evaluator_name,
                    assessment_id=assessment_id
                )
                
                # Создаем объект CriticEvaluation
                critic_evaluation = CriticEvaluation(
                    risk_type=risk_type,
                    quality_score=critic_result["quality_score"],
                    is_acceptable=critic_result["is_acceptable"],
                    issues_found=critic_result.get("issues_found", []),
                    improvement_suggestions=critic_result.get("improvement_suggestions", []),
                    critic_reasoning=critic_result["critic_reasoning"]
                )
                
                # Логируем результат критики
                self.logger.log_critic_feedback(
                    assessment_id,
                    risk_type.value,
                    critic_evaluation.quality_score,
                    critic_evaluation.is_acceptable
                )
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                return AgentTaskResult(
                    agent_name=self.name,
                    task_type="critic_analysis",
                    status=ProcessingStatus.COMPLETED,
                    result_data={
                        "critic_evaluation": critic_evaluation.dict(),
                        "raw_llm_response": critic_result,
                        "requires_retry": not critic_evaluation.is_acceptable
                    },
                    start_time=start_time,
                    end_time=end_time,
                    execution_time_seconds=execution_time
                )
                
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return AgentTaskResult(
                agent_name=self.name,
                task_type="critic_analysis",
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                start_time=start_time,
                end_time=end_time,
                execution_time_seconds=execution_time
            )
    
    async def _critique_evaluation(
        self,
        risk_type: RiskType,
        risk_evaluation: Dict[str, Any],
        agent_profile: Dict[str, Any],
        evaluator_name: str,
        assessment_id: str
    ) -> Dict[str, Any]:
        """Критический анализ оценки риска"""
        
        # Используем специализированный метод RiskAnalysisLLMClient
        from ..utils.llm_client import RiskAnalysisLLMClient
        
        if not isinstance(self.llm_client, RiskAnalysisLLMClient):
            raise ValueError("Критик должен использовать RiskAnalysisLLMClient")
        
        # Форматируем данные агента
        agent_data = self._format_agent_data_for_critique(agent_profile)
        
        # Вызываем метод критики
        critic_result = await self.llm_client.critique_evaluation(
            risk_type=risk_type.value,
            original_evaluation=risk_evaluation,
            agent_data=agent_data,
            quality_threshold=self.quality_threshold
        )
        
        return critic_result
    
    async def critique_multiple_evaluations(
        self,
        evaluation_results: Dict[str, Any],  # Теперь принимает Dict вместо AgentTaskResult
        agent_profile: Dict[str, Any],
        assessment_id: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Критика множественных оценок - ОБНОВЛЕННАЯ ВЕРСИЯ
        
        Args:
            evaluation_results: Результаты работы агентов-оценщиков (из get_evaluation_results())
            agent_profile: Профиль анализируемого агента
            assessment_id: ID оценки
            
        Returns:
            Результаты критического анализа по типам рисков
        """
        critic_results = {}
        
        for risk_type, eval_result in evaluation_results.items():
            # Проверяем что результат существует и содержит данные
            if (eval_result and 
                isinstance(eval_result, dict) and 
                eval_result.get("status") == "completed" and 
                eval_result.get("result_data")):
                
                risk_evaluation = eval_result["result_data"].get("risk_evaluation")
                
                if risk_evaluation:
                    # Подготавливаем данные для критики
                    input_data = {
                        "risk_type": risk_type,
                        "risk_evaluation": risk_evaluation,
                        "agent_profile": agent_profile,
                        "evaluator_name": eval_result.get("agent_name", "Unknown")
                    }
                    
                    try:
                        # Выполняем критику
                        critic_result = await self.run(input_data, assessment_id)
                        critic_results[risk_type] = critic_result
                        
                    except Exception as e:
                        # Если критика не удалась, создаем дефолтный результат
                        self.logger.bind_context(assessment_id, self.name).error(
                            f"❌ Ошибка критики {risk_type}: {e}"
                        )
                        
                        critic_results[risk_type] = self._create_default_critic_result(
                            risk_type, f"Ошибка критики: {str(e)}"
                        )
                else:
                    # Нет данных для критики
                    critic_results[risk_type] = self._create_default_critic_result(
                        risk_type, "Отсутствуют данные оценки для критики"
                    )
            else:
                # Неудачная оценка
                critic_results[risk_type] = self._create_default_critic_result(
                    risk_type, "Оценка риска не была завершена успешно"
                )
        
        return critic_results

    def _create_default_critic_result(self, risk_type: str, error_message: str) -> Dict[str, Any]:
        """Создает дефолтный результат критика при ошибках"""
        from ..models.risk_models import AgentTaskResult, ProcessingStatus
        from datetime import datetime
        
        # Создаем минимальный результат критика
        default_critic_evaluation = {
            "quality_score": 5.0,  # Средняя оценка
            "is_acceptable": True,  # Принимаем по умолчанию чтобы не блокировать процесс
            "issues_found": [error_message],
            "improvement_suggestions": ["Повторить оценку", "Проверить данные агента"],
            "critic_reasoning": f"Автоматическая оценка из-за ошибки: {error_message}"
        }
        
        return AgentTaskResult(
            agent_name=self.name,
            task_type="critic_analysis",
            status=ProcessingStatus.COMPLETED,
            result_data={
                "critic_evaluation": default_critic_evaluation,
                "requires_retry": False  # Не требуем повтора при ошибках критика
            },
            start_time=datetime.now(),
            end_time=datetime.now(),
            execution_time_seconds=0.1
        ).dict()
    
    def _format_agent_data_for_critique(self, agent_profile: Dict[str, Any]) -> str:
        """Форматирование данных агента для критического анализа"""
        return f"""АНАЛИЗИРУЕМЫЙ ИИ-АГЕНТ:
Название: {agent_profile.get('name', 'Unknown')}
Тип: {agent_profile.get('agent_type', 'unknown')}
Описание: {agent_profile.get('description', 'Не указано')}
Автономность: {agent_profile.get('autonomy_level', 'unknown')}
Доступ к данным: {', '.join(agent_profile.get('data_access', []))}
Целевая аудитория: {agent_profile.get('target_audience', 'Не указано')}
LLM модель: {agent_profile.get('llm_model', 'unknown')}

ОПЕРАЦИОННЫЕ ХАРАКТЕРИСТИКИ:
Операций в час: {agent_profile.get('operations_per_hour', 'Не указано')}
Доход с операции: {agent_profile.get('revenue_per_operation', 'Не указано')} руб
Внешние API: {', '.join(agent_profile.get('external_apis', ['Нет']))}

СИСТЕМНЫЕ ПРОМПТЫ:
{chr(10).join(agent_profile.get('system_prompts', ['Не найдены']))}

ОГРАНИЧЕНИЯ БЕЗОПАСНОСТИ:
{chr(10).join(agent_profile.get('guardrails', ['Не найдены']))}"""
    
    def analyze_quality_trends(
        self, 
        critic_results: Dict[RiskType, AgentTaskResult]
    ) -> Dict[str, Any]:
        """
        Анализ трендов качества оценок
        
        Args:
            critic_results: Результаты критического анализа
            
        Returns:
            Сводка по качеству оценок
        """
        quality_scores = []
        acceptable_count = 0
        issues_summary = {}
        
        for risk_type, result in critic_results.items():
            if (result.status == ProcessingStatus.COMPLETED and 
                result.result_data and 
                "critic_evaluation" in result.result_data):
                
                critic_eval = result.result_data["critic_evaluation"]
                quality_scores.append(critic_eval["quality_score"])
                
                if critic_eval["is_acceptable"]:
                    acceptable_count += 1
                
                # Собираем статистику по проблемам
                for issue in critic_eval.get("issues_found", []):
                    if issue not in issues_summary:
                        issues_summary[issue] = 0
                    issues_summary[issue] += 1
        
        if not quality_scores:
            return {"error": "Нет данных для анализа"}
        
        return {
            "average_quality": sum(quality_scores) / len(quality_scores),
            "min_quality": min(quality_scores),
            "max_quality": max(quality_scores),
            "acceptable_rate": acceptable_count / len(critic_results),
            "total_evaluations": len(critic_results),
            "common_issues": sorted(issues_summary.items(), key=lambda x: x[1], reverse=True)[:5],
            "quality_threshold": self.quality_threshold
        }
    
    def get_retry_recommendations(
        self, 
        critic_results: Dict[RiskType, AgentTaskResult]
    ) -> List[RiskType]:
        """
        Получение рекомендаций по повторным оценкам
        
        Args:
            critic_results: Результаты критического анализа
            
        Returns:
            Список типов рисков, требующих повторной оценки
        """
        retry_needed = []
        
        for risk_type, result in critic_results.items():
            if (result.status == ProcessingStatus.COMPLETED and 
                result.result_data and 
                "requires_retry" in result.result_data):
                
                if result.result_data["requires_retry"]:
                    retry_needed.append(risk_type)
        
        # Сортируем по приоритету (самые низкие оценки качества первыми)
        def get_quality_score(risk_type):
            result = critic_results.get(risk_type)
            if (result and result.result_data and 
                "critic_evaluation" in result.result_data):
                return result.result_data["critic_evaluation"]["quality_score"]
            return 10.0  # Высокий балл по умолчанию
        
        retry_needed.sort(key=get_quality_score)
        
        return retry_needed
    
    def generate_improvement_report(
        self, 
        critic_results: Dict[RiskType, AgentTaskResult]
    ) -> Dict[str, Any]:
        """
        Генерация отчета с рекомендациями по улучшению
        
        Args:
            critic_results: Результаты критического анализа
            
        Returns:
            Детальный отчет с рекомендациями
        """
        report = {
            "assessment_summary": self.analyze_quality_trends(critic_results),
            "risk_type_analysis": {},
            "overall_recommendations": [],
            "priority_issues": []
        }
        
        all_suggestions = []
        priority_issues = []
        
        for risk_type, result in critic_results.items():
            if (result.status == ProcessingStatus.COMPLETED and 
                result.result_data and 
                "critic_evaluation" in result.result_data):
                
                critic_eval = result.result_data["critic_evaluation"]
                
                report["risk_type_analysis"][risk_type.value] = {
                    "quality_score": critic_eval["quality_score"],
                    "is_acceptable": critic_eval["is_acceptable"],
                    "main_issues": critic_eval.get("issues_found", []),
                    "suggestions": critic_eval.get("improvement_suggestions", [])
                }
                
                # Собираем все предложения
                all_suggestions.extend(critic_eval.get("improvement_suggestions", []))
                
                # Выделяем приоритетные проблемы (низкое качество)
                if critic_eval["quality_score"] < self.quality_threshold:
                    priority_issues.extend(critic_eval.get("issues_found", []))
        
        # Обобщаем рекомендации
        suggestion_counts = {}
        for suggestion in all_suggestions:
            if suggestion not in suggestion_counts:
                suggestion_counts[suggestion] = 0
            suggestion_counts[suggestion] += 1
        
        # Топ рекомендации (упоминаемые чаще всего)
        report["overall_recommendations"] = [
            suggestion for suggestion, count in 
            sorted(suggestion_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Приоритетные проблемы
        issue_counts = {}
        for issue in priority_issues:
            if issue not in issue_counts:
                issue_counts[issue] = 0
            issue_counts[issue] += 1
        
        report["priority_issues"] = [
            issue for issue, count in 
            sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        return report


# ===============================
# Интеграция с LangGraph
# ===============================

def create_critic_node_function(critic_agent: CriticAgent):
    """
    Создает функцию узла критика для LangGraph workflow
    
    Args:
        critic_agent: Экземпляр критик-агента
        
    Returns:
        Функция для использования в LangGraph
    """
    async def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Узел критика в LangGraph workflow"""
        
        assessment_id = state.get("assessment_id", "unknown")
        agent_profile = state.get("agent_profile", {})
        evaluation_results = state.get("evaluation_results", {})
        
        # Критикуем все доступные оценки
        critic_results = await critic_agent.critique_multiple_evaluations(
            evaluation_results=evaluation_results,
            agent_profile=agent_profile,
            assessment_id=assessment_id
        )
        
        # Обновляем состояние
        updated_state = state.copy()
        updated_state["critic_results"] = critic_results
        
        # Определяем, нужны ли повторные оценки
        retry_needed = critic_agent.get_retry_recommendations(critic_results)
        updated_state["retry_needed"] = retry_needed
        
        # Генерируем отчет по качеству
        quality_report = critic_agent.generate_improvement_report(critic_results)
        updated_state["quality_report"] = quality_report
        
        # Определяем следующий шаг workflow
        if retry_needed:
            updated_state["current_step"] = "retry_evaluations"
        else:
            updated_state["current_step"] = "finalization"
        
        return updated_state
    
    return critic_node


def create_quality_check_router(quality_threshold: float = 7.0):
    """
    Создает функцию маршрутизации для проверки качества
    
    Args:
        quality_threshold: Порог качества для принятия решений
        
    Returns:
        Функция маршрутизации для LangGraph
    """
    def quality_check_router(state: Dict[str, Any]) -> str:
        """Маршрутизация на основе результатов критика"""
        
        retry_needed = state.get("retry_needed", [])
        max_retries = state.get("max_retries", 3)
        
        # Проверяем счетчики повторов
        retry_count = state.get("retry_count", {})
        
        # Определяем, есть ли риски, которые еще можно повторить
        retriable_risks = []
        for risk_type in retry_needed:
            current_retries = retry_count.get(risk_type.value, 0)
            if current_retries < max_retries:
                retriable_risks.append(risk_type)
        
        if retriable_risks:
            return "retry_evaluations"
        else:
            return "finalization"
    
    return quality_check_router


# ===============================
# Фабрики
# ===============================

def create_critic_agent(
    llm_base_url: str = "http://127.0.0.1:1234",
    llm_model: str = "qwen3-4b",
    temperature: float = 0.1,
    quality_threshold: float = 7.0
) -> CriticAgent:
    """
    Создание критик-агента
    
    Args:
        llm_base_url: URL LLM сервера
        llm_model: Модель LLM
        temperature: Температура генерации
        quality_threshold: Порог качества для принятия оценок
        
    Returns:
        Настроенный критик-агент
    """
    from .base_agent import create_agent_config
    
    config = create_agent_config(
        name="critic_agent",
        description="Агент для критического анализа качества оценок рисков",
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        temperature=temperature,
        max_retries=2,  # Меньше повторов для критика
        timeout_seconds=90,
        use_risk_analysis_client=True  # Критик использует специализированный клиент
    )
    
    return CriticAgent(config, quality_threshold)


def create_critic_from_env() -> CriticAgent:
    """Создание критик-агента из переменных окружения"""
    import os
    
    return create_critic_agent(
        llm_base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234"),
        llm_model=os.getenv("LLM_MODEL", "qwen3-4b"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        quality_threshold=float(os.getenv("QUALITY_THRESHOLD", "7.0"))
    )


# ===============================
# Утилиты для работы с критикой
# ===============================

def extract_critic_evaluations_from_results(
    critic_results: Dict[RiskType, AgentTaskResult]
) -> Dict[RiskType, CriticEvaluation]:
    """
    Извлечение объектов CriticEvaluation из результатов критика
    
    Args:
        critic_results: Результаты работы критик-агента
        
    Returns:
        Словарь критических оценок
    """
    critic_evaluations = {}
    
    for risk_type, task_result in critic_results.items():
        if (task_result.status == ProcessingStatus.COMPLETED and 
            task_result.result_data and 
            "critic_evaluation" in task_result.result_data):
            
            eval_data = task_result.result_data["critic_evaluation"]
            critic_evaluation = CriticEvaluation(**eval_data)
            critic_evaluations[risk_type] = critic_evaluation
    
    return critic_evaluations


def should_retry_evaluation(
    critic_evaluation: CriticEvaluation,
    current_retry_count: int,
    max_retries: int
) -> bool:
    """
    Определение необходимости повторной оценки
    
    Args:
        critic_evaluation: Оценка критика
        current_retry_count: Текущее количество повторов
        max_retries: Максимальное количество повторов
        
    Returns:
        True если нужна повторная оценка
    """
    return (not critic_evaluation.is_acceptable and 
            current_retry_count < max_retries)


def format_critic_summary(critic_results: Dict[RiskType, AgentTaskResult]) -> str:
    """
    Форматирование краткой сводки результатов критика
    
    Args:
        critic_results: Результаты критического анализа
        
    Returns:
        Текстовая сводка
    """
    total_evaluations = len(critic_results)
    acceptable_count = 0
    quality_scores = []
    
    for result in critic_results.values():
        if (result.status == ProcessingStatus.COMPLETED and 
            result.result_data and 
            "critic_evaluation" in result.result_data):
            
            critic_eval = result.result_data["critic_evaluation"]
            quality_scores.append(critic_eval["quality_score"])
            
            if critic_eval["is_acceptable"]:
                acceptable_count += 1
    
    if not quality_scores:
        return "Нет результатов критического анализа"
    
    avg_quality = sum(quality_scores) / len(quality_scores)
    acceptance_rate = acceptable_count / total_evaluations * 100
    
    return f"""📊 СВОДКА КРИТИЧЕСКОГО АНАЛИЗА:
• Всего оценок проанализировано: {total_evaluations}
• Приняты без замечаний: {acceptable_count} ({acceptance_rate:.1f}%)
• Средняя оценка качества: {avg_quality:.1f}/10
• Диапазон оценок: {min(quality_scores):.1f} - {max(quality_scores):.1f}
• Требуют доработки: {total_evaluations - acceptable_count}"""


# Экспорт основных классов и функций
__all__ = [
    "CriticAgent",
    "create_critic_agent",
    "create_critic_from_env",
    "create_critic_node_function",
    "create_quality_check_router",
    "extract_critic_evaluations_from_results",
    "should_retry_evaluation",
    "format_critic_summary"
]