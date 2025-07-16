# src/agents/critic_agent.py
"""
Критик-агент для оценки качества работы агентов-оценщиков рисков

ОБНОВЛЕНО: Убраны LLM параметры, используется центральный конфигуратор
"""

import json
import asyncio
from typing import Dict, Any, List, Optional, Set
from datetime import datetime

from .base_agent import BaseAgent, AgentConfig
from ..models.risk_models import (
    RiskType, AgentTaskResult, ProcessingStatus, CriticEvaluation,
    WorkflowState
)
from ..utils.logger import get_logger


class CriticAgent(BaseAgent):
    """
    Критик-агент для контроля качества оценок рисков
    
    Функции:
    1. Анализ качества оценок от агентов-оценщиков
    2. Проверка обоснованности выводов
    3. Выявление недостатков в анализе
    4. Принятие решений о необходимости повторной оценки
    5. Обеспечение высокого качества итоговых результатов
    """
    
    def __init__(self, config: AgentConfig, quality_threshold: float = 7.0):
        super().__init__(config)
        self.quality_threshold = quality_threshold
        
        # Критерии оценки качества
        self.quality_criteria = {
            "completeness": "Полнота анализа всех аспектов риска",
            "reasoning": "Логичность и обоснованность выводов", 
            "evidence": "Соответствие оценок предоставленным данным",
            "actionability": "Практичность рекомендаций",
            "methodology": "Соответствие методике оценки рисков"
        }
    
    def get_system_prompt(self) -> str:
        """Системный промпт для критик-агента"""
        return f"""Ты - эксперт-критик по оценке качества анализа рисков ИИ-агентов.

Твоя задача: критически оценивать качество работы агентов-оценщиков рисков.

КРИТЕРИИ ОЦЕНКИ КАЧЕСТВА:
1. Полнота анализа (учтены ли все важные аспекты)
2. Обоснованность выводов (логичность рассуждений)
3. Соответствие данным (подтверждены ли оценки фактами)
4. Практичность рекомендаций (применимость советов)
5. Методологическая корректность (следование стандартам)

ШКАЛА КАЧЕСТВА: 0.0-10.0 баллов
ПОРОГ ПРИЕМЛЕМОСТИ: {self.quality_threshold} баллов

ПРИНЦИПЫ КРИТИКИ:
- Будь объективным и строгим
- Ищи конкретные недостатки
- Предлагай улучшения
- Требуй высокого качества анализа

ФОРМАТ ОТВЕТА: Структурированный JSON с детальной критикой"""
    
    async def process(
        self, 
        input_data: Dict[str, Any], 
        assessment_id: str = "unknown"
    ) -> AgentTaskResult:
        """
        Основная обработка: критический анализ оценок
        
        Args:
            input_data: Данные для критики (agent_data + evaluations)
            assessment_id: ID оценки
            
        Returns:
            Результат критического анализа
        """
        start_time = datetime.now()
        
        try:
            # Извлекаем необходимые данные
            agent_data = input_data.get("agent_data", {})
            evaluations = input_data.get("evaluations", {})
            
            if not evaluations:
                raise ValueError("Отсутствуют оценки для критического анализа")
            
            # Критикуем каждую оценку
            critic_results = {}
            
            for risk_type_str, evaluation in evaluations.items():
                try:
                    risk_type = RiskType(risk_type_str)
                    
                    critic_result = await self._evaluate_single_assessment(
                        agent_data=agent_data,
                        risk_evaluation=evaluation,
                        risk_type=risk_type,
                        assessment_id=assessment_id
                    )
                    
                    critic_results[risk_type] = critic_result
                    
                except Exception as e:
                    self.logger.error(f"Ошибка критики {risk_type_str}: {e}")
                    # Создаем fallback результат
                    critic_results[RiskType(risk_type_str)] = self._create_fallback_critic_result(str(e))
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.update_stats(execution_time, True)
            
            return AgentTaskResult(
                agent_name=self.name,
                status=ProcessingStatus.COMPLETED,
                result_data=critic_results,
                execution_time=execution_time,
                assessment_id=assessment_id
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.update_stats(execution_time, False)
            
            self.logger.error(f"Критический анализ завершился с ошибкой: {e}")
            
            return AgentTaskResult(
                agent_name=self.name,
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time,
                assessment_id=assessment_id
            )
    
    async def _evaluate_single_assessment(
        self,
        agent_data: Dict[str, Any],
        risk_evaluation: Dict[str, Any],
        risk_type: RiskType,
        assessment_id: str
    ) -> CriticEvaluation:
        """Критическая оценка одной оценки риска"""
        
        # Подготавливаем контекст для анализа
        context = self._prepare_evaluation_context(agent_data, risk_evaluation, risk_type)
        
        # Формируем промпт для критики
        critique_prompt = f"""Критически оцени качество оценки {risk_type.value} рисков.

ОЦЕНКА ДЛЯ АНАЛИЗА:
{json.dumps(risk_evaluation, ensure_ascii=False, indent=2)}

Проанализируй по критериям качества и дай структурированную оценку."""
        
        # Получаем критику от LLM
        critic_response = await self.call_llm_structured(
            data_to_analyze=context,
            extraction_prompt=critique_prompt,
            assessment_id=assessment_id,
            expected_format="JSON"
        )
        
        # Создаем объект CriticEvaluation
        return self._parse_critic_response(critic_response, risk_type)
    
    def _prepare_evaluation_context(
        self,
        agent_data: Dict[str, Any],
        risk_evaluation: Dict[str, Any],
        risk_type: RiskType
    ) -> str:
        """Подготовка контекста для критического анализа"""
        
        context = f"""АНАЛИЗИРУЕМЫЙ АГЕНТ:
Название: {agent_data.get('name', 'Неизвестно')}
Тип: {agent_data.get('agent_type', 'Неизвестно')}
Описание: {agent_data.get('description', 'Описание отсутствует')}

КОНТЕКСТ ДАННЫХ АГЕНТА:
{json.dumps(agent_data, ensure_ascii=False, indent=2)}

ТИП ОЦЕНИВАЕМОГО РИСКА: {risk_type.value}

КРИТЕРИИ КАЧЕСТВЕННОЙ ОЦЕНКИ:
- Риск-скор должен быть обоснован конкретными фактами
- Рассуждения должны быть логичными и последовательными  
- Рекомендации должны быть практичными и выполнимыми
- Анализ должен учитывать специфику типа агента
- Выводы должны соответствовать представленным данным"""
        
        return context
    
    def _parse_critic_response(
        self,
        critic_response: Dict[str, Any],
        risk_type: RiskType
    ) -> CriticEvaluation:
        """Парсинг ответа критика в объект CriticEvaluation"""
        
        quality_score = float(critic_response.get("quality_score", 5.0))
        is_acceptable = quality_score >= self.quality_threshold
        
        return CriticEvaluation(
            risk_type=risk_type,
            quality_score=quality_score,
            is_acceptable=is_acceptable,
            issues_found=critic_response.get("issues_found", []),
            improvement_suggestions=critic_response.get("improvement_suggestions", []),
            critic_reasoning=critic_response.get("critic_reasoning", "Обоснование отсутствует"),
            evaluated_at=datetime.now()
        )
    
    def _create_fallback_critic_result(self, error_message: str) -> CriticEvaluation:
        """Создание fallback результата критики при ошибках"""
        return CriticEvaluation(
            risk_type=RiskType.ETHICAL,  # Default type
            quality_score=0.0,
            is_acceptable=False,
            issues_found=[f"Ошибка критического анализа: {error_message}"],
            improvement_suggestions=["Повторить анализ с корректными данными"],
            critic_reasoning=f"Критический анализ не выполнен из-за ошибки: {error_message}",
            evaluated_at=datetime.now()
        )
    
    async def batch_critique(
        self,
        evaluations_batch: Dict[str, Dict[str, Any]],
        agent_data: Dict[str, Any],
        assessment_id: str = "unknown"
    ) -> Dict[RiskType, CriticEvaluation]:
        """Пакетная критика нескольких оценок"""
        
        tasks = []
        for risk_type_str, evaluation in evaluations_batch.items():
            risk_type = RiskType(risk_type_str)
            
            task = self._evaluate_single_assessment(
                agent_data=agent_data,
                risk_evaluation=evaluation,
                risk_type=risk_type,
                assessment_id=assessment_id
            )
            tasks.append((risk_type, task))
        
        # Выполняем критику параллельно
        results = {}
        for risk_type, task in tasks:
            try:
                critic_result = await task
                results[risk_type] = critic_result
            except Exception as e:
                self.logger.error(f"Ошибка критики {risk_type.value}: {e}")
                results[risk_type] = self._create_fallback_critic_result(str(e))
        
        return results


# ===============================
# Функции для LangGraph интеграции
# ===============================

def create_quality_check_router(critic_agent: CriticAgent, max_retries: int = 3):
    """
    Создание роутера для проверки качества в LangGraph
    
    Args:
        critic_agent: Экземпляр критик-агента
        max_retries: Максимальное количество повторов
        
    Returns:
        Функция роутера для LangGraph
    """
    
    def quality_check_router(state: WorkflowState) -> str:
        """Роутер для принятия решений на основе критики"""
        
        critic_results = state.get("critic_results", {})
        retry_count = state.get("retry_count", {})
        
        if not critic_results:
            return "finalization"  # Нет результатов критики
        
        # Определяем какие оценки нужно повторить
        retry_needed = []
        for risk_type, critic_eval in critic_results.items():
            if not critic_eval.is_acceptable:
                retry_needed.append(risk_type)
        
        if not retry_needed:
            return "finalization"  # Все оценки приемлемого качества
        
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
# Фабрики (ОБНОВЛЕННЫЕ)
# ===============================

def create_critic_agent(
    quality_threshold: float = 7.0,
    max_retries: int = 2,
    timeout_seconds: int = 90
) -> CriticAgent:
    """
    Создание критик-агента (новая версия без LLM параметров)
    
    Args:
        quality_threshold: Порог качества для принятия оценок
        max_retries: Максимум повторов для критика
        timeout_seconds: Тайм-аут в секундах
        
    Returns:
        Настроенный критик-агент
    """
    config = AgentConfig(
        name="critic_agent",
        description="Агент для критического анализа качества оценок рисков",
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        use_risk_analysis_client=True  # Критик использует специализированный клиент
    )
    
    return CriticAgent(config, quality_threshold)


def create_critic_from_env() -> CriticAgent:
    """Создание критик-агента из переменных окружения"""
    import os
    
    return create_critic_agent(
        quality_threshold=float(os.getenv("QUALITY_THRESHOLD", "7.0")),
        max_retries=int(os.getenv("MAX_RETRY_COUNT", "2")),
        timeout_seconds=90
    )


# Legacy функция для обратной совместимости (DEPRECATED)
def create_critic_agent_legacy(
    llm_base_url: str = "http://127.0.0.1:1234",
    llm_model: str = "qwen3-4b", 
    temperature: float = 0.1,
    quality_threshold: float = 7.0
) -> CriticAgent:
    """
    DEPRECATED: Создание критик-агента (старая версия)
    Используйте create_critic_agent() без LLM параметров
    """
    import warnings
    from ..utils.llm_client import LLMConfig
    
    warnings.warn(
        "create_critic_agent_legacy deprecated. Use create_critic_agent() without LLM params.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Создаем переопределение для legacy кода
    llm_override = LLMConfig(
        base_url=llm_base_url,
        model=llm_model,
        temperature=temperature,
        timeout=90
    )
    
    config = AgentConfig(
        name="critic_agent",
        description="Агент для критического анализа качества оценок рисков",
        max_retries=2,
        timeout_seconds=90,
        use_risk_analysis_client=True,
        llm_override=llm_override
    )
    
    return CriticAgent(config, quality_threshold)


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
            hasattr(task_result.result_data, 'get')):
            critic_evaluations[risk_type] = task_result.result_data
    
    return critic_evaluations


def summarize_critic_results(
    critic_evaluations: Dict[RiskType, CriticEvaluation]
) -> Dict[str, Any]:
    """Суммирование результатов критического анализа"""
    
    total_evaluations = len(critic_evaluations)
    acceptable_count = sum(1 for eval in critic_evaluations.values() if eval.is_acceptable)
    average_quality = sum(eval.quality_score for eval in critic_evaluations.values()) / total_evaluations if total_evaluations > 0 else 0
    
    all_issues = []
    all_suggestions = []
    
    for eval in critic_evaluations.values():
        all_issues.extend(eval.issues_found)
        all_suggestions.extend(eval.improvement_suggestions)
    
    return {
        "total_evaluations": total_evaluations,
        "acceptable_evaluations": acceptable_count,
        "acceptance_rate": acceptable_count / total_evaluations if total_evaluations > 0 else 0,
        "average_quality_score": average_quality,
        "common_issues": list(set(all_issues)),
        "improvement_suggestions": list(set(all_suggestions))
    }
def create_critic_agent_legacy(**kwargs) -> CriticAgent:
    """
    DEPRECATED: Legacy функция создания критика
    
    Используйте create_critic_agent() без параметров.
    """
    import warnings
    warnings.warn(
        "create_critic_agent_legacy deprecated. Use create_critic_agent() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_critic_agent()

# Экспорт
__all__ = [
    "CriticAgent",
    "create_critic_agent",
    "create_critic_from_env",
    "create_quality_check_router",
    "extract_critic_evaluations_from_results",
    "summarize_critic_results",
    # Legacy exports (deprecated)
    "create_critic_agent_legacy"
]