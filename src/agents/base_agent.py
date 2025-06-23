# src/agents/base_agent.py
"""
Базовый класс для всех агентов в системе оценки рисков ИИ-агентов
Предоставляет общий интерфейс и функциональность для всех типов агентов
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass

from ..utils.llm_client import LLMClient, LLMConfig, LLMMessage, RiskAnalysisLLMClient
from ..utils.logger import get_logger, log_agent_execution, log_llm_call
from ..models.risk_models import AgentTaskResult, ProcessingStatus


@dataclass
class AgentConfig:
    """Конфигурация агента"""
    name: str
    description: str
    llm_config: LLMConfig
    max_retries: int = 3
    timeout_seconds: int = 120
    temperature: float = 0.1
    use_risk_analysis_client: bool = False


class BaseAgent(ABC):
    """
    Базовый класс для всех агентов системы оценки рисков
    
    Предоставляет:
    - Подключение к LLM
    - Логирование
    - Обработка ошибок и повторы
    - Валидация результатов
    - Стандартный интерфейс для агентов
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = get_logger()
        
        # Инициализируем LLM клиент
        if config.use_risk_analysis_client:
            self.llm_client = RiskAnalysisLLMClient(config.llm_config)
        else:
            self.llm_client = LLMClient(config.llm_config)
        
        # Статистика работы агента
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_execution_time": 0.0,
            "average_response_time": 0.0
        }
    
    @property
    def name(self) -> str:
        """Имя агента"""
        return self.config.name
    
    @property
    def description(self) -> str:
        """Описание агента"""
        return self.config.description
    
    @abstractmethod
    async def process(
        self, 
        input_data: Dict[str, Any], 
        assessment_id: str
    ) -> AgentTaskResult:
        """
        Основной метод обработки для агента
        
        Args:
            input_data: Входные данные для обработки
            assessment_id: Идентификатор оценки
            
        Returns:
            Результат работы агента
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Получение системного промпта для агента"""
        pass
    
    async def run(
        self, 
        input_data: Dict[str, Any], 
        assessment_id: str
    ) -> AgentTaskResult:
        """
        Запуск агента с обработкой ошибок и повторами
        
        Args:
            input_data: Входные данные
            assessment_id: ID оценки
            
        Returns:
            Результат выполнения
        """
        task_result = AgentTaskResult(
            agent_name=self.name,
            task_type=self._get_task_type(),
            status=ProcessingStatus.IN_PROGRESS,
            start_time=datetime.now()
        )
        
        # Логируем начало работы
        self.logger.log_agent_start(self.name, self._get_task_type(), assessment_id)
        
        for attempt in range(self.config.max_retries):
            try:
                # Выполняем основную обработку
                result = await asyncio.wait_for(
                    self.process(input_data, assessment_id),
                    timeout=self.config.timeout_seconds
                )
                
                # Обновляем статистику
                self._update_stats(True, result.execution_time_seconds or 0)
                
                # Логируем успех
                self.logger.log_agent_success(
                    self.name, 
                    self._get_task_type(), 
                    assessment_id, 
                    result.execution_time_seconds or 0
                )
                
                return result
                
            except asyncio.TimeoutError:
                error_msg = f"Тайм-аут выполнения ({self.config.timeout_seconds}с)"
                await self._handle_retry(task_result, error_msg, attempt, assessment_id)
                
            except Exception as e:
                error_msg = f"Ошибка выполнения: {str(e)}"
                await self._handle_retry(task_result, error_msg, attempt, assessment_id)
        
        # Все попытки исчерпаны
        task_result.status = ProcessingStatus.FAILED
        task_result.end_time = datetime.now()
        task_result.execution_time_seconds = (
            task_result.end_time - task_result.start_time
        ).total_seconds()
        
        self._update_stats(False, task_result.execution_time_seconds)
        
        self.logger.log_agent_error(
            self.name, 
            self._get_task_type(), 
            assessment_id, 
            Exception(task_result.error_message or "Неизвестная ошибка")
        )
        
        return task_result
    
    async def call_llm(
        self,
        system_prompt: str,
        user_message: str,
        context: Optional[str] = None,
        assessment_id: str = "unknown",
        temperature: Optional[float] = None
    ) -> str:
        """
        Вызов LLM с логированием и обработкой ошибок
        
        Args:
            system_prompt: Системный промпт
            user_message: Сообщение пользователя
            context: Дополнительный контекст
            assessment_id: ID оценки для логирования
            temperature: Температура генерации
            
        Returns:
            Ответ от LLM
        """
        messages = [
            LLMMessage(role="system", content=system_prompt)
        ]
        
        # Добавляем контекст если есть
        if context:
            messages.append(
                LLMMessage(
                    role="user", 
                    content=f"Контекст:\n{context}\n\nЗадача:\n{user_message}"
                )
            )
        else:
            messages.append(LLMMessage(role="user", content=user_message))
        
        # Вызываем LLM
        response = await self.llm_client.complete_chat(
            messages=messages,
            temperature=temperature or self.config.temperature
        )
        
        # Логируем вызов LLM
        self.logger.log_llm_request(
            self.name,
            assessment_id,
            response.model,
            response.usage.get("total_tokens", 0)
        )
        
        return response.content
    
    async def call_llm_structured(
        self,
        data_to_analyze: str,
        extraction_prompt: str,
        assessment_id: str = "unknown",
        expected_format: str = "JSON"
    ) -> Dict[str, Any]:
        """
        Вызов LLM для получения структурированных данных
        
        Args:
            data_to_analyze: Данные для анализа
            extraction_prompt: Промпт для извлечения
            assessment_id: ID оценки
            expected_format: Ожидаемый формат ответа
            
        Returns:
            Структурированные данные
        """
        if not isinstance(self.llm_client, (LLMClient, RiskAnalysisLLMClient)):
            raise ValueError("LLM клиент не поддерживает структурированные запросы")
        
        result = await self.llm_client.extract_structured_data(
            data_to_analyze=data_to_analyze,
            extraction_prompt=extraction_prompt,
            expected_format=expected_format
        )
        
        # Логируем вызов
        self.logger.log_llm_request(
            self.name,
            assessment_id,
            self.llm_client.config.model,
            0  # Токены уже залогированы внутри extract_structured_data
        )
        
        return result
    
    def validate_result(self, result_data: Dict[str, Any]) -> bool:
        """
        Базовая валидация результата агента
        Переопределяется в наследниках для специфичной валидации
        
        Args:
            result_data: Данные результата для валидации
            
        Returns:
            True если результат валиден
        """
        # Базовая проверка
        if not isinstance(result_data, dict):
            return False
        
        # Проверяем обязательные поля (переопределяется в наследниках)
        required_fields = self._get_required_result_fields()
        for field in required_fields:
            if field not in result_data:
                return False
        
        return True
    
    def _get_task_type(self) -> str:
        """Получение типа задачи агента"""
        return self.__class__.__name__.lower().replace('agent', '')
    
    def _get_required_result_fields(self) -> List[str]:
        """Получение списка обязательных полей результата (переопределяется в наследниках)"""
        return []
    
    async def _handle_retry(
        self, 
        task_result: AgentTaskResult, 
        error_msg: str, 
        attempt: int, 
        assessment_id: str
    ):
        """Обработка повторной попытки"""
        task_result.error_message = error_msg
        
        if attempt < self.config.max_retries - 1:
            # Логируем повтор
            self.logger.log_agent_retry(
                self.name, 
                self._get_task_type(), 
                assessment_id, 
                attempt + 1
            )
            
            # Небольшая задержка перед повтором
            await asyncio.sleep(1.0 * (attempt + 1))
        else:
            # Последняя попытка - фиксируем ошибку
            task_result.status = ProcessingStatus.FAILED
            task_result.end_time = datetime.now()
    
    def _update_stats(self, success: bool, execution_time: float):
        """Обновление статистики агента"""
        self.stats["total_requests"] += 1
        self.stats["total_execution_time"] += execution_time
        
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        # Пересчитываем среднее время ответа
        self.stats["average_response_time"] = (
            self.stats["total_execution_time"] / self.stats["total_requests"]
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики работы агента"""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_requests"] / max(self.stats["total_requests"], 1)
            ),
            "agent_name": self.name,
            "agent_type": self._get_task_type()
        }
    
    async def health_check(self) -> bool:
        """Проверка работоспособности агента"""
        try:
            # Проверяем доступность LLM
            llm_healthy = await self.llm_client.health_check()
            
            # Можно добавить дополнительные проверки
            return llm_healthy
            
        except Exception:
            return False
    
    async def cleanup(self):
        """Очистка ресурсов агента"""
        try:
            await self.llm_client.close()
        except Exception:
            pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()


class AnalysisAgent(BaseAgent):
    """
    Базовый класс для агентов анализа
    Расширяет BaseAgent функциональностью для анализа данных
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
    
    async def analyze_data(
        self,
        data: str,
        analysis_type: str,
        criteria: str,
        assessment_id: str,
        examples: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Общий метод анализа данных
        
        Args:
            data: Данные для анализа
            analysis_type: Тип анализа
            criteria: Критерии анализа
            assessment_id: ID оценки
            examples: Примеры для контекста
            
        Returns:
            Результат анализа
        """
        system_prompt = self.get_system_prompt()
        
        if examples:
            system_prompt += f"\n\nПРИМЕРЫ:\n{examples}"
        
        user_message = f"""ДАННЫЕ ДЛЯ АНАЛИЗА:
{data}

ТИП АНАЛИЗА: {analysis_type}

КРИТЕРИИ:
{criteria}

Выполни анализ согласно указанным критериям."""
        
        response = await self.call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            assessment_id=assessment_id
        )
        
        return {"analysis_result": response, "analysis_type": analysis_type}


class EvaluationAgent(BaseAgent):
    """
    Базовый класс для агентов-оценщиков рисков
    Расширяет BaseAgent функциональностью для оценки рисков
    """
    
    def __init__(self, config: AgentConfig):
        # Оценщики должны использовать специализированный клиент
        config.use_risk_analysis_client = True
        super().__init__(config)
    
    async def evaluate_risk(
        self,
        risk_type: str,
        agent_data: str,
        evaluation_criteria: str,
        assessment_id: str,
        examples: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Оценка риска с использованием специализированного клиента
        
        Args:
            risk_type: Тип риска
            agent_data: Данные об агенте
            evaluation_criteria: Критерии оценки
            assessment_id: ID оценки
            examples: Примеры оценок
            
        Returns:
            Результат оценки риска
        """
        if not isinstance(self.llm_client, RiskAnalysisLLMClient):
            raise ValueError("Агент-оценщик должен использовать RiskAnalysisLLMClient")
        
        result = await self.llm_client.evaluate_risk(
            risk_type=risk_type,
            agent_data=agent_data,
            evaluation_criteria=evaluation_criteria,
            examples=examples
        )
        
        # Логируем оценку
        self.logger.log_risk_evaluation(
            self.name,
            assessment_id,
            risk_type,
            result["total_score"],
            result["risk_level"]
        )
        
        return result
    
    def _get_required_result_fields(self) -> List[str]:
        """Обязательные поля для результата оценки риска"""
        return [
            "probability_score", "impact_score", "total_score", 
            "risk_level", "probability_reasoning", "impact_reasoning"
        ]


# ===============================
# Фабрики для создания агентов
# ===============================

def create_agent_config(
    name: str,
    description: str,
    llm_base_url: str = "http://127.0.0.1:1234",
    llm_model: str = "qwen3-8b",
    temperature: float = 0.1,
    max_retries: int = 3,
    timeout_seconds: int = 120,
    use_risk_analysis_client: bool = False
) -> AgentConfig:
    """
    Создание конфигурации агента
    
    Args:
        name: Имя агента
        description: Описание агента
        llm_base_url: URL LLM сервера
        llm_model: Модель LLM
        temperature: Температура генерации
        max_retries: Максимум повторов
        timeout_seconds: Тайм-аут в секундах
        use_risk_analysis_client: Использовать специализированный клиент
        
    Returns:
        Конфигурация агента
    """
    llm_config = LLMConfig(
        base_url=llm_base_url,
        model=llm_model,
        temperature=temperature,
        timeout=timeout_seconds
    )
    
    return AgentConfig(
        name=name,
        description=description,
        llm_config=llm_config,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        use_risk_analysis_client=use_risk_analysis_client
    )


def create_default_config_from_env() -> AgentConfig:
    """Создание конфигурации по умолчанию из переменных окружения"""
    import os
    
    return create_agent_config(
        name="default_agent",
        description="Агент с настройками по умолчанию",
        llm_base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234"),
        llm_model=os.getenv("LLM_MODEL", "qwen3-8b"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_retries=int(os.getenv("MAX_RETRY_COUNT", "3")),
        timeout_seconds=120
    )


# Экспорт основных классов
__all__ = [
    "BaseAgent",
    "AnalysisAgent", 
    "EvaluationAgent",
    "AgentConfig",
    "create_agent_config",
    "create_default_config_from_env"
]