# src/agents/base_agent.py
"""
Базовый класс для всех агентов в системе оценки рисков ИИ-агентов
Предоставляет общий интерфейс и функциональность для всех типов агентов
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass

from ..utils.llm_client import LLMClient, LLMConfig, LLMMessage, RiskAnalysisLLMClient
from ..utils.logger import get_logger, log_agent_execution, log_llm_call
from ..models.risk_models import AgentTaskResult, ProcessingStatus, RiskEvaluation


@dataclass
class AgentConfig:
    """Конфигурация агента"""
    name: str
    description: str
    llm_config: LLMConfig
    max_retries: int = 3
    timeout_seconds: int = 180
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
    
    async def run(self, input_data: Dict[str, Any], assessment_id: str) -> AgentTaskResult:
        """Выполнение оценки риска - ТОЛЬКО для EvaluationAgent"""
        
        # Проверяем что это действительно оценщик риска
        if not hasattr(self, 'risk_type'):
            raise ValueError(f"Класс {self.__class__.__name__} должен переопределить метод run()")
        
        start_time = datetime.now()
        
        try:
            agent_profile = input_data.get("agent_profile", {})
            
            # Получаем данные для оценки
            agent_data = self._prepare_agent_data(agent_profile)
            evaluation_criteria = self._get_evaluation_criteria()
            examples = self._get_evaluation_examples()
            
            # Выполняем оценку риска
            risk_data = await self.evaluate_risk(
                risk_type=self.risk_type,
                agent_data=agent_data,
                evaluation_criteria=evaluation_criteria,
                assessment_id=assessment_id,
                examples=examples
            )
            
            # Создаем RiskEvaluation с обязательными полями
            try:
                risk_evaluation = RiskEvaluation(
                    risk_type=self.risk_type,
                    evaluator_agent=self.name,
                    **risk_data
                )
            except Exception as e:
                # Fallback при ошибке валидации
                self.logger.bind_context(assessment_id, self.name).warning(
                    f"⚠️ Ошибка создания RiskEvaluation: {e}"
                )
                
                risk_evaluation = RiskEvaluation(
                    risk_type=self.risk_type,
                    evaluator_agent=self.name,
                    probability_score=risk_data.get("probability_score", 3),
                    impact_score=risk_data.get("impact_score", 3),
                    total_score=risk_data.get("total_score", 9),
                    risk_level=risk_data.get("risk_level", "medium"),
                    probability_reasoning=risk_data.get("probability_reasoning", "Дефолтное обоснование"),
                    impact_reasoning=risk_data.get("impact_reasoning", "Дефолтное обоснование"),
                    key_factors=risk_data.get("key_factors", ["Ошибка получения данных"]),
                    recommendations=risk_data.get("recommendations", ["Повторить оценку"]),
                    confidence_level=risk_data.get("confidence_level", 0.3)
                )
            
            # Создаем результат
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return AgentTaskResult(
                agent_name=self.name,
                task_type=f"evaluate_{self.risk_type}",
                status=ProcessingStatus.COMPLETED,
                result_data={"risk_evaluation": risk_evaluation.dict()},
                start_time=start_time,
                end_time=end_time,
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            # Полный fallback 
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            fallback_evaluation = RiskEvaluation(
                risk_type=getattr(self, 'risk_type', 'unknown'),
                evaluator_agent=self.name,
                probability_score=3,
                impact_score=3,
                total_score=9,
                risk_level="medium",
                probability_reasoning="Ошибка выполнения оценки",
                impact_reasoning="Используются дефолтные значения",
                key_factors=["Ошибка агента-оценщика"],
                recommendations=["Проверить настройки", "Повторить оценку"],
                confidence_level=0.1
            )
            
            return AgentTaskResult(
                agent_name=self.name,
                task_type=f"evaluate_{getattr(self, 'risk_type', 'unknown')}",
                status=ProcessingStatus.COMPLETED,
                result_data={"risk_evaluation": fallback_evaluation.dict()},
                start_time=start_time,
                end_time=end_time,
                execution_time_seconds=execution_time,
                error_message=str(e)
            )
    
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
        """Оценка риска с использованием специализированного клиента"""
        
        try:
            if not isinstance(self.llm_client, RiskAnalysisLLMClient):
                raise ValueError("Агент-оценщик должен использовать RiskAnalysisLLMClient")
            
            result = await self.llm_client.evaluate_risk(
                risk_type=risk_type,
                agent_data=agent_data,
                evaluation_criteria=evaluation_criteria,
                examples=examples
            )
            
            # ИСПРАВЛЕНИЕ: Применяем дополнительную валидацию
            validated_result = self._ensure_required_fields(result)
            
            # Логируем оценку
            self.logger.log_risk_evaluation(
                self.name,
                assessment_id,
                risk_type,
                validated_result["total_score"],
                validated_result["risk_level"]
            )
            
            return validated_result
            
        except Exception as e:
            # ИСПРАВЛЕНИЕ: В случае любой ошибки возвращаем безопасные дефолтные данные
            self.logger.bind_context(assessment_id, self.name).error(
                f"❌ Ошибка оценки риска {risk_type}: {e}"
            )
            
            # Возвращаем дефолтные данные вместо exception
            return self._get_default_evaluation_data(f"Ошибка оценки риска: {str(e)}")
    
    def _get_required_result_fields(self) -> List[str]:
        """Обязательные поля для результата оценки риска"""
        return [
            "probability_score", "impact_score", "total_score", 
            "risk_level", "probability_reasoning", "impact_reasoning"
        ]
    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """Парсинг ответа LLM с извлечением JSON - ПОЛНОСТЬЮ ИСПРАВЛЕННАЯ ВЕРСИЯ"""
        try:
            # Удаляем теги <think>...</think> если есть
            cleaned_content = response_content
            if '<think>' in cleaned_content and '</think>' in cleaned_content:
                start = cleaned_content.find('</think>') + 8
                cleaned_content = cleaned_content[start:].strip()
            
            # Ищем JSON блок в ответе
            json_content = None
            
            if "```json" in cleaned_content:
                start = cleaned_content.find("```json") + 7
                end = cleaned_content.find("```", start)
                if end != -1:
                    json_content = cleaned_content[start:end].strip()
                else:
                    json_content = cleaned_content[start:].strip()
            else:
                # Пытаемся найти JSON по фигурным скобкам
                start = cleaned_content.find("{")
                end = cleaned_content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_content = cleaned_content[start:end+1]
                else:
                    json_content = cleaned_content.strip()
            
            # Пытаемся парсить JSON
            parsed_data = json.loads(json_content)
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Проверяем и дополняем обязательные поля
            parsed_data = self._ensure_required_fields(parsed_data)
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            # Если парсинг не удался, возвращаем безопасные дефолтные данные
            self.logger.bind_context("unknown", self.name).warning(
                f"⚠️ Ошибка парсинга JSON, используем дефолтные значения: {e}"
            )
            return self._get_default_evaluation_data(f"Ошибка парсинга JSON: {str(e)}")
        
        except Exception as e:
            # Любая другая ошибка
            self.logger.bind_context("unknown", self.name).error(
                f"❌ Неожиданная ошибка при парсинге ответа LLM: {e}"
            )
            return self._get_default_evaluation_data(f"Неожиданная ошибка: {str(e)}")

    def _ensure_required_fields(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Обеспечивает наличие всех обязательных полей с валидными значениями"""
        
        # Определяем обязательные поля с дефолтными значениями
        required_fields = {
            "probability_score": 3,
            "impact_score": 3, 
            "total_score": 9,
            "risk_level": "medium",
            "probability_reasoning": "Автоматически сгенерированное обоснование вероятности",
            "impact_reasoning": "Автоматически сгенерированное обоснование воздействия",
            "key_factors": [],
            "recommendations": [],
            "confidence_level": 0.7
        }
        
        # Добавляем отсутствующие поля
        for field, default_value in required_fields.items():
            if field not in parsed_data or parsed_data[field] is None:
                parsed_data[field] = default_value
                self.logger.bind_context("unknown", self.name).debug(
                    f"🔧 Добавлено отсутствующее поле {field}: {default_value}"
                )
        
        # Валидируем и исправляем числовые поля
        parsed_data = self._validate_numeric_fields(parsed_data)
        
        # Валидируем и исправляем строковые поля
        parsed_data = self._validate_string_fields(parsed_data)
        
        # Валидируем и исправляем списковые поля
        parsed_data = self._validate_list_fields(parsed_data)
        
        # Пересчитываем зависимые поля
        parsed_data["total_score"] = parsed_data["probability_score"] * parsed_data["impact_score"]
        
        # Определяем risk_level на основе total_score
        total_score = parsed_data["total_score"]
        if total_score <= 6:
            parsed_data["risk_level"] = "low"
        elif total_score <= 14:
            parsed_data["risk_level"] = "medium"
        else:
            parsed_data["risk_level"] = "high"
        
        return parsed_data

    def _validate_numeric_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидирует и исправляет числовые поля"""
        
        # Валидация probability_score (1-5)
        try:
            data["probability_score"] = int(data["probability_score"])
            if not (1 <= data["probability_score"] <= 5):
                data["probability_score"] = 3
        except (ValueError, TypeError):
            data["probability_score"] = 3
        
        # Валидация impact_score (1-5)
        try:
            data["impact_score"] = int(data["impact_score"])
            if not (1 <= data["impact_score"] <= 5):
                data["impact_score"] = 3
        except (ValueError, TypeError):
            data["impact_score"] = 3
        
        # Валидация confidence_level (0.0-1.0)
        try:
            data["confidence_level"] = float(data["confidence_level"])
            if not (0.0 <= data["confidence_level"] <= 1.0):
                data["confidence_level"] = 0.7
        except (ValueError, TypeError):
            data["confidence_level"] = 0.7
        
        return data

    def _validate_string_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидирует и исправляет строковые поля"""
        
        # Валидация risk_level
        valid_levels = ["low", "medium", "high"]
        if data.get("risk_level") not in valid_levels:
            data["risk_level"] = "medium"
        
        # Валидация reasoning полей
        if not data.get("probability_reasoning") or len(str(data["probability_reasoning"]).strip()) < 10:
            data["probability_reasoning"] = "Обоснование вероятности не предоставлено или некорректно"
        
        if not data.get("impact_reasoning") or len(str(data["impact_reasoning"]).strip()) < 10:
            data["impact_reasoning"] = "Обоснование воздействия не предоставлено или некорректно"
        
        return data

    def _validate_list_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидирует и исправляет списковые поля"""
        
        list_fields = ["key_factors", "recommendations"]
        
        for field in list_fields:
            if not isinstance(data.get(field), list):
                data[field] = []
            
            # Убираем пустые строки и None
            data[field] = [
                item for item in data[field] 
                if item and isinstance(item, str) and item.strip()
            ]
            
            # Ограничиваем количество элементов
            data[field] = data[field][:10]
        
        return data

    def _get_default_evaluation_data(self, error_message: str) -> Dict[str, Any]:
        """Возвращает безопасные дефолтные данные для оценки"""
        return {
            "probability_score": 3,
            "impact_score": 3,
            "total_score": 9,
            "risk_level": "medium",
            "probability_reasoning": f"Не удалось получить обоснование от LLM. Ошибка: {error_message}",
            "impact_reasoning": "Использованы дефолтные значения из-за ошибки парсинга",
            "key_factors": ["Ошибка получения данных от LLM"],
            "recommendations": ["Проверить промпт и формат ответа", "Повторить оценку"],
            "confidence_level": 0.3
        }

# ===============================
# Фабрики для создания агентов
# ===============================

def create_agent_config(
    name: str,
    description: str,
    llm_base_url: str = "http://127.0.0.1:1234",
    llm_model: str = "qwen3-4b",
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
        llm_model=os.getenv("LLM_MODEL", "qwen3-4b"),
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
