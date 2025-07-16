# src/agents/base_agent.py
"""
Базовый класс для всех агентов в системе оценки рисков ИИ-агентов
Предоставляет общий интерфейс и функциональность для всех типов агентов

ОБНОВЛЕНО: Интеграция с центральным LLM конфигуратором
"""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass

from ..utils.llm_client import LLMClient, LLMConfig, LLMMessage, RiskAnalysisLLMClient
from ..utils.logger import get_logger, log_agent_execution, log_llm_call
from ..models.risk_models import AgentTaskResult, ProcessingStatus
from ..config import get_global_llm_config


@dataclass
class AgentConfig:
    """Конфигурация агента (упрощенная версия без LLM параметров)"""
    name: str
    description: str
    max_retries: int = 3
    timeout_seconds: int = 180
    use_risk_analysis_client: bool = False
    # Опциональные переопределения LLM настроек (для тестирования)
    llm_override: Optional[LLMConfig] = None


class BaseAgent(ABC):
    """
    Базовый класс для всех агентов системы оценки рисков
    
    Предоставляет:
    - Подключение к LLM через центральный конфигуратор
    - Логирование
    - Обработка ошибок и повторы
    - Валидация результатов
    - Стандартный интерфейс для агентов
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = get_logger()
        
        # Получаем LLM конфигурацию из центрального менеджера
        self._setup_llm_client()
        
        # Статистика работы агента
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_execution_time": 0.0,
            "average_response_time": 0.0
        }
    
    def _setup_llm_client(self):
        """Настройка LLM клиента через центральный конфигуратор"""
        if self.config.llm_override:
            # Используем переопределенную конфигурацию (для тестирования)
            llm_config = self.config.llm_override
        else:
            # Получаем конфигурацию из центрального менеджера
            global_config_manager = get_global_llm_config()
            unified_config = global_config_manager.get_config()
            
            # Преобразуем унифицированную конфигурацию в LLMConfig
            llm_config = LLMConfig(
                base_url=unified_config.base_url,
                model=unified_config.model,
                temperature=unified_config.temperature,
                max_tokens=unified_config.max_tokens,
                timeout=unified_config.timeout,
                max_retries=unified_config.max_retries,
                retry_delay=unified_config.retry_delay
            )
        
        # Создаем подходящий клиент
        if self.config.use_risk_analysis_client:
            self.llm_client = RiskAnalysisLLMClient(llm_config)
        else:
            self.llm_client = LLMClient(llm_config)
    
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
        assessment_id: str = "unknown"
    ) -> AgentTaskResult:
        """
        Основной метод обработки данных агентом
        
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
    
    async def execute_with_retry(
        self,
        task_func,
        *args,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Выполнение задачи с повторами при ошибках
        
        Args:
            task_func: Функция для выполнения
            max_retries: Максимальное количество повторов
            *args, **kwargs: Аргументы для функции
            
        Returns:
            Результат выполнения функции
        """
        retries = max_retries or self.config.max_retries
        last_error = None
        
        for attempt in range(retries + 1):
            try:
                result = await task_func(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(f"Задача выполнена успешно с попытки {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Попытка {attempt + 1} неудачна: {str(e)}")
                
                if attempt < retries:
                    await asyncio.sleep(2 ** attempt)  # Экспоненциальная задержка
                else:
                    self.logger.error(f"Все {retries + 1} попыток неудачны. Финальная ошибка: {str(e)}")
        
        raise last_error
    
    async def call_llm(
        self,
        prompt: str,
        context: str = "",
        assessment_id: str = "unknown",
        temperature: Optional[float] = None
    ) -> str:
        """
        Универсальный метод для вызова LLM
        
        Args:
            prompt: Основной промпт
            context: Контекстная информация
            assessment_id: ID оценки
            temperature: Температура (переопределяет настройки)
            
        Returns:
            Ответ от LLM
        """
        system_prompt = self.get_system_prompt()
        
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=f"{context}\n\n{prompt}")
        ]
        
        response = await self.llm_client.chat(
            messages=messages,
            temperature=temperature or self.llm_client.config.temperature
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
        
        return True
    
    def _get_required_result_fields(self) -> List[str]:
        """Обязательные поля для результата оценки риска"""
        return [
            "probability_score", "impact_score", "total_score", 
            "risk_level", "probability_reasoning", "impact_reasoning"
        ]
    
    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """ПОЛНОСТЬЮ ИСПРАВЛЕННЫЙ парсинг ответа LLM с максимальной надежностью"""
        
        try:
            # Шаг 1: Очистка контента
            cleaned_content = response_content.strip()
            
            # Удаляем теги <think>...</think> если есть
            import re
            cleaned_content = re.sub(r'<think>.*?</think>', '', cleaned_content, flags=re.DOTALL)
            cleaned_content = cleaned_content.strip()
            
            # Удаляем markdown блоки
            if "```json" in cleaned_content:
                # Находим все JSON блоки
                json_blocks = re.findall(r'```json\s*(.*?)\s*```', cleaned_content, re.DOTALL)
                if json_blocks:
                    json_content = json_blocks[-1].strip()  # Берем последний блок
                else:
                    # Если не найден закрывающий тег, берем все после ```json
                    start = cleaned_content.find("```json") + 7
                    json_content = cleaned_content[start:].strip()
            else:
                # Ищем JSON по фигурным скобкам
                json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
                if json_match:
                    json_content = json_match.group().strip()
                else:
                    json_content = cleaned_content
            
            # Шаг 2: Дополнительная очистка JSON
            # Убираем потенциальные концевые символы после }
            if '}' in json_content:
                end_pos = json_content.rfind('}')
                json_content = json_content[:end_pos + 1]
            
            # Шаг 3: Пытаемся парсить JSON
            try:
                parsed_data = json.loads(json_content)
            except json.JSONDecodeError:
                # Пытаемся починить распространенные ошибки JSON
                json_content = self._fix_common_json_errors(json_content)
                parsed_data = json.loads(json_content)
            
            # Шаг 4: КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ - всегда обеспечиваем обязательные поля
            parsed_data = self._ensure_required_fields(parsed_data)
            
            return parsed_data
            
        except Exception as e:
            # Если ничего не помогло, возвращаем безопасные дефолтные данные
            self.logger.warning(
                f"⚠️ Критическая ошибка парсинга LLM ответа, используем fallback: {e}"
            )
            self.logger.debug(
                f"Проблемный контент: {response_content[:200]}..."
            )
            return self._get_default_evaluation_data(f"Критическая ошибка парсинга: {str(e)}")

    def _fix_common_json_errors(self, json_content: str) -> str:
        """Исправляет распространенные ошибки в JSON от LLM"""
        
        # Убираем trailing commas
        json_content = re.sub(r',\s*}', '}', json_content)
        json_content = re.sub(r',\s*]', ']', json_content)
        
        # Исправляем неэкранированные кавычки в строках
        # Это простое исправление, может потребовать доработки
        json_content = re.sub(r'(?<!\\)"(?=[^,}\]]*[,}\]])', '\\"', json_content)
        
        # Убираем комментарии в JSON (если есть)
        json_content = re.sub(r'//.*?\n', '\n', json_content)
        json_content = re.sub(r'/\*.*?\*/', '', json_content, flags=re.DOTALL)
        
        return json_content

    def _ensure_required_fields(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """УЛУЧШЕННАЯ версия обеспечения обязательных полей"""
        
        # Шаг 1: Определяем обязательные поля с умными дефолтами
        required_fields = {
            "probability_score": 3,
            "impact_score": 3, 
            "total_score": 9,
            "risk_level": "medium",
            "probability_reasoning": "Обоснование вероятности не предоставлено LLM",
            "impact_reasoning": "Обоснование воздействия не предоставлено LLM",
            "key_factors": [],
            "recommendations": [],
            "confidence_level": 0.7
        }
        
        # Шаг 2: Добавляем отсутствующие поля
        for field, default_value in required_fields.items():
            if field not in parsed_data or parsed_data[field] is None:
                parsed_data[field] = default_value
                self.logger.debug(
                    f"🔧 Добавлено отсутствующее поле {field}: {default_value}"
                )
        
        # Шаг 3: Валидируем и исправляем типы данных
        parsed_data = self._validate_and_fix_field_types(parsed_data)
        
        # Шаг 4: Валидируем бизнес-логику
        parsed_data = self._validate_business_logic(parsed_data)
        
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
    
    def _validate_and_fix_field_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидирует и исправляет типы полей"""
        
        # Числовые поля (1-5)
        score_fields = ["probability_score", "impact_score"]
        for field in score_fields:
            try:
                value = int(data[field])
                data[field] = max(1, min(5, value))  # Ограничиваем диапазон 1-5
            except (ValueError, TypeError):
                data[field] = 3  # Средний балл
                self.logger.warning(
                    f"🔧 Исправлен некорректный {field}: установлено значение 3"
                )
        
        # Confidence level (0.0-1.0)
        try:
            value = float(data["confidence_level"])
            data["confidence_level"] = max(0.0, min(1.0, value))
        except (ValueError, TypeError):
            data["confidence_level"] = 0.7
        
        # Risk level (enum)
        valid_levels = ["low", "medium", "high"]
        if data.get("risk_level") not in valid_levels:
            data["risk_level"] = "medium"
            self.logger.warning(
                "🔧 Исправлен некорректный risk_level: установлено 'medium'"
            )
        
        # Строковые поля
        string_fields = ["probability_reasoning", "impact_reasoning"]
        for field in string_fields:
            if not isinstance(data.get(field), str) or len(str(data[field]).strip()) < 5:
                data[field] = f"Автоматически сгенерированное обоснование для {field}"
        
        # Списковые поля
        list_fields = ["key_factors", "recommendations"]
        for field in list_fields:
            if not isinstance(data.get(field), list):
                data[field] = []
            else:
                # Очищаем список от пустых элементов
                data[field] = [
                    str(item).strip() for item in data[field] 
                    if item and str(item).strip()
                ][:10]  # Ограничиваем до 10 элементов
        
        return data

    def _validate_business_logic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидирует бизнес-логику оценки риска"""
        
        # Пересчитываем total_score для консистентности
        data["total_score"] = data["probability_score"] * data["impact_score"]
        
        # Корректируем risk_level на основе total_score
        total_score = data["total_score"]
        if total_score <= 6:
            correct_level = "low"
        elif total_score <= 14:
            correct_level = "medium"
        else:
            correct_level = "high"
        
        if data["risk_level"] != correct_level:
            old_level = data["risk_level"]
            data["risk_level"] = correct_level
            self.logger.debug(
                f"🔧 Скорректирован risk_level: {old_level} → {correct_level} (total_score: {total_score})"
            )
        
        return data
    
    def _get_default_evaluation_data(self, error_message: str) -> Dict[str, Any]:
        """УЛУЧШЕННЫЕ безопасные дефолтные данные для оценки"""
        return {
            "probability_score": 3,
            "impact_score": 3,
            "total_score": 9,
            "risk_level": "medium",
            "probability_reasoning": f"LLM не смог предоставить обоснование. Ошибка: {error_message}",
            "impact_reasoning": f"LLM не смог предоставить обоснование. Ошибка: {error_message}",
            "key_factors": ["Недостаточно данных для анализа"],
            "recommendations": ["Провести дополнительный анализ", "Улучшить качество входных данных"],
            "confidence_level": 0.3  # Низкая уверенность для fallback данных
        }
    
    def update_stats(self, execution_time: float, success: bool):
        """Обновление статистики агента"""
        self.stats["total_requests"] += 1
        self.stats["total_execution_time"] += execution_time
        
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        self.stats["average_response_time"] = (
            self.stats["total_execution_time"] / self.stats["total_requests"]
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики работы агента"""
        return self.stats.copy()


class AnalysisAgent(BaseAgent):
    """
    Базовый класс для агентов анализа
    Специализированный для задач анализа данных и извлечения информации
    """
    
    async def analyze_data(
        self,
        data: str,
        analysis_type: str,
        assessment_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Универсальный метод анализа данных
        
        Args:
            data: Данные для анализа
            analysis_type: Тип анализа
            assessment_id: ID оценки
            
        Returns:
            Результат анализа
        """
        prompt = f"Проведи {analysis_type} анализ предоставленных данных."
        
        return await self.call_llm_structured(
            data_to_analyze=data,
            extraction_prompt=prompt,
            assessment_id=assessment_id
        )


class EvaluationAgent(BaseAgent):
    """
    Базовый класс для агентов оценки
    Специализированный для задач оценки рисков
    """
    
    def __init__(self, config: AgentConfig, risk_type: str):
        super().__init__(config)
        self.risk_type = risk_type
    
    async def evaluate_risk(
        self,
        agent_data: Dict[str, Any],
        assessment_id: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Оценка риска определенного типа
        
        Args:
            agent_data: Данные об агенте
            assessment_id: ID оценки
            
        Returns:
            Результат оценки риска
        """
        evaluation_prompt = f"Оцени {self.risk_type} риски агента на основе предоставленных данных."
        
        agent_data_str = json.dumps(agent_data, ensure_ascii=False, indent=2)
        
        # Используем новый метод для структурированного вызова
        try:
            result = await self.call_llm_structured(
                data_to_analyze=agent_data_str,
                extraction_prompt=evaluation_prompt,
                assessment_id=assessment_id
            )
            
            # Дополнительная валидация специфично для оценки рисков
            result = self._ensure_required_fields(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка оценки {self.risk_type} риска: {e}")
            return self.create_fallback_result(str(e))
    
    def create_fallback_result(self, error_message: str) -> Dict[str, Any]:
        """Создание fallback результата при ошибках"""
        return self._get_default_evaluation_data(error_message)
    
    def validate_result(self, result_data: Dict[str, Any]) -> bool:
        """Валидация результата оценки риска"""
        required_fields = self._get_required_result_fields()
        
        # Проверяем наличие всех обязательных полей
        for field in required_fields:
            if field not in result_data:
                return False
        
        # Проверяем валидность значений
        try:
            prob_score = int(result_data["probability_score"])
            impact_score = int(result_data["impact_score"])
            
            if not (1 <= prob_score <= 5) or not (1 <= impact_score <= 5):
                return False
                
            if result_data["risk_level"] not in ["low", "medium", "high"]:
                return False
                
            return True
            
        except (ValueError, TypeError, KeyError):
            return False


# ===============================
# Фабрики для создания агентов (ОБНОВЛЕННЫЕ)
# ===============================

def create_agent_config(
    name: str,
    description: str,
    max_retries: int = 3,
    timeout_seconds: int = 120,
    use_risk_analysis_client: bool = False,
    llm_override: Optional[LLMConfig] = None
) -> AgentConfig:
    """
    Создание конфигурации агента (новая версия без LLM параметров)
    
    Args:
        name: Имя агента
        description: Описание агента
        max_retries: Максимум повторов
        timeout_seconds: Тайм-аут в секундах
        use_risk_analysis_client: Использовать специализированный клиент
        llm_override: Переопределение LLM конфигурации (для тестирования)
        
    Returns:
        Конфигурация агента
    """
    return AgentConfig(
        name=name,
        description=description,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        use_risk_analysis_client=use_risk_analysis_client,
        llm_override=llm_override
    )


def create_default_config() -> AgentConfig:
    """Создание конфигурации по умолчанию (использует центральный конфигуратор)"""
    return create_agent_config(
        name="default_agent",
        description="Агент с настройками по умолчанию",
        max_retries=3,
        timeout_seconds=120
    )


# Обратная совместимость - старые функции с LLM параметрами (DEPRECATED)
def create_agent_config_legacy(
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
    DEPRECATED: Создание конфигурации агента (старая версия для обратной совместимости)
    
    Используйте create_agent_config() без LLM параметров.
    LLM конфигурация теперь управляется централизованно.
    """
    import warnings
    warnings.warn(
        "create_agent_config_legacy deprecated. Use create_agent_config() without LLM params.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Создаем переопределение для тестирования/legacy кода
    llm_override = LLMConfig(
        base_url=llm_base_url,
        model=llm_model,
        temperature=temperature,
        timeout=timeout_seconds
    )
    
    return create_agent_config(
        name=name,
        description=description,
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        use_risk_analysis_client=use_risk_analysis_client,
        llm_override=llm_override
    )


def create_default_config_from_env() -> AgentConfig:
    """DEPRECATED: Используйте create_default_config()"""
    import warnings
    warnings.warn(
        "create_default_config_from_env deprecated. Use create_default_config().",
        DeprecationWarning,
        stacklevel=2
    )
    return create_default_config()


# Экспорт основных классов
__all__ = [
    "BaseAgent",
    "AnalysisAgent", 
    "EvaluationAgent",
    "AgentConfig",
    "create_agent_config",
    "create_default_config",
    # Legacy exports (deprecated)
    "create_agent_config_legacy",
    "create_default_config_from_env"
]