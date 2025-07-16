# src/utils/llm_client.py
"""
Клиент для работы с LLM моделями
Поддерживает qwen3-4b через LM Studio (localhost:1234)
"""

import json
import asyncio
try:
    from langchain_gigachat import GigaChat
    GIGACHAT_AVAILABLE = True
except ImportError:
    GIGACHAT_AVAILABLE = False
    GigaChat = None

from .llm_config_manager import LLMProvider
from typing import Dict, List, Optional, Any, AsyncGenerator

from dataclasses import dataclass
from datetime import datetime
from .llm_config_manager import get_llm_config_manager
import httpx
from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """Сообщение для LLM"""
    role: str = Field(..., description="Роль: system, user, assistant")
    content: str = Field(..., description="Содержание сообщения")


class LLMResponse(BaseModel):
    """Ответ от LLM"""
    content: str = Field(..., description="Содержание ответа")
    finish_reason: str = Field(..., description="Причина завершения")
    usage: Dict[str, int] = Field(default_factory=dict, description="Статистика использования")
    model: str = Field(..., description="Использованная модель")
    created: datetime = Field(default_factory=datetime.now, description="Время создания")


class LLMError(Exception):
    """Исключение при работе с LLM"""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


# ТОЧЕЧНОЕ ИСПРАВЛЕНИЕ: Замените класс LLMConfig в llm_client.py

@dataclass
class LLMConfig:
    """Конфигурация LLM клиента"""
    base_url: str
    model: str  
    temperature: float
    max_tokens: int
    timeout: int
    max_retries: int
    retry_delay: float
    
    @classmethod
    def from_manager(cls, **overrides) -> 'LLMConfig':
        """Создание конфигурации из центрального менеджера с возможностью переопределения"""
        manager = get_llm_config_manager()
        config = manager.get_config()
        
        # Применяем переопределения если есть
        return cls(
            base_url=overrides.get('base_url', config.base_url),
            model=overrides.get('model', config.model),
            temperature=overrides.get('temperature', config.temperature),
            max_tokens=overrides.get('max_tokens', config.max_tokens),
            timeout=overrides.get('timeout', config.timeout),
            max_retries=overrides.get('max_retries', config.max_retries),
            retry_delay=overrides.get('retry_delay', config.retry_delay)
        )
    
    @classmethod 
    def create_default(cls) -> 'LLMConfig':
        """Создание конфигурации по умолчанию (fallback для обратной совместимости)"""
        return cls(
            base_url="http://127.0.0.1:1234",
            model="qwen3-4b", 
            temperature=0.1,
            max_tokens=4096,
            timeout=120,
            max_retries=3,
            retry_delay=1.0
        )

class LLMClient:
    """Асинхронный клиент для работы с LLM через OpenAI-совместимый API"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            try:
                self.config = LLMConfig.from_manager()
            except Exception:
                # Fallback для случаев когда конфигуратор недоступен
                self.config = LLMConfig.create_default()
        else:
            self.config = config
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout
        )
        
        # Статистика
        self.total_requests = 0
        self.total_tokens = 0
        self.error_count = 0
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Закрытие клиента"""
        await self.client.aclose()
    
    async def health_check(self) -> bool:
        """Проверка доступности LLM сервера"""
        try:
            response = await self.client.get("/v1/models", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    async def get_available_models(self) -> List[str]:
        """Получение списка доступных моделей"""
        try:
            response = await self.client.get("/v1/models")
            response.raise_for_status()
            
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except Exception as e:
            raise LLMError(f"Ошибка получения списка моделей: {str(e)}")
    
    async def complete_chat(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> LLMResponse:
        """
        Выполнение chat completion запроса
        
        Args:
            messages: Список сообщений для модели
            model: Модель для использования (по умолчанию из config)
            temperature: Температура генерации
            max_tokens: Максимум токенов
            stream: Потоковый режим
        """
        
        # Подготовка параметров
        request_data = {
            "model": model or self.config.model,
            "messages": [msg.dict() for msg in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": stream
        }
        
        # Выполнение запроса с повторами
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.post(
                    "/v1/chat/completions",
                    json=request_data,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Парсинг ответа
                choice = data["choices"][0]
                usage = data.get("usage", {})
                
                # Обновление статистики
                self.total_requests += 1
                self.total_tokens += usage.get("total_tokens", 0)
                
                return LLMResponse(
                    content=choice["message"]["content"],
                    finish_reason=choice["finish_reason"],
                    usage=usage,
                    model=data["model"],
                    created=datetime.now()
                )
                
            except httpx.HTTPStatusError as e:
                self.error_count += 1
                error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
                
                if attempt == self.config.max_retries - 1:
                    raise LLMError(
                        f"Ошибка запроса к LLM после {self.config.max_retries} попыток: {error_msg}",
                        status_code=e.response.status_code,
                        response_data=e.response.json() if e.response.text else None
                    )
                
                # Ждем перед повтором
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                
            except Exception as e:
                self.error_count += 1
                if attempt == self.config.max_retries - 1:
                    raise LLMError(f"Неожиданная ошибка: {str(e)}")
                
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
    
    async def complete_chat_stream(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Потоковый chat completion
        
        Yields:
            Фрагменты текста ответа
        """
        
        request_data = {
            "model": model or self.config.model,
            "messages": [msg.dict() for msg in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": True
        }
        
        try:
            async with self.client.stream(
                "POST", 
                "/v1/chat/completions",
                json=request_data
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Убираем "data: "
                        
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            choice = data["choices"][0]
                            
                            if "delta" in choice and "content" in choice["delta"]:
                                content = choice["delta"]["content"]
                                if content:
                                    yield content
                                    
                        except json.JSONDecodeError:
                            # Пропускаем некорректные JSON строки
                            continue
                            
        except Exception as e:
            raise LLMError(f"Ошибка потокового запроса: {str(e)}")
    
    async def analyze_with_prompt(
        self,
        system_prompt: str,
        user_input: str,
        context: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> LLMResponse:
        """
        Упрощенный метод для анализа с промптом
        
        Args:
            system_prompt: Системный промпт с инструкциями
            user_input: Пользовательский ввод для анализа
            context: Дополнительный контекст
            model: Модель для использования
            temperature: Температура генерации
        """
        
        messages = [
            LLMMessage(role="system", content=system_prompt)
        ]
        
        # Добавляем контекст если есть
        if context:
            messages.append(
                LLMMessage(
                    role="user", 
                    content=f"Контекст:\n{context}\n\nЗадача:\n{user_input}"
                )
            )
        else:
            messages.append(LLMMessage(role="user", content=user_input))
        
        return await self.complete_chat(
            messages=messages,
            model=model,
            temperature=temperature
        )
    
    async def extract_structured_data(
        self,
        data_to_analyze: str,
        extraction_prompt: str,
        expected_format: str = "JSON",
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """МАКСИМАЛЬНО НАДЕЖНОЕ извлечение структурированных данных"""
        
        system_prompt = f"""Ты - эксперт по анализу данных. 

        Твоя задача: {extraction_prompt}

        КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ:
        - Отвечай ТОЛЬКО валидным {expected_format} без дополнительного текста
        - НЕ добавляй комментарии, пояснения, теги <think> или markdown блоки
        - Если данных недостаточно, используй разумные значения по умолчанию
        - Строго следуй указанной структуре данных
        - Все числовые поля ОБЯЗАТЕЛЬНО должны быть числами, не строками
        - Все обязательные поля должны присутствовать
        - НЕ используй запятые в конце объектов или массивов

        ПРИМЕР ПРАВИЛЬНОГО ФОРМАТА:
        {{
            "probability_score": 3,
            "impact_score": 4,
            "total_score": 12,
            "risk_level": "medium",
            "probability_reasoning": "Обоснование вероятности",
            "impact_reasoning": "Обоснование воздействия",
            "key_factors": ["фактор1", "фактор2"],
            "recommendations": ["рекомендация1", "рекомендация2"],
            "confidence_level": 0.8
        }}

        СТРОГО: отвечай только JSON, начинающийся с {{ и заканчивающийся }}"""

        max_retries = 4  # Увеличиваем количество попыток
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = await self.analyze_with_prompt(
                    system_prompt=system_prompt,
                    user_input=f"Данные для анализа:\n{data_to_analyze}",
                    model=model,
                    temperature=0.05 if attempt == 0 else 0.1  # Очень низкая температура для первой попытки
                )
                
                # КРИТИЧЕСКИ УЛУЧШЕННАЯ очистка ответа
                parsed_result = self._ultra_robust_json_parser(response.content)
                
                # Дополнительная валидация и автоисправление
                validated_result = self._validate_and_fix_json_structure(parsed_result)
                
                return validated_result
                
            except Exception as e:
                last_error = e
                
                if attempt < max_retries - 1:
                    # Модифицируем промпт для следующей попытки
                    system_prompt += f"\n\nВНИМАНИЕ: Попытка {attempt + 1} из {max_retries}. Предыдущая ошибка: {str(e)[:100]}. Будь ОСОБЕННО внимательным к формату JSON!"
                    
                    # Увеличиваем паузу между попытками
                    await asyncio.sleep(1 + attempt)
                else:
                    # Последняя попытка не удалась - возвращаем fallback
                    return self._create_emergency_fallback_result(extraction_prompt, str(e))
        
        # Не должно достигаться, но на всякий случай
        return self._create_emergency_fallback_result(extraction_prompt, str(last_error))
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики использования"""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "error_count": self.error_count,
            "success_rate": (self.total_requests - self.error_count) / max(self.total_requests, 1),
            "avg_tokens_per_request": self.total_tokens / max(self.total_requests, 1)
        }
    def _ultra_robust_json_parser(self, content: str) -> Dict[str, Any]:
        """Максимально надежный парсер JSON с множественными стратегиями"""
        
        import re
        import json
        
        # Стратегия 1: Базовая очистка
        cleaned = content.strip()
        
        # Удаляем теги <think>...</think>
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
        
        # Удаляем markdown блоки
        if '```json' in cleaned:
            # Находим все JSON блоки
            json_blocks = re.findall(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
            if json_blocks:
                cleaned = json_blocks[-1].strip()
            else:
                # Если нет закрывающего тега
                start = cleaned.find('```json') + 7
                cleaned = cleaned[start:].strip()
        
        # Удаляем обычные markdown блоки
        cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL).strip()
        
        # Стратегия 2: Поиск JSON объекта
        strategies = [
            # 1. Прямой парсинг
            lambda x: json.loads(x),
            
            # 2. Поиск по фигурным скобкам
            lambda x: json.loads(self._extract_json_by_braces(x)),
            
            # 3. Поиск по регулярному выражению
            lambda x: json.loads(self._extract_json_by_regex(x)),
            
            # 4. Исправление и парсинг
            lambda x: json.loads(self._fix_common_json_issues(x)),
            
            # 5. Агрессивное исправление
            lambda x: json.loads(self._aggressive_json_fix(x))
        ]
        
        for i, strategy in enumerate(strategies):
            try:
                result = strategy(cleaned)
                if isinstance(result, dict):
                    return result
            except Exception as e:
                if i == len(strategies) - 1:
                    # Последняя стратегия не сработала
                    raise Exception(f"Все стратегии парсинга не удались. Последняя ошибка: {e}")
                continue
        
        raise Exception("Невозможно распарсить JSON ни одной стратегией")
    def _extract_json_by_braces(self, content: str) -> str:
        """Извлекает JSON по фигурным скобкам"""
        
        # Ищем первую открывающую скобку
        start = content.find('{')
        if start == -1:
            raise ValueError("Не найдена открывающая фигурная скобка")
        
        # Ищем соответствующую закрывающую скобку
        brace_count = 0
        end = start
        
        for i, char in enumerate(content[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break
        
        if brace_count != 0:
            # Не найдена закрывающая скобка, берем до конца
            end = len(content) - 1
        
        return content[start:end + 1]

    def _extract_json_by_regex(self, content: str) -> str:
        """Извлекает JSON с помощью регулярного выражения"""
        
        import re
        
        # Паттерн для поиска JSON объекта
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        
        matches = re.findall(json_pattern, content, re.DOTALL)
        if matches:
            # Берем самый длинный найденный JSON
            return max(matches, key=len)
        
        raise ValueError("JSON объект не найден регулярным выражением")

    def _fix_common_json_issues(self, content: str) -> str:
        """Исправляет распространенные ошибки JSON"""
        
        import re
        
        # 1. Убираем trailing commas
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        
        # 2. Исправляем одинарные кавычки на двойные
        content = re.sub(r"'([^']*)':", r'"\1":', content)  # Ключи
        content = re.sub(r":\s*'([^']*)'", r': "\1"', content)  # Значения
        
        # 3. Добавляем кавычки к значениям без кавычек (кроме чисел, bool, null)
        content = re.sub(r':\s*([^",{\[\]\s][^,}\]]*[^",}\]\s])\s*[,}]', 
                        lambda m: f': "{m.group(1).strip()}"' + m.group(0)[-1], content)
        
        # 4. Исправляем неэкранированные кавычки внутри строк
        content = re.sub(r'(?<!\\)"(?=[^,}\]]*[,}\]])', '\\"', content)
        
        # 5. Убираем комментарии
        content = re.sub(r'//.*?\n', '\n', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # 6. Исправляем множественные пробелы
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()

    def _aggressive_json_fix(self, content: str) -> str:
        """Агрессивное исправление JSON - последняя попытка"""
        
        import re
        
        # Начинаем с базового исправления
        content = self._fix_common_json_issues(content)
        
        # Если не начинается с {, добавляем
        if not content.strip().startswith('{'):
            content = '{' + content
        
        # Если не заканчивается на }, добавляем
        if not content.strip().endswith('}'):
            content = content + '}'
        
        # Пытаемся найти и исправить незакрытые кавычки
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Считаем кавычки в строке
            quote_count = line.count('"') - line.count('\\"')
            
            # Если нечетное количество кавычек, добавляем недостающую
            if quote_count % 2 == 1:
                if ':' in line and not line.strip().endswith('"'):
                    line = line.rstrip() + '"'
            
            fixed_lines.append(line)
        
        content = '\n'.join(fixed_lines)
        
        # Последняя попытка - убираем все до первой { и после последней }
        start = content.find('{')
        end = content.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            content = content[start:end + 1]
        
        return content

    def _validate_and_fix_json_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидирует и исправляет структуру JSON результата"""
        
        # Определяем обязательные поля для разных типов запросов
        if self._looks_like_risk_evaluation(data):
            return self._fix_risk_evaluation_structure(data)
        elif self._looks_like_critic_evaluation(data):
            return self._fix_critic_evaluation_structure(data)
        else:
            # Общая валидация
            return self._fix_general_structure(data)

    def _looks_like_risk_evaluation(self, data: Dict[str, Any]) -> bool:
        """Определяет, является ли это оценкой риска"""
        
        risk_fields = {"probability_score", "impact_score", "total_score", "risk_level"}
        return any(field in data for field in risk_fields)

    def _looks_like_critic_evaluation(self, data: Dict[str, Any]) -> bool:
        """Определяет, является ли это критической оценкой"""
        
        critic_fields = {"quality_score", "is_acceptable", "critic_reasoning"}
        return any(field in data for field in critic_fields)

    def _fix_risk_evaluation_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Исправляет структуру оценки риска"""
        
        # Обязательные поля с дефолтными значениями
        required_fields = {
            "probability_score": 3,
            "impact_score": 3,
            "total_score": 9,
            "risk_level": "medium",
            "probability_reasoning": "Обоснование не предоставлено",
            "impact_reasoning": "Обоснование не предоставлено",
            "key_factors": [],
            "recommendations": [],
            "confidence_level": 0.7
        }
        
        # Добавляем отсутствующие поля
        for field, default_value in required_fields.items():
            if field not in data or data[field] is None:
                data[field] = default_value
        
        # Валидируем типы и диапазоны
        data["probability_score"] = self._ensure_int_range(data["probability_score"], 1, 5, 3)
        data["impact_score"] = self._ensure_int_range(data["impact_score"], 1, 5, 3)
        data["confidence_level"] = self._ensure_float_range(data["confidence_level"], 0.0, 1.0, 0.7)
        
        # Пересчитываем total_score
        data["total_score"] = data["probability_score"] * data["impact_score"]
        
        # Корректируем risk_level
        total_score = data["total_score"]
        if total_score <= 6:
            data["risk_level"] = "low"
        elif total_score <= 14:
            data["risk_level"] = "medium"
        else:
            data["risk_level"] = "high"
        
        # Валидируем списки
        data["key_factors"] = self._ensure_string_list(data["key_factors"])
        data["recommendations"] = self._ensure_string_list(data["recommendations"])
        
        # Валидируем строки
        data["probability_reasoning"] = self._ensure_string(data["probability_reasoning"], "Обоснование не предоставлено")
        data["impact_reasoning"] = self._ensure_string(data["impact_reasoning"], "Обоснование не предоставлено")
        
        return data

    def _fix_critic_evaluation_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Исправляет структуру критической оценки"""
        
        required_fields = {
            "quality_score": 5.0,
            "is_acceptable": True,
            "issues_found": [],
            "improvement_suggestions": [],
            "critic_reasoning": "Обоснование не предоставлено"
        }
        
        for field, default_value in required_fields.items():
            if field not in data or data[field] is None:
                data[field] = default_value
        
        # Валидация
        data["quality_score"] = self._ensure_float_range(data["quality_score"], 0.0, 10.0, 5.0)
        data["is_acceptable"] = bool(data.get("is_acceptable", True))
        data["issues_found"] = self._ensure_string_list(data["issues_found"])
        data["improvement_suggestions"] = self._ensure_string_list(data["improvement_suggestions"])
        data["critic_reasoning"] = self._ensure_string(data["critic_reasoning"], "Обоснование не предоставлено")
        
        return data

    def _fix_general_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Общее исправление структуры"""
        
        # Просто возвращаем данные, убедившись что это dict
        if not isinstance(data, dict):
            return {"error": "Неверный формат данных", "original_data": str(data)}
        
        return data

    # Вспомогательные методы валидации

    def _ensure_int_range(self, value: Any, min_val: int, max_val: int, default: int) -> int:
        """Обеспечивает целое число в заданном диапазоне"""
        try:
            int_val = int(float(value))  # Сначала float, потом int для обработки "3.0"
            return max(min_val, min(max_val, int_val))
        except (ValueError, TypeError):
            return default

    def _ensure_float_range(self, value: Any, min_val: float, max_val: float, default: float) -> float:
        """Обеспечивает число с плавающей точкой в заданном диапазоне"""
        try:
            float_val = float(value)
            return max(min_val, min(max_val, float_val))
        except (ValueError, TypeError):
            return default

    def _ensure_string(self, value: Any, default: str) -> str:
        """Обеспечивает строковое значение"""
        if not value or not isinstance(value, str) or len(value.strip()) < 3:
            return default
        return str(value).strip()

    def _ensure_string_list(self, value: Any) -> List[str]:
        """Обеспечивает список строк"""
        if not isinstance(value, list):
            return []
        
        result = []
        for item in value:
            if item and isinstance(item, str) and len(item.strip()) > 0:
                result.append(str(item).strip())
        
        return result[:10]  # Ограничиваем до 10 элементов

    def _create_emergency_fallback_result(self, extraction_prompt: str, error_message: str) -> Dict[str, Any]:
        """Создает аварийный fallback результат когда все стратегии не удались"""
        
        # Пытаемся определить тип запроса по промпту
        prompt_lower = extraction_prompt.lower()
        
        if any(keyword in prompt_lower for keyword in ['риск', 'risk', 'оцен', 'evaluat']):
            # Это похоже на оценку риска
            return {
                "probability_score": 3,
                "impact_score": 3,
                "total_score": 9,
                "risk_level": "medium",
                "probability_reasoning": f"Аварийная оценка: LLM вернул некорректный JSON. Ошибка: {error_message}",
                "impact_reasoning": f"Аварийная оценка: LLM вернул некорректный JSON. Ошибка: {error_message}",
                "key_factors": ["Ошибка парсинга ответа LLM"],
                "recommendations": ["Проверить качество промпта", "Повторить оценку", "Проверить настройки LLM"],
                "confidence_level": 0.1
            }
        
        elif any(keyword in prompt_lower for keyword in ['критик', 'critic', 'качеств', 'quality']):
            # Это похоже на критическую оценку
            return {
                "quality_score": 3.0,
                "is_acceptable": False,
                "issues_found": ["LLM вернул некорректный JSON", f"Ошибка парсинга: {error_message}"],
                "improvement_suggestions": ["Улучшить промпт", "Проверить настройки LLM", "Повторить оценку"],
                "critic_reasoning": f"Аварийная оценка качества: не удалось распарсить ответ LLM. Ошибка: {error_message}"
            }
        
        else:
            # Общий fallback
            return {
                "error": "Ошибка парсинга LLM ответа",
                "error_message": error_message,
                "extraction_prompt": extraction_prompt,
                "fallback_response": True
            }

# ===============================
# Специализированные клиенты для агентов
# ===============================

class RiskAnalysisLLMClient(LLMClient):
    """Специализированный клиент для анализа рисков"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        
        # Специальные настройки для анализа рисков
        if self.config.temperature > 0.3:
            self.config.temperature = 0.2  # Более детерминированные ответы
    
    async def evaluate_risk(
        self,
        risk_type: str,
        agent_data: str,
        evaluation_criteria: str,
        examples: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Оценка риска с использованием специализированного промпта
        
        Args:
            risk_type: Тип риска для оценки
            agent_data: Данные об агенте
            evaluation_criteria: Критерии оценки
            examples: Примеры для контекста
        """
        
        system_prompt = f"""Ты - эксперт по оценке операционных рисков ИИ-агентов в банковской сфере.

Твоя задача: оценить {risk_type} для предоставленного ИИ-агента.

КРИТЕРИИ ОЦЕНКИ:
{evaluation_criteria}

ШКАЛА ОЦЕНКИ:
- Вероятность: 1-5 баллов (1=низкая, 5=высокая)
- Тяжесть: 1-5 баллов (1=низкие потери, 5=высокие потери)
- Итоговый балл = Вероятность × Тяжесть

ФОРМАТ ОТВЕТА (СТРОГО JSON):
{{
    "probability_score": <1-5>,
    "impact_score": <1-5>,
    "total_score": <1-25>,
    "risk_level": "<low|medium|high>",
    "probability_reasoning": "<подробное обоснование вероятности>",
    "impact_reasoning": "<подробное обоснование тяжести>",
    "key_factors": ["<фактор1>", "<фактор2>", ...],
    "recommendations": ["<рекомендация1>", "<рекомендация2>", ...],
    "confidence_level": <0.0-1.0>
}}

УРОВНИ РИСКА:
- low: 1-6 баллов
- medium: 7-14 баллов  
- high: 15-25 баллов"""

        if examples:
            system_prompt += f"\n\nПРИМЕРЫ ОЦЕНОК:\n{examples}"

        response = await self.extract_structured_data(
            data_to_analyze=agent_data,
            extraction_prompt=f"Оцени {risk_type} согласно методике",
            expected_format="JSON"
        )
        
        # Валидация обязательных полей
        required_fields = [
            "probability_score", "impact_score", "total_score", 
            "risk_level", "probability_reasoning", "impact_reasoning"
        ]
        
        for field in required_fields:
            if field not in response:
                raise LLMError(f"Отсутствует обязательное поле в ответе: {field}")
        
        # Валидация значений
        if not (1 <= response["probability_score"] <= 5):
            raise LLMError(f"Некорректный probability_score: {response['probability_score']}")
        
        if not (1 <= response["impact_score"] <= 5):
            raise LLMError(f"Некорректный impact_score: {response['impact_score']}")
        
        # Автоматический пересчет total_score для надежности
        response["total_score"] = response["probability_score"] * response["impact_score"]
        
        # Автоматическое определение risk_level
        score = response["total_score"]
        if score <= 6:
            response["risk_level"] = "low"
        elif score <= 14:
            response["risk_level"] = "medium"
        else:
            response["risk_level"] = "high"
        
        return response
    
    async def critique_evaluation(
        self,
        risk_type: str,
        original_evaluation: Dict[str, Any],
        agent_data: str,
        quality_threshold: float = 7.0
    ) -> Dict[str, Any]:
        """
        Критика оценки другого агента
        
        Args:
            risk_type: Тип риска
            original_evaluation: Оригинальная оценка для критики
            agent_data: Данные об агенте
            quality_threshold: Порог качества (0-10)
        """
        
        system_prompt = f"""Ты - критик-эксперт по оценке качества анализа рисков ИИ-агентов.

Твоя задача: оценить качество предоставленной оценки {risk_type}.

КРИТЕРИИ КАЧЕСТВА:
1. Обоснованность оценок (соответствие данным агента)
2. Полнота анализа (учтены ли все аспекты)
3. Логичность рассуждений
4. Практичность рекомендаций
5. Соответствие методике оценки

ШКАЛА КАЧЕСТВА: 0-10 баллов
ПОРОГ ПРИЕМЛЕМОСТИ: {quality_threshold} баллов

ФОРМАТ ОТВЕТА (СТРОГО JSON):
{{
    "quality_score": <0.0-10.0>,
    "is_acceptable": <true|false>,
    "issues_found": ["<проблема1>", "<проблема2>", ...],
    "improvement_suggestions": ["<предложение1>", "<предложение2>", ...],
    "critic_reasoning": "<подробное обоснование оценки качества>"
}}"""

        evaluation_text = json.dumps(original_evaluation, ensure_ascii=False, indent=2)
        
        context = f"""ДАННЫЕ ОБ АГЕНТЕ:
{agent_data}

ОЦЕНКА ДЛЯ КРИТИКИ:
{evaluation_text}"""

        response = await self.extract_structured_data(
            data_to_analyze=context,
            extraction_prompt="Критически оцени качество представленной оценки риска",
            expected_format="JSON"
        )
        
        # Валидация
        required_fields = ["quality_score", "is_acceptable", "critic_reasoning"]
        for field in required_fields:
            if field not in response:
                raise LLMError(f"Отсутствует обязательное поле: {field}")
        
        # Автоматическое определение приемлемости
        response["is_acceptable"] = response["quality_score"] >= quality_threshold
        
        return response

class GigaChatLLMClient(LLMClient):
    """Специализированный клиент для работы с GigaChat через langchain_gigachat"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        # НЕ вызываем super().__init__() так как не используем httpx для GigaChat
        self.config = config or LLMConfig.from_manager()
        
        if not GIGACHAT_AVAILABLE:
            raise ImportError(
                "langchain_gigachat не установлен! Установите: pip install langchain-gigachat"
            )
        
        if self.config.provider != LLMProvider.GIGACHAT:
            raise ValueError("GigaChatLLMClient требует provider=GIGACHAT")
        
        # Проверяем наличие сертификатов
        if not (self.config.cert_file and self.config.key_file):
            raise ValueError("Для GigaChat необходимы cert_file и key_file")
        
        # Создаем GigaChat клиент
        self.gigachat = GigaChat(
            base_url=self.config.base_url,
            cert_file=self.config.cert_file,
            key_file=self.config.key_file,
            model=self.config.model,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            verify_ssl_certs=self.config.verify_ssl_certs,
            profanity_check=self.config.profanity_check,
            streaming=self.config.streaming
        )
        
        # Статистика
        self.total_requests = 0
        self.total_tokens = 0
        self.error_count = 0
    
    async def health_check(self) -> bool:
        """Проверка доступности GigaChat"""
        try:
            # Простой тестовый запрос
            response = self.gigachat.invoke("Привет")
            return bool(response and len(response.content) > 0)
        except Exception:
            return False
    
    async def get_available_models(self) -> List[str]:
        """Получение списка доступных моделей для GigaChat"""
        # GigaChat обычно поддерживает фиксированный набор моделей
        return ["GigaChat", "GigaChat-Pro", "GigaChat-Max"]
    
    async def complete_chat(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> LLMResponse:
        """Выполнение chat completion через GigaChat"""
        
        try:
            self.total_requests += 1
            
            # Формируем единый промпт из сообщений для GigaChat
            prompt = self._format_messages_for_gigachat(messages)
            
            # Временно изменяем параметры модели если нужно
            original_temp = self.gigachat.temperature
            original_model = self.gigachat.model
            
            if temperature is not None:
                self.gigachat.temperature = temperature
            if model is not None:
                self.gigachat.model = model
            
            try:
                # Выполняем запрос
                response = self.gigachat.invoke(prompt)
                
                # Извлекаем контент
                if hasattr(response, 'content'):
                    content = response.content
                else:
                    content = str(response)
                
                # Примерная оценка токенов (так как GigaChat может не возвращать точное число)
                estimated_tokens = len(prompt.split()) + len(content.split())
                self.total_tokens += estimated_tokens
                
                return LLMResponse(
                    content=content,
                    finish_reason="stop",
                    usage={"total_tokens": estimated_tokens, "estimated": True},
                    model=model or self.config.model,
                    created=datetime.now()
                )
                
            finally:
                # Восстанавливаем оригинальные параметры
                self.gigachat.temperature = original_temp
                self.gigachat.model = original_model
                
        except Exception as e:
            self.error_count += 1
            raise LLMError(f"Ошибка GigaChat: {str(e)}")
    
    def _format_messages_for_gigachat(self, messages: List[LLMMessage]) -> str:
        """Форматирование сообщений в единый промпт для GigaChat"""
        formatted_parts = []
        
        for message in messages:
            role = message.role
            content = message.content
            
            if role == "system":
                formatted_parts.append(f"Системная инструкция: {content}")
            elif role == "user":
                formatted_parts.append(f"Пользователь: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Ассистент: {content}")
            else:
                formatted_parts.append(content)
        
        return "\n\n".join(formatted_parts)
    
    async def simple_completion(self, prompt: str, **kwargs) -> str:
        """Простой интерфейс для получения ответа на промпт"""
        try:
            response = self.gigachat.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            raise LLMError(f"Ошибка GigaChat простого запроса: {str(e)}")
    
    async def extract_structured_data(
        self,
        data_to_analyze: str,
        extraction_prompt: str,
        expected_format: str = "JSON",
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """Извлечение структурированных данных через GigaChat"""
        
        system_prompt = f"""Ты - эксперт по анализу данных. 

        Твоя задача: {extraction_prompt}

        КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ:
        - Отвечай ТОЛЬКО валидным {expected_format} без дополнительного текста
        - НЕ добавляй комментарии, пояснения, теги <think> или markdown блоки
        - Если данных недостаточно, используй разумные значения по умолчанию
        - Строго следуй указанной структуре данных
        - Все числовые поля ОБЯЗАТЕЛЬНО должны быть числами, не строками
        - Все обязательные поля должны присутствовать
        - НЕ используй запятые в конце объектов или массивов

        СТРОГО: отвечай только JSON, начинающийся с {{ и заканчивающийся }}"""

        last_error = None
        
        for attempt in range(max_attempts):
            try:
                messages = [LLMMessage(role="user", content=f"{system_prompt}\n\nДанные для анализа:\n{data_to_analyze}")]
                response = await self.complete_chat(messages, temperature=0.05 if attempt == 0 else 0.1)
                
                # Используем тот же парсер что и в базовом классе
                parsed_result = self._ultra_robust_json_parser(response.content)
                validated_result = self._validate_and_fix_json_structure(parsed_result)
                
                return validated_result
                
            except Exception as e:
                last_error = e
                
                if attempt < max_attempts - 1:
                    # Небольшая пауза перед повтором
                    await asyncio.sleep(1 + attempt)
                else:
                    # Последняя попытка не удалась - возвращаем fallback
                    return self._create_emergency_fallback_result(extraction_prompt, str(e))
        
        return self._create_emergency_fallback_result(extraction_prompt, str(last_error))
    
    async def close(self):
        """Закрытие клиента GigaChat"""
        # GigaChat клиент из langchain не требует явного закрытия
        pass

# ===============================
# Фабрика клиентов
# ===============================

def create_llm_client(
    client_type: str = "standard",
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None
) -> LLMClient:
    """
    Фабрика для создания LLM клиентов
    ОБНОВЛЕНО: Поддержка GigaChat
    """
    
    # Получаем конфигурацию из центрального менеджера
    overrides = {}
    if base_url is not None:
        overrides['base_url'] = base_url
    if model is not None:
        overrides['model'] = model
    if temperature is not None:
        overrides['temperature'] = temperature
    
    config = LLMConfig.from_manager(**overrides)
    
    # Определяем какой клиент создавать
    if config.provider == LLMProvider.GIGACHAT:
        # Для GigaChat всегда используем специализированный клиент
        return GigaChatLLMClient(config)
    else:
        # Для LM Studio и OpenAI используем обычные клиенты
        if client_type == "risk_analysis":
            return RiskAnalysisLLMClient(config)
        else:
            return LLMClient(config)


# ===============================
# Глобальный экземпляр
# ===============================

# Глобальный клиент для переиспользования
_global_client: Optional[LLMClient] = None

async def get_llm_client() -> LLMClient:
    """
    Получение глобального LLM клиента
    ОБНОВЛЕНО: Поддержка GigaChat
    """
    global _global_client
    
    if _global_client is None:
        config = LLMConfig.from_manager()
        
        # Создаем клиент в зависимости от провайдера
        if config.provider == LLMProvider.GIGACHAT:
            _global_client = GigaChatLLMClient(config)
        else:
            _global_client = LLMClient(config)
        
        # Проверяем доступность
        if not await _global_client.health_check():
            provider_name = config.provider.value
            raise LLMError(f"{provider_name} сервер недоступен. Проверьте настройки подключения.")
    
    return _global_client

__all__ = [
    "LLMClient",
    "GigaChatLLMClient",  # ← ДОБАВИТЬ ЭТУ СТРОКУ
    "RiskAnalysisLLMClient", 
    "LLMConfig",
    "LLMMessage",
    "LLMResponse",
    "LLMError",
    "create_llm_client",
    "get_llm_client"
]