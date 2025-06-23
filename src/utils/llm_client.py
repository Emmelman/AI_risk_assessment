# src/utils/llm_client.py
"""
Клиент для работы с LLM моделями
Поддерживает qwen3-4b через LM Studio (localhost:1234)
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

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


@dataclass
class LLMConfig:
    """Конфигурация LLM клиента"""
    base_url: str = "http://127.0.0.1:1234"
    model: str = "qwen3-4b"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0


class LLMClient:
    """Асинхронный клиент для работы с LLM через OpenAI-совместимый API"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
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
        """
        Извлечение структурированных данных с улучшенной валидацией
        """
        
        system_prompt = f"""Ты - эксперт по извлечению структурированных данных.
        
        Твоя задача: {extraction_prompt}

        КРИТИЧЕСКИ ВАЖНО:
        - Отвечай ТОЛЬКО валидным {expected_format}
        - НЕ добавляй никаких комментариев, пояснений или тегов <think>
        - Если данных недостаточно, используй разумные дефолтные значения
        - Строго следуй указанной структуре
        - Все числовые поля должны быть числами, не строками
        - Все обязательные поля должны присутствовать

        ФОРМАТ ОТВЕТА: ТОЛЬКО чистый JSON без дополнительного текста"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.analyze_with_prompt(
                    system_prompt=system_prompt,
                    user_input=f"Данные для анализа:\n{data_to_analyze}",
                    model=model,
                    temperature=0.1  # Очень низкая температура для стабильности
                )
                
                # Улучшенная очистка ответа
                cleaned_content = response.content.strip()
                
                # Удаляем теги <think>...</think>
                if '<think>' in cleaned_content and '</think>' in cleaned_content:
                    start = cleaned_content.find('</think>') + 8
                    cleaned_content = cleaned_content[start:].strip()
                
                # Удаляем markdown блоки
                if cleaned_content.startswith("```json"):
                    cleaned_content = cleaned_content[7:]
                if cleaned_content.endswith("```"):
                    cleaned_content = cleaned_content[:-3]
                
                # Ищем JSON объект в тексте
                import re
                json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    json_str = cleaned_content.strip()
                
                # Пытаемся парсить
                result = json.loads(json_str)
                return result
                
            except json.JSONDecodeError as e:
                if attempt == max_retries - 1:
                    # Последняя попытка - возвращаем ошибку
                    raise LLMError(
                        f"Не удалось распарсить JSON после {max_retries} попыток: {e}\n"
                        f"Последний ответ: {response.content[:300]}..."
                    )
                
                # Повторяем с более строгим промптом
                system_prompt += f"\n\nВНИМАНИИ: Предыдущая попытка содержала ошибку JSON. Попытка {attempt + 1}/{max_retries}."
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise LLMError(f"Неожиданная ошибка при извлечении данных: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики использования"""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "error_count": self.error_count,
            "success_rate": (self.total_requests - self.error_count) / max(self.total_requests, 1),
            "avg_tokens_per_request": self.total_tokens / max(self.total_requests, 1)
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


# ===============================
# Фабрика клиентов
# ===============================

def create_llm_client(
    client_type: str = "standard",
    base_url: str = "http://127.0.0.1:1234",
    model: str = "qwen3-4b",
    temperature: float = 0.1
) -> LLMClient:
    """
    Фабрика для создания LLM клиентов
    
    Args:
        client_type: Тип клиента (standard, risk_analysis)
        base_url: URL LLM сервера
        model: Модель для использования
        temperature: Температура генерации
    """
    
    config = LLMConfig(
        base_url=base_url,
        model=model,
        temperature=temperature
    )
    
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
    """Получение глобального LLM клиента"""
    global _global_client
    
    if _global_client is None:
        # Получаем настройки из переменных окружения
        import os
        
        config = LLMConfig(
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234"),
            model=os.getenv("LLM_MODEL", "qwen3-4b"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096"))
        )
        
        _global_client = LLMClient(config)
        
        # Проверяем доступность
        if not await _global_client.health_check():
            raise LLMError("LLM сервер недоступен. Убедитесь, что LM Studio запущен на localhost:1234")
    
    return _global_client