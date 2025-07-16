"""
Провайдер для GigaChat (заглушка)

Заготовка для будущей интеграции с GigaChat Max
"""

import os
from typing import Dict, Any
from .base_provider import BaseLLMProvider, ProviderConfig


class GigaChatProvider(BaseLLMProvider):
    """Провайдер для GigaChat (в разработке)"""
    
    def validate_config(self) -> None:
        """Валидация конфигурации GigaChat"""
        # TODO: Реализовать валидацию для GigaChat
        if not self.config.api_key:
            raise ValueError("GigaChat требует API ключа")
        
        # Заглушка - базовые проверки
        if not self.config.model:
            raise ValueError("GigaChat требует указания модели")
    
    def get_client_config(self) -> Dict[str, Any]:
        """Получение конфигурации для LLM клиента"""
        # TODO: Адаптировать под реальный API GigaChat
        return {
            "base_url": self.config.base_url or "https://gigachat.devices.sberbank.ru/api/v1",
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
            "retry_delay": self.config.retry_delay,
            "api_key": self.config.api_key,
            # Дополнительные параметры для GigaChat
            "headers": {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        }
    
    def is_available(self) -> bool:
        """Проверка доступности GigaChat"""
        # TODO: Реализовать реальную проверку доступности
        # Пока возвращаем False, так как это заглушка
        return False
    
    @classmethod
    def create_from_env(cls) -> 'GigaChatProvider':
        """Создание провайдера из переменных окружения"""
        config = ProviderConfig(
            provider_name="GigaChat",
            base_url=os.getenv("GIGACHAT_BASE_URL", "https://gigachat.devices.sberbank.ru/api/v1"),
            model=os.getenv("GIGACHAT_MODEL", "GigaChat-Max"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            timeout=int(os.getenv("LLM_TIMEOUT", "120")),
            max_retries=int(os.getenv("MAX_RETRY_COUNT", "3")),
            retry_delay=float(os.getenv("LLM_RETRY_DELAY", "1.0")),
            api_key=os.getenv("GIGACHAT_API_KEY")
        )
        
        return cls(config)