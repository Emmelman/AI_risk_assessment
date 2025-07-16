"""
Провайдер для LM Studio

Реализует интеграцию с локальным LM Studio сервером
"""

import os
import requests
from typing import Dict, Any
from .base_provider import BaseLLMProvider, ProviderConfig


class LMStudioProvider(BaseLLMProvider):
    """Провайдер для LM Studio"""
    
    def validate_config(self) -> None:
        """Валидация конфигурации LM Studio"""
        if not self.config.base_url:
            raise ValueError("LM Studio требует указания base_url")
        
        if not self.config.model:
            raise ValueError("LM Studio требует указания модели")
        
        # Проверяем, что URL имеет правильный формат
        if not self.config.base_url.startswith(('http://', 'https://')):
            raise ValueError("base_url должен начинаться с http:// или https://")
    
    def get_client_config(self) -> Dict[str, Any]:
        """Получение конфигурации для LLM клиента"""
        return {
            "base_url": self.config.base_url,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
            "retry_delay": self.config.retry_delay
        }
    
    def is_available(self) -> bool:
        """Проверка доступности LM Studio сервера"""
        try:
            # Проверяем доступность через /v1/models endpoint
            models_url = f"{self.config.base_url.rstrip('/')}/v1/models"
            response = requests.get(models_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    @classmethod
    def create_from_env(cls) -> 'LMStudioProvider':
        """Создание провайдера из переменных окружения"""
        config = ProviderConfig(
            provider_name="LM Studio",
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234"),
            model=os.getenv("LLM_MODEL", "qwen3-4b"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            timeout=int(os.getenv("LLM_TIMEOUT", "120")),
            max_retries=int(os.getenv("MAX_RETRY_COUNT", "3")),
            retry_delay=float(os.getenv("LLM_RETRY_DELAY", "1.0"))
        )
        
        return cls(config)
    
    def get_models_list(self) -> list:
        """Получение списка доступных моделей"""
        try:
            models_url = f"{self.config.base_url.rstrip('/')}/v1/models"
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model["id"] for model in data.get("data", [])]
            return []
        except Exception:
            return []