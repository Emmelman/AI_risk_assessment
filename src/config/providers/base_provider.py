"""
Базовые классы для провайдеров LLM

Определяет интерфейс для различных провайдеров LLM сервисов
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import os


@dataclass
class ProviderConfig:
    """Базовая конфигурация провайдера LLM"""
    provider_name: str
    base_url: str
    model: str
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    api_key: Optional[str] = None
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class BaseLLMProvider(ABC):
    """Базовый класс для всех провайдеров LLM"""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.validate_config()
    
    @abstractmethod
    def validate_config(self) -> None:
        """Валидация конфигурации провайдера"""
        pass
    
    @abstractmethod
    def get_client_config(self) -> Dict[str, Any]:
        """Получение конфигурации для LLM клиента"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Проверка доступности провайдера"""
        pass
    
    @classmethod
    @abstractmethod
    def create_from_env(cls) -> 'BaseLLMProvider':
        """Создание провайдера из переменных окружения"""
        pass
    
    def get_display_name(self) -> str:
        """Получение отображаемого имени провайдера"""
        return f"{self.config.provider_name} ({self.config.model})"