"""
Центральный менеджер конфигурации LLM

Единая точка управления всеми настройками LLM в системе.
Обеспечивает централизованную конфигурацию и легкое переключение между провайдерами.
"""

import os
import logging
from typing import Dict, Any, Optional, Type
from dataclasses import dataclass

from .providers import BaseLLMProvider, AVAILABLE_PROVIDERS, LMStudioProvider


logger = logging.getLogger(__name__)


@dataclass
class LLMClientConfig:
    """Унифицированная конфигурация для LLM клиентов"""
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    max_retries: int
    retry_delay: float
    api_key: Optional[str] = None
    additional_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class LLMConfigManager:
    """
    Центральный менеджер конфигурации LLM
    
    Управляет всеми настройками LLM в системе, включая:
    - Выбор провайдера (LM Studio, GigaChat, etc.)
    - Конфигурацию моделей
    - Параметры подключения
    - Валидацию настроек
    """
    
    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        """
        Инициализация менеджера конфигурации
        
        Args:
            provider: Провайдер LLM. Если None, создается из переменных окружения
        """
        self._provider = provider
        self._config_cache: Optional[LLMClientConfig] = None
        
        if self._provider is None:
            self._provider = self._create_provider_from_env()
        
        logger.info(f"LLM Config Manager инициализирован с провайдером: {self._provider.get_display_name()}")
    
    def _create_provider_from_env(self) -> BaseLLMProvider:
        """Создание провайдера из переменных окружения"""
        provider_name = os.getenv("LLM_PROVIDER", "lm_studio").lower()
        
        if provider_name not in AVAILABLE_PROVIDERS:
            logger.warning(f"Неизвестный провайдер '{provider_name}', используется LM Studio по умолчанию")
            provider_name = "lm_studio"
        
        provider_class = AVAILABLE_PROVIDERS[provider_name]
        provider = provider_class.create_from_env()
        
        return provider
    
    def get_config(self, force_refresh: bool = False) -> LLMClientConfig:
        """
        Получение конфигурации для LLM клиентов
        
        Args:
            force_refresh: Принудительное обновление кэша конфигурации
            
        Returns:
            Унифицированная конфигурация для LLM клиентов
        """
        if self._config_cache is None or force_refresh:
            self._config_cache = self._build_config()
        
        return self._config_cache
    
    def _build_config(self) -> LLMClientConfig:
        """Построение конфигурации из текущего провайдера"""
        provider_config = self._provider.get_client_config()
        
        return LLMClientConfig(
            base_url=provider_config["base_url"],
            model=provider_config["model"],
            temperature=provider_config["temperature"],
            max_tokens=provider_config["max_tokens"],
            timeout=provider_config["timeout"],
            max_retries=provider_config["max_retries"],
            retry_delay=provider_config["retry_delay"],
            api_key=provider_config.get("api_key"),
            additional_params={
                k: v for k, v in provider_config.items() 
                if k not in ["base_url", "model", "temperature", "max_tokens", 
                           "timeout", "max_retries", "retry_delay", "api_key"]
            }
        )
    
    def get_provider(self) -> BaseLLMProvider:
        """Получение текущего провайдера"""
        return self._provider
    
    def set_provider(self, provider: BaseLLMProvider) -> None:
        """
        Установка нового провайдера
        
        Args:
            provider: Новый провайдер LLM
        """
        self._provider = provider
        self._config_cache = None  # Сбрасываем кэш
        logger.info(f"Провайдер изменен на: {provider.get_display_name()}")
    
    def is_available(self) -> bool:
        """Проверка доступности текущего провайдера"""
        return self._provider.is_available()
    
    def validate_configuration(self) -> bool:
        """
        Валидация текущей конфигурации
        
        Returns:
            True если конфигурация валидна
        """
        try:
            self._provider.validate_config()
            config = self.get_config()
            
            # Дополнительные проверки
            if not config.base_url or not config.model:
                return False
            
            if config.temperature < 0 or config.temperature > 2:
                logger.warning(f"Подозрительное значение temperature: {config.temperature}")
            
            return True
        except Exception as e:
            logger.error(f"Ошибка валидации конфигурации: {e}")
            return False
    
    def get_status_info(self) -> Dict[str, Any]:
        """Получение информации о статусе конфигурации"""
        config = self.get_config()
        
        return {
            "provider": self._provider.get_display_name(),
            "provider_type": self._provider.config.provider_name,
            "model": config.model,
            "base_url": config.base_url,
            "is_available": self.is_available(),
            "is_valid": self.validate_configuration(),
            "temperature": config.temperature,
            "max_tokens": config.max_tokens
        }
    
    @classmethod
    def create_with_provider_type(cls, provider_type: str) -> 'LLMConfigManager':
        """
        Создание менеджера с указанным типом провайдера
        
        Args:
            provider_type: Тип провайдера (lm_studio, gigachat, etc.)
            
        Returns:
            Настроенный менеджер конфигурации
        """
        if provider_type not in AVAILABLE_PROVIDERS:
            raise ValueError(f"Неизвестный тип провайдера: {provider_type}")
        
        provider_class = AVAILABLE_PROVIDERS[provider_type]
        provider = provider_class.create_from_env()
        
        return cls(provider)


# Глобальный экземпляр менеджера конфигурации
_global_config_manager: Optional[LLMConfigManager] = None


def get_global_llm_config() -> LLMConfigManager:
    """
    Получение глобального экземпляра менеджера конфигурации LLM
    
    Returns:
        Глобальный менеджер конфигурации
    """
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = LLMConfigManager()
    
    return _global_config_manager


def set_global_llm_config(config_manager: LLMConfigManager) -> None:
    """
    Установка глобального экземпляра менеджера конфигурации
    
    Args:
        config_manager: Новый менеджер конфигурации
    """
    global _global_config_manager
    _global_config_manager = config_manager


def reset_global_llm_config() -> None:
    """Сброс глобального экземпляра менеджера конфигурации"""
    global _global_config_manager
    _global_config_manager = None