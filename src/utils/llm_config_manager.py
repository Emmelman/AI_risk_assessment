"""
Центральный менеджер конфигурации LLM.
Единая точка управления всеми настройками языковых моделей.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    """Поддерживаемые провайдеры LLM"""
    LM_STUDIO = "lm_studio"
    GIGACHAT = "gigachat"
    OPENAI = "openai"


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
    provider: LLMProvider = LLMProvider.LM_STUDIO


class LLMConfigManager:
    """
    Singleton менеджер конфигурации LLM.
    Управляет всеми настройками языковых моделей из одного места.
    """
    
    _instance: Optional['LLMConfigManager'] = None
    _config: Optional[LLMConfig] = None
    
    def __new__(cls) -> 'LLMConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._load_config()
    
    def _load_config(self) -> None:
        """Загрузка конфигурации из переменных окружения"""
        
        # Базовые настройки LLM
        base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234")
        model = os.getenv("LLM_MODEL", "qwen3-4b")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        
        # Настройки подключения
        timeout = int(os.getenv("LLM_TIMEOUT", "120"))
        max_retries = int(os.getenv("MAX_RETRY_COUNT", "3"))
        retry_delay = float(os.getenv("LLM_RETRY_DELAY", "1.0"))
        
        # Определение провайдера по URL
        provider = self._detect_provider(base_url)
        
        self._config = LLMConfig(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            provider=provider
        )
    
    def _detect_provider(self, base_url: str) -> LLMProvider:
        """Автоматическое определение провайдера по URL"""
        if "127.0.0.1" in base_url or "localhost" in base_url:
            return LLMProvider.LM_STUDIO
        elif "gigachat" in base_url.lower():
            return LLMProvider.GIGACHAT
        elif "openai" in base_url.lower() or "api.openai.com" in base_url:
            return LLMProvider.OPENAI
        else:
            return LLMProvider.LM_STUDIO  # По умолчанию
    
    def get_config(self) -> LLMConfig:
        """Получение полной конфигурации"""
        return self._config
    
    def get_base_url(self) -> str:
        """Получение базового URL для LLM API"""
        return self._config.base_url
    
    def get_model(self) -> str:
        """Получение названия модели"""
        return self._config.model
    
    def get_temperature(self) -> float:
        """Получение температуры для генерации"""
        return self._config.temperature
    
    def get_max_tokens(self) -> int:
        """Получение максимального количества токенов"""
        return self._config.max_tokens
    
    def get_timeout(self) -> int:
        """Получение таймаута запроса в секундах"""
        return self._config.timeout
    
    def get_max_retries(self) -> int:
        """Получение максимального количества повторных попыток"""
        return self._config.max_retries
    
    def get_retry_delay(self) -> float:
        """Получение задержки между повторными попытками"""
        return self._config.retry_delay
    
    def get_provider(self) -> LLMProvider:
        """Получение текущего провайдера"""
        return self._config.provider
    
    def get_quality_threshold(self) -> float:
        """Получение порога качества для оценок"""
        return float(os.getenv("QUALITY_THRESHOLD", "7.0"))
    
    def create_agent_config_dict(self, **overrides) -> Dict[str, Any]:
        """
        Создание словаря конфигурации для агентов с возможностью переопределения.
        
        Args:
            **overrides: Параметры для переопределения базовых настроек
        
        Returns:
            Dict с конфигурацией для создания агентов
        """
        config = {
            "llm_base_url": self.get_base_url(),
            "llm_model": self.get_model(),
            "temperature": self.get_temperature(),
            "max_retries": self.get_max_retries(),
            "timeout_seconds": self.get_timeout()
        }
        
        # Применяем переопределения
        config.update(overrides)
        return config
    
    def create_llm_client_config_dict(self, **overrides) -> Dict[str, Any]:
        """
        Создание словаря конфигурации для LLM клиента с возможностью переопределения.
        
        Args:
            **overrides: Параметры для переопределения базовых настроек
        
        Returns:
            Dict с конфигурацией для создания LLM клиента
        """
        config = {
            "base_url": self.get_base_url(),
            "model": self.get_model(),
            "temperature": self.get_temperature()
        }
        
        # Применяем переопределения
        config.update(overrides)
        return config
    
    def reload_config(self) -> None:
        """Перезагрузка конфигурации из переменных окружения"""
        self._config = None
        self._load_config()
    
    def set_provider_config(self, provider: LLMProvider, config_updates: Dict[str, Any]) -> None:
        """
        Обновление конфигурации для конкретного провайдера.
        Полезно для переключения между LM Studio и GigaChat Max.
        
        Args:
            provider: Тип провайдера
            config_updates: Словарь с обновлениями конфигурации
        """
        # Обновляем переменные окружения
        for key, value in config_updates.items():
            os.environ[key] = str(value)
        
        # Перезагружаем конфигурацию
        self.reload_config()
    
    def get_info(self) -> Dict[str, Any]:
        """Получение информации о текущей конфигурации для отладки"""
        return {
            "provider": self._config.provider.value,
            "base_url": self._config.base_url,
            "model": self._config.model,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "timeout": self._config.timeout,
            "max_retries": self._config.max_retries,
            "quality_threshold": self.get_quality_threshold()
        }


# Глобальный экземпляр менеджера конфигурации
config_manager = LLMConfigManager()


def get_llm_config_manager() -> LLMConfigManager:
    """Получение глобального экземпляра менеджера конфигурации"""
    return config_manager


# Convenience функции для быстрого доступа к параметрам
def get_base_url() -> str:
    """Быстрый доступ к базовому URL"""
    return config_manager.get_base_url()


def get_model() -> str:
    """Быстрый доступ к модели"""
    return config_manager.get_model()


def get_temperature() -> float:
    """Быстрый доступ к температуре"""
    return config_manager.get_temperature()


def get_llm_config() -> LLMConfig:
    """Быстрый доступ к полной конфигурации"""
    return config_manager.get_config()