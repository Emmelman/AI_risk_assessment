"""
Центральный менеджер конфигурации LLM.
Единая точка управления всеми настройками языковых моделей.
ОБНОВЛЕНО: Добавлена поддержка GigaChat
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


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
    
    # НОВЫЕ поля для GigaChat
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    top_p: float = 0.2
    verify_ssl_certs: bool = False
    profanity_check: bool = False
    streaming: bool = True


class LLMConfigManager:
    """
    Singleton менеджер конфигурации LLM.
    Управляет всеми настройками языковых моделей из одного места.
    ОБНОВЛЕНО: Поддержка множественных провайдеров
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
        
        # Определяем провайдера
        provider_str = os.getenv("LLM_PROVIDER", "lm_studio").lower()
        provider = self._parse_provider(provider_str)
        
        # Загружаем конфигурацию в зависимости от провайдера
        if provider == LLMProvider.GIGACHAT:
            self._config = self._load_gigachat_config()
        else:
            self._config = self._load_lm_studio_config()
    
    def _parse_provider(self, provider_str: str) -> LLMProvider:
        """Парсинг провайдера из строки"""
        provider_mapping = {
            "lm_studio": LLMProvider.LM_STUDIO,
            "gigachat": LLMProvider.GIGACHAT,
            "openai": LLMProvider.OPENAI
        }
        return provider_mapping.get(provider_str, LLMProvider.LM_STUDIO)
    
    def _load_gigachat_config(self) -> LLMConfig:
        """Загрузка конфигурации для GigaChat"""
        
        # Базовые настройки
        base_url = os.getenv("GIGACHAT_BASE_URL", "https://gigachat-ift.sberdevices.delta.sbrf.ru/v1")
        model = os.getenv("GIGACHAT_MODEL", "GigaChat-Max")
        
        # Пути к сертификатам
        cert_path = os.getenv("GIGACHAT_CERT_PATH", "lib/llm/client_cert.pem")
        key_path = os.getenv("GIGACHAT_KEY_PATH", "lib/llm/client_key.pem")
        
        # Проверяем абсолютные пути
        if not os.path.isabs(cert_path):
            cert_path = os.path.join(os.getcwd(), cert_path)
        if not os.path.isabs(key_path):
            key_path = os.path.join(os.getcwd(), key_path)
        
        # Проверяем существование сертификатов
        if not (os.path.exists(cert_path) and os.path.exists(key_path)):
            raise FileNotFoundError(
                f"GigaChat SSL-файлы не найдены!\n"
                f"Cert: {cert_path}\n"
                f"Key: {key_path}\n"
                f"Проверьте пути в .env файле"
            )
        
        # Общие настройки
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        timeout = int(os.getenv("LLM_TIMEOUT", "120"))
        max_retries = int(os.getenv("MAX_RETRY_COUNT", "3"))
        retry_delay = float(os.getenv("LLM_RETRY_DELAY", "1.0"))
        
        return LLMConfig(
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            provider=LLMProvider.GIGACHAT,
            # GigaChat специфичные настройки
            cert_file=cert_path,
            key_file=key_path,
            top_p=float(os.getenv("GIGACHAT_TOP_P", "0.2")),
            verify_ssl_certs=os.getenv("GIGACHAT_VERIFY_SSL", "false").lower() == "true",
            profanity_check=os.getenv("GIGACHAT_PROFANITY_CHECK", "false").lower() == "true",
            streaming=os.getenv("GIGACHAT_STREAMING", "true").lower() == "true"
        )
    
    def _load_lm_studio_config(self) -> LLMConfig:
        """Загрузка конфигурации для LM Studio (старая логика)"""
        
        base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234")
        model = os.getenv("LLM_MODEL", "qwen3-4b")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        timeout = int(os.getenv("LLM_TIMEOUT", "120"))
        max_retries = int(os.getenv("MAX_RETRY_COUNT", "3"))
        retry_delay = float(os.getenv("LLM_RETRY_DELAY", "1.0"))
        
        provider = self._detect_provider(base_url)
        
        return LLMConfig(
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
        elif "gigachat" in base_url.lower() or "sber" in base_url.lower():
            return LLMProvider.GIGACHAT
        elif "openai" in base_url.lower() or "api.openai.com" in base_url:
            return LLMProvider.OPENAI
        else:
            return LLMProvider.LM_STUDIO
    
    # БАЗОВЫЕ методы (без изменений)
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
    
    # НОВЫЕ методы для GigaChat
    def get_cert_file(self) -> Optional[str]:
        """Получение пути к файлу сертификата"""
        return self._config.cert_file
    
    def get_key_file(self) -> Optional[str]:
        """Получение пути к файлу ключа"""
        return self._config.key_file
    
    def get_top_p(self) -> float:
        """Получение параметра top_p для GigaChat"""
        return self._config.top_p
    
    def get_verify_ssl_certs(self) -> bool:
        """Получение настройки проверки SSL сертификатов"""
        return self._config.verify_ssl_certs
    
    def get_profanity_check(self) -> bool:
        """Получение настройки проверки ненормативной лексики"""
        return self._config.profanity_check
    
    def get_streaming(self) -> bool:
        """Получение настройки потокового режима"""
        return self._config.streaming
    
    def is_gigachat(self) -> bool:
        """Проверка, используется ли GigaChat"""
        return self._config.provider == LLMProvider.GIGACHAT
    
    def is_lm_studio(self) -> bool:
        """Проверка, используется ли LM Studio"""
        return self._config.provider == LLMProvider.LM_STUDIO
    
    # СУЩЕСТВУЮЩИЕ методы (без изменений)
    def create_agent_config_dict(self, **overrides) -> Dict[str, Any]:
        """Создание словаря конфигурации для агентов с возможностью переопределения"""
        config = {
            "llm_base_url": self.get_base_url(),
            "llm_model": self.get_model(),
            "temperature": self.get_temperature(),
            "max_retries": self.get_max_retries(),
            "timeout_seconds": self.get_timeout()
        }
        
        config.update(overrides)
        return config
    
    def create_llm_client_config_dict(self, **overrides) -> Dict[str, Any]:
        """Создание словаря конфигурации для LLM клиента с возможностью переопределения"""
        config = {
            "base_url": self.get_base_url(),
            "model": self.get_model(),
            "temperature": self.get_temperature()
        }
        
        config.update(overrides)
        return config
    
    def reload_config(self) -> None:
        """Перезагрузка конфигурации из переменных окружения"""
        self._config = None
        self._load_config()
    
    def set_provider_config(self, provider: LLMProvider, config_updates: Dict[str, Any]) -> None:
        """Обновление конфигурации для конкретного провайдера"""
        for key, value in config_updates.items():
            os.environ[key] = str(value)
        
        self.reload_config()
    
    def get_info(self) -> Dict[str, Any]:
        """Получение информации о текущей конфигурации для отладки"""
        info = {
            "provider": self._config.provider.value,
            "base_url": self._config.base_url,
            "model": self._config.model,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "timeout": self._config.timeout,
            "max_retries": self._config.max_retries,
            "quality_threshold": self.get_quality_threshold()
        }
        
        # Добавляем GigaChat специфичную информацию
        if self.is_gigachat():
            info.update({
                "cert_file": self._config.cert_file,
                "key_file": self._config.key_file,
                "top_p": self._config.top_p,
                "streaming": self._config.streaming,
                "verify_ssl_certs": self._config.verify_ssl_certs,
                "profanity_check": self._config.profanity_check
            })
        
        return info


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


def is_gigachat() -> bool:
    """Быстрая проверка, используется ли GigaChat"""
    return config_manager.is_gigachat()