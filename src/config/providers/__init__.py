"""
Провайдеры LLM для системы оценки рисков

Содержит реализации различных провайдеров LLM сервисов
"""

from .base_provider import BaseLLMProvider, ProviderConfig
from .lm_studio import LMStudioProvider
from .gigachat import GigaChatProvider

# Реестр доступных провайдеров
AVAILABLE_PROVIDERS = {
    "lm_studio": LMStudioProvider,
    "gigachat": GigaChatProvider
}

__all__ = [
    "BaseLLMProvider",
    "ProviderConfig", 
    "LMStudioProvider",
    "GigaChatProvider",
    "AVAILABLE_PROVIDERS"
]