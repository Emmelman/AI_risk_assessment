"""
Модуль конфигурации системы оценки рисков ИИ-агентов

Этот модуль предоставляет централизованное управление конфигурацией,
включая настройки LLM провайдеров, базы данных и других компонентов системы.
"""

from .llm_config_manager import LLMConfigManager, get_global_llm_config, set_global_llm_config

__all__ = [
    "LLMConfigManager",
    "get_global_llm_config",
    "set_global_llm_config"
]