from src.agents.base_agent import create_agent_config, create_default_config_from_env

# Тест 1: Конфигурация агента из центрального менеджера
config1 = create_agent_config("test_agent", "Тестовый агент")
print("✅ Agent config модель:", config1.llm_config.model)

# Тест 2: Конфигурация с переопределением
config2 = create_agent_config("test_agent", "Тестовый агент", temperature=0.8)
print("✅ Переопределенная температура:", config2.llm_config.temperature)

# Тест 3: Дефолтная конфигурация из env
config3 = create_default_config_from_env()
print("✅ Default config модель:", config3.llm_config.model)

print("🎯 Base Agent использует центральную конфигурацию!")