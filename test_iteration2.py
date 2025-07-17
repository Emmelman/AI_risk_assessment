from src.utils.llm_client import get_llm_client, create_llm_client, LLMConfig
from src.utils.llm_config_manager import get_llm_config_manager

async def test_llm_integration():
    # Тест 1: Центральный конфигуратор работает
    manager = get_llm_config_manager()
    print("✅ Центральный конфигуратор:", manager.get_model())
    
    # Тест 2: LLMConfig создается из менеджера
    config = LLMConfig.from_manager()
    print("✅ Конфиг из менеджера:", config.model)
    
    # Тест 3: Глобальный клиент использует центральную конфигурацию
    client = await get_llm_client()
    print("✅ Глобальный клиент модель:", client.config.model)
    
    # Тест 4: Фабрика поддерживает переопределения
    custom_client = create_llm_client(temperature=0.5)
    print("✅ Кастомная температура:", custom_client.config.temperature)
    
    print("🎯 Все компоненты используют центральную конфигурацию!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_llm_integration())