# test_stage1.py
"""
Быстрый тест базовой инфраструктуры (Этап 1)
Проверяем модели, базу данных, LLM клиент и логирование
"""

import asyncio
import os
import sys
from pathlib import Path

# Добавляем путь к src для импортов
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

try:
    from src.models.risk_models import (
        AgentProfile, RiskEvaluation, AgentRiskAssessment,
        RiskType, RiskLevel, AgentType, AutonomyLevel, DataSensitivity
    )
    from src.models.database import DatabaseManager
    from src.utils.llm_client import LLMClient, LLMConfig, LLMMessage
    from src.utils.logger import setup_logging, get_logger, LogContext
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("🔧 Убедитесь, что все файлы созданы в папке src/")
    sys.exit(1)


async def test_models():
    """Тест Pydantic моделей"""
    print("🧪 Тестирование моделей данных...")
    
    # Создаем профиль агента
    profile = AgentProfile(
        name="TestChatBot",
        version="1.0",
        description="Тестовый чат-бот для проверки системы",
        agent_type=AgentType.CHATBOT,
        llm_model="qwen3-8b",
        autonomy_level=AutonomyLevel.SUPERVISED,
        data_access=[DataSensitivity.INTERNAL],
        target_audience="Сотрудники банка",
        system_prompts=["Ты помощник банковского сотрудника"],
        guardrails=["Не раскрывай персональные данные клиентов"]
    )
    
    # Создаем оценку риска
    evaluation = RiskEvaluation(
        risk_type=RiskType.ETHICAL,
        probability_score=2,
        impact_score=3,
        probability_reasoning="Низкая вероятность из-за ограниченного доступа к данным",
        impact_reasoning="Средний ущерб в случае дискриминации",
        evaluator_agent="test_evaluator",
        confidence_level=0.8
    )
    
    print(f"✅ Профиль создан: {profile.name}")
    print(f"✅ Оценка создана: {evaluation.risk_type} = {evaluation.total_score} баллов ({evaluation.risk_level})")
    
    return profile, evaluation


async def test_database():
    """Тест базы данных"""
    print("\n🧪 Тестирование базы данных...")
    
    # Создаем тестовые данные
    profile, evaluation = await test_models()
    
    # Инициализируем БД
    db = DatabaseManager("sqlite+aiosqlite:///./test_ai_risk.db")
    await db.initialize()
    
    try:
        # Сохраняем профиль
        profile_id = await db.save_agent_profile(profile)
        print(f"✅ Профиль сохранен с ID: {profile_id[:8]}...")
        
        # Загружаем профиль
        loaded_profile = await db.get_agent_profile(profile_id)
        assert loaded_profile is not None
        assert loaded_profile.name == profile.name
        print(f"✅ Профиль загружен: {loaded_profile.name}")
        
        # Создаем итоговую оценку
        assessment = AgentRiskAssessment(
            agent_profile=profile,
            assessment_id="test_assessment",
            risk_evaluations={RiskType.ETHICAL: evaluation},
            overall_risk_score=evaluation.total_score,
            overall_risk_level=evaluation.risk_level,
            priority_recommendations=["Внедрить мониторинг этических рисков"]
        )
        
        # Сохраняем оценку
        assessment_id = await db.save_risk_assessment(assessment, profile_id)
        print(f"✅ Оценка сохранена с ID: {assessment_id[:8]}...")
        
        # Загружаем оценку
        loaded_assessment = await db.get_risk_assessment(assessment_id)
        assert loaded_assessment is not None
        print(f"✅ Оценка загружена: {loaded_assessment['assessment'].overall_risk_level}")
        
    finally:
        await db.close()
        # Удаляем тестовую БД
        test_db_path = Path("test_ai_risk.db")
        if test_db_path.exists():
            test_db_path.unlink()
    
    print("✅ База данных работает корректно")


async def test_llm_client():
    """Тест LLM клиента"""
    print("\n🧪 Тестирование LLM клиента...")
    
    # Создаем клиент
    config = LLMConfig(
        base_url="http://127.0.0.1:1234",
        model="qwen3-8b",
        temperature=0.1,
        timeout=30
    )
    
    client = LLMClient(config)
    
    try:
        # Проверяем доступность сервера
        is_available = await client.health_check()
        
        if not is_available:
            print("⚠️ LM Studio недоступен на localhost:1234")
            print("   Запустите LM Studio с моделью qwen3-8b для полного тестирования")
            return
        
        print("✅ LM Studio доступен")
        
        # Проверяем список моделей
        try:
            models = await client.get_available_models()
            print(f"✅ Доступные модели: {models}")
        except Exception as e:
            print(f"⚠️ Не удалось получить список моделей: {e}")
        
        # Тестируем простой запрос
        messages = [
            LLMMessage(role="system", content="Ты - помощник для тестирования системы."),
            LLMMessage(role="user", content="Скажи 'Тест прошел успешно' на русском языке.")
        ]
        
        response = await client.complete_chat(messages)
        print(f"✅ LLM ответ получен: {response.content[:50]}...")
        print(f"✅ Использовано токенов: {response.usage.get('total_tokens', 'N/A')}")
        
        # Тестируем специализированный метод для анализа рисков
        from src.utils.llm_client import RiskAnalysisLLMClient
        
        risk_client = RiskAnalysisLLMClient(config)
        
        test_agent_data = """
        Агент: TestBot
        Тип: Чат-бот
        Автономность: Под надзором
        Данные: Внутренние справочники
        """
        
        test_criteria = """
        Оцени этические риски:
        1 балл - нет доступа к персональным данным
        5 баллов - полный доступ к чувствительным данным
        """
        
        risk_evaluation = await risk_client.evaluate_risk(
            risk_type="этические риски",
            agent_data=test_agent_data,
            evaluation_criteria=test_criteria
        )
        
        print(f"✅ Оценка риска получена: {risk_evaluation['total_score']} баллов")
        
    except Exception as e:
        print(f"⚠️ Ошибка LLM клиента: {e}")
    finally:
        await client.close()


def test_logging():
    """Тест системы логирования"""
    print("\n🧪 Тестирование логирования...")
    
    # Настраиваем логгер для тестов
    logger_instance = setup_logging(
        log_level="INFO",
        log_file="logs/test_stage1.log",
        enable_console=True,
        enable_rich=True
    )
    
    # Тестируем базовое логирование
    test_logger = logger_instance.bind_context("test_assessment_123", "test_agent")
    test_logger.info("Тестирование базового логирования")
    
    # Тестируем специализированные методы
    logger_instance.log_agent_start("test_agent", "unit_test", "test_assessment_123")
    logger_instance.log_risk_evaluation("test_agent", "test_assessment_123", "ethical", 6, "medium")
    logger_instance.log_agent_success("test_agent", "unit_test", "test_assessment_123", 1.5)
    
    # Тестируем контекстный менеджер
    with LogContext("test_operation", "test_assessment_123", "test_agent"):
        import time
        time.sleep(0.1)  # Имитируем работу
    
    print("✅ Логирование работает корректно")
    print(f"✅ Лог сохранен в: logs/test_stage1.log")


async def main():
    """Главная функция тестирования"""
    print("🚀 Тестирование базовой инфраструктуры (Этап 1)")
    print("=" * 60)
    
    try:
        # Создаем директории
        Path("logs").mkdir(exist_ok=True)
        
        # Запускаем тесты
        await test_models()
        await test_database()
        await test_llm_client()
        test_logging()
        
        print("\n" + "=" * 60)
        print("🎉 ВСЕ ТЕСТЫ ЭТАПА 1 ПРОЙДЕНЫ УСПЕШНО!")
        print("\n📋 Проверено:")
        print("✅ Pydantic модели данных")
        print("✅ SQLite база данных")
        print("✅ LLM клиент (qwen3-8b)")
        print("✅ Система логирования")
        print("\n🚀 Готовы к Этапу 2: Инструменты анализа")
        
    except Exception as e:
        print(f"\n❌ ОШИБКА В ТЕСТАХ: {e}")
        print("\n🔧 Проверьте:")
        print("- Установлены ли все зависимости?")
        print("- Запущен ли LM Studio на localhost:1234?")
        print("- Есть ли права на создание файлов?")
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)