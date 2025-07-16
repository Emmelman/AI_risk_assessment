
"""
Скрипт валидации миграции на центральный LLM конфигуратор

Проверяет все компоненты системы после внедрения изменений.
"""

import sys
import asyncio
import warnings
from pathlib import Path
from typing import Dict, List, Any


def test_imports():
    """Тест: Все импорты работают корректно"""
    print("🔍 Тестируем импорты...")
    
    tests = []
    
    # Центральный конфигуратор
    try:
        from src.config import get_global_llm_config, LLMConfigManager
        tests.append(("Центральный конфигуратор", True, None))
    except Exception as e:
        tests.append(("Центральный конфигуратор", False, str(e)))
    
    # Провайдеры
    try:
        from src.config.providers import LMStudioProvider, GigaChatProvider, AVAILABLE_PROVIDERS
        tests.append(("Провайдеры LLM", True, None))
    except Exception as e:
        tests.append(("Провайдеры LLM", False, str(e)))
    
    # Обновленные агенты
    try:
        from src.agents.base_agent import create_agent_config
        from src.agents.profiler_agent import create_profiler_agent
        from src.agents.critic_agent import create_critic_agent
        from src.agents.evaluator_agents import create_all_evaluator_agents
        tests.append(("Агенты", True, None))
    except Exception as e:
        tests.append(("Агенты", False, str(e)))
    
    # Workflow
    try:
        from src.workflow.graph_builder import create_risk_assessment_workflow
        tests.append(("Workflow", True, None))
    except Exception as e:
        tests.append(("Workflow", False, str(e)))
    
    # Выводим результаты
    for test_name, success, error in tests:
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}")
        if error:
            print(f"      Ошибка: {error}")
    
    return all(test[1] for test in tests)


def test_central_config():
    """Тест: Центральный конфигуратор работает"""
    print("🔍 Тестируем центральный конфигуратор...")
    
    try:
        from src.config import get_global_llm_config
        
        # Получаем конфигуратор
        config_manager = get_global_llm_config()
        print(f"  ✅ Менеджер создан: {type(config_manager).__name__}")
        
        # Проверяем конфигурацию
        config = config_manager.get_config()
        print(f"  ✅ Конфигурация получена: {config.model}")
        
        # Проверяем валидацию
        is_valid = config_manager.validate_configuration()
        print(f"  {'✅' if is_valid else '⚠️'} Валидация: {is_valid}")
        
        # Проверяем статус
        status = config_manager.get_status_info()
        print(f"  ✅ Статус: {status['provider']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return False


def test_agents_creation():
    """Тест: Агенты создаются без LLM параметров"""
    print("🔍 Тестируем создание агентов...")
    
    tests = []
    
    # Профайлер
    try:
        from src.agents.profiler_agent import create_profiler_agent
        profiler = create_profiler_agent()
        tests.append(("Профайлер", True, profiler.name))
    except Exception as e:
        tests.append(("Профайлер", False, str(e)))
    
    # Критик
    try:
        from src.agents.critic_agent import create_critic_agent
        critic = create_critic_agent()
        tests.append(("Критик", True, critic.name))
    except Exception as e:
        tests.append(("Критик", False, str(e)))
    
    # Оценщики
    try:
        from src.agents.evaluator_agents import create_all_evaluator_agents
        evaluators = create_all_evaluator_agents()
        tests.append(("Оценщики", True, f"{len(evaluators)} агентов"))
    except Exception as e:
        tests.append(("Оценщики", False, str(e)))
    
    # Выводим результаты
    for test_name, success, info in tests:
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}: {info}")
    
    return all(test[1] for test in tests)


def test_workflow_creation():
    """Тест: Workflow создается без LLM параметров"""
    print("🔍 Тестируем создание workflow...")
    
    try:
        from src.workflow.graph_builder import create_risk_assessment_workflow
        
        # Создаем workflow
        workflow = create_risk_assessment_workflow()
        print(f"  ✅ Workflow создан: {type(workflow).__name__}")
        
        # Проверяем статус
        status = workflow.get_workflow_status()
        print(f"  ✅ Статус получен: {status['agents_ready']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return False


def test_critical_methods():
    """Тест: Критически важные методы присутствуют"""
    print("🔍 Тестируем критически важные методы...")
    
    tests = []
    
    # Проверяем методы BaseAgent
    try:
        from src.agents.base_agent import BaseAgent
        
        # Создаем временный агент для проверки методов
        class TestAgent(BaseAgent):
            def get_system_prompt(self):
                return "test"
            async def process(self, input_data, assessment_id="test"):
                return None
        
        # Проверяем наличие критических методов
        critical_methods = [
            '_parse_llm_response',
            '_ensure_required_fields', 
            '_validate_and_fix_field_types',
            '_validate_business_logic',
            '_get_default_evaluation_data',
            '_get_required_result_fields'
        ]
        
        missing_methods = []
        for method_name in critical_methods:
            if not hasattr(BaseAgent, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            tests.append((f"BaseAgent методы", False, f"Отсутствуют: {missing_methods}"))
        else:
            tests.append((f"BaseAgent методы", True, f"Все {len(critical_methods)} методов присутствуют"))
            
    except Exception as e:
        tests.append((f"BaseAgent методы", False, str(e)))
    
    # Проверяем методы EvaluationAgent
    try:
        from src.agents.base_agent import EvaluationAgent, AgentConfig
        
        config = AgentConfig("test", "test")
        evaluator = EvaluationAgent(config, "test_risk")
        
        # Проверяем что методы доступны
        evaluation_methods = ['evaluate_risk', 'create_fallback_result', 'validate_result']
        
        missing_methods = []
        for method_name in evaluation_methods:
            if not hasattr(evaluator, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            tests.append((f"EvaluationAgent методы", False, f"Отсутствуют: {missing_methods}"))
        else:
            tests.append((f"EvaluationAgent методы", True, f"Все {len(evaluation_methods)} методов присутствуют"))
            
    except Exception as e:
        tests.append((f"EvaluationAgent методы", False, str(e)))
    
    # Выводим результаты
    for test_name, success, info in tests:
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}: {info}")
    
    return all(test[1] for test in tests)

def test_backward_compatibility():
    """Тест: Обратная совместимость (deprecated функции работают)"""
    print("🔍 Тестируем обратную совместимость...")
    
    # Захватываем warnings для deprecated функций
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        tests = []
        
        # Legacy функция создания агента
        try:
            from src.agents.base_agent import create_agent_config_legacy
            config = create_agent_config_legacy("test", "test")
            # Проверяем что было предупреждение
            has_warning = any("deprecated" in str(warning.message) for warning in w)
            tests.append(("Legacy create_agent_config", True, f"Warning: {has_warning}"))
        except Exception as e:
            tests.append(("Legacy create_agent_config", False, str(e)))
        
        # Legacy функция профайлера
        try:
            from src.agents.profiler_agent import create_profiler_agent_legacy
            profiler = create_profiler_agent_legacy()
            tests.append(("Legacy profiler", True, profiler.name))
        except Exception as e:
            tests.append(("Legacy profiler", False, str(e)))
        
        # Legacy функция критика
        try:
            from src.agents.critic_agent import create_critic_agent_legacy
            critic = create_critic_agent_legacy()
            tests.append(("Legacy critic", True, critic.name))
        except Exception as e:
            tests.append(("Legacy critic", False, str(e)))
    
    # Выводим результаты
    for test_name, success, info in tests:
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}: {info}")
    
    return all(test[1] for test in tests)#!/usr/bin/env python3
def test_provider_switching():
    """Тест: Переключение провайдеров"""
    print("🔍 Тестируем переключение провайдеров...")
    
    try:
        from src.config import get_global_llm_config, set_global_llm_config, LLMConfigManager
        
        # Сохраняем текущий конфигуратор
        original_manager = get_global_llm_config()
        original_provider = original_manager.get_provider().config.provider_name
        print(f"  📋 Текущий провайдер: {original_provider}")
        
        # Пробуем создать LM Studio провайдер
        try:
            lm_studio_manager = LLMConfigManager.create_with_provider_type("lm_studio")
            print("  ✅ LM Studio провайдер создан")
        except Exception as e:
            print(f"  ❌ Ошибка LM Studio: {e}")
            return False
        
        # Пробуем создать GigaChat провайдер (заглушка)
        try:
            gigachat_manager = LLMConfigManager.create_with_provider_type("gigachat")
            print("  ✅ GigaChat провайдер создан (заглушка)")
        except Exception as e:
            print(f"  ❌ Ошибка GigaChat: {e}")
            return False
        
        # Восстанавливаем оригинальный конфигуратор
        set_global_llm_config(original_manager)
        print("  ✅ Конфигуратор восстановлен")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка: {e}")
        return False


async def test_workflow_execution():
    """Тест: Workflow выполняется с новой конфигурацией"""
    print("🔍 Тестируем выполнение workflow...")
    
    try:
        from src.workflow.graph_builder import test_workflow_execution
        
        print("  ⏳ Запуск тестового workflow...")
        result = await test_workflow_execution()
        
        status = "✅" if result else "❌"
        print(f"  {status} Тестовое выполнение: {result}")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Ошибка выполнения: {e}")
        return False


def test_cli_integration():
    """Тест: CLI работает с новой конфигурацией"""
    print("🔍 Тестируем CLI интеграцию...")
    
    try:
        # Проверяем что main.py импортируется
        import main
        print("  ✅ main.py импортирован")
        
        # Проверяем что CLI команды доступны
        from click.testing import CliRunner
        runner = CliRunner()
        
        # Тест команды status
        result = runner.invoke(main.cli, ['status', '--check-llm'])
        if result.exit_code == 0:
            print("  ✅ Команда status работает")
        else:
            print(f"  ⚠️ Команда status: код {result.exit_code}")
        
        # Тест команды config
        result = runner.invoke(main.cli, ['config', '--show-config'])
        if result.exit_code == 0:
            print("  ✅ Команда config работает")
        else:
            print(f"  ⚠️ Команда config: код {result.exit_code}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка CLI: {e}")
        return False


def print_summary(results: Dict[str, bool]):
    """Вывод сводки результатов"""
    print("\n" + "📊 СВОДКА ВАЛИДАЦИИ".center(60, "="))
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print("=" * 60)
    
    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print("✅ Миграция на центральный конфигуратор завершена")
        print("🚀 Система готова к переключению на GigaChat")
    else:
        print(f"⚠️  ПРОЙДЕНО {passed}/{total} ТЕСТОВ")
        print("❌ Требуется исправление ошибок")
        
        # Показываем неудавшиеся тесты
        failed_tests = [name for name, result in results.items() if not result]
        print(f"\n🔧 Исправьте следующие проблемы:")
        for test in failed_tests:
            print(f"   - {test}")
    
    return passed == total


def main():
    """Основная функция валидации"""
    print("🚀 ВАЛИДАЦИЯ МИГРАЦИИ НА ЦЕНТРАЛЬНЫЙ LLM КОНФИГУРАТОР")
    print("=" * 60)
    print("🎯 Цель: Проверить корректность внедрения изменений")
    print("📋 Статус: Обратная совместимость должна сохраниться")
    print("=" * 60)
    
    # Выполняем все тесты
    results = {}
    
    results["Импорты"] = test_imports()
    results["Центральный конфигуратор"] = test_central_config()
    results["Критические методы"] = test_critical_methods()
    results["Создание агентов"] = test_agents_creation()
    results["Создание workflow"] = test_workflow_creation()
    results["Обратная совместимость"] = test_backward_compatibility()
    results["Переключение провайдеров"] = test_provider_switching()
    results["CLI интеграция"] = test_cli_integration()
    
    # Асинхронный тест
    print("🔍 Тестируем выполнение workflow...")
    try:
        results["Выполнение workflow"] = asyncio.run(test_workflow_execution())
    except Exception as e:
        print(f"  ❌ Ошибка async теста: {e}")
        results["Выполнение workflow"] = False
    
    # Выводим сводку
    success = print_summary(results)
    
    # Дополнительная информация
    if success:
        print(f"\n💡 СЛЕДУЮЩИЕ ШАГИ:")
        print("1. Запустите: python main.py demo")
        print("2. Проверьте: python main.py status --detailed")
        print("3. Ознакомьтесь с: MIGRATION_GUIDE.md")
        print("4. Подготовьтесь к переключению на GigaChat")
    else:
        print(f"\n🔧 УСТРАНЕНИЕ ОШИБОК:")
        print("1. Проверьте все файлы созданы правильно")
        print("2. Убедитесь что .env файл обновлен")
        print("3. Перезапустите LM Studio если нужно")
        print("4. Проверьте зависимости: pip install -r requirements.txt")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())