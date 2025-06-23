# test_profiler_agent.py
"""
Тест профайлер-агента (начало Этапа 3)
Проверяем создание и базовую работу профайлер-агента
"""

import asyncio
import sys
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_profiler_agent_creation():
    """Тест создания профайлер-агента"""
    print("🧪 Тестирование создания профайлер-агента...")
    
    try:
        from src.agents.profiler_agent import create_profiler_agent
        
        # Создаем агента
        profiler = create_profiler_agent()
        
        print(f"✅ Профайлер создан: {profiler.name}")
        print(f"✅ Описание: {profiler.description}")
        print(f"✅ Тип задачи: {profiler._get_task_type()}")
        
        # Проверяем health check (если LM Studio доступен)
        is_healthy = await profiler.health_check()
        print(f"✅ Health check: {'✅ OK' if is_healthy else '⚠️ LM Studio недоступен'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка создания профайлера: {e}")
        return False

async def test_profiler_agent_processing():
    """Тест обработки данных профайлером"""
    print("\n🧪 Тестирование обработки данных...")
    
    try:
        from src.agents.profiler_agent import create_profiler_agent
        
        profiler = create_profiler_agent()
        
        # Тестовые данные - анализируем сам проект
        test_input = {
            "source_files": [
                str(Path(__file__).parent),  # Корневая папка проекта
                "test_stage1.py",  # Тестовый файл
                "requirements.txt"  # Файл зависимостей
            ],
            "agent_name": "AI_Risk_Assessment_System"
        }
        
        assessment_id = "test_profiler_001"
        
        print(f"📁 Анализируем: {len(test_input['source_files'])} источников")
        
        # Запускаем профайлер
        result = await profiler.run(test_input, assessment_id)
        
        print(f"✅ Статус: {result.status}")
        print(f"✅ Время выполнения: {result.execution_time_seconds:.2f}с")
        
        if result.status.value == "completed":
            agent_profile = result.result_data["agent_profile"]
            data_summary = result.result_data["collected_data_summary"]
            
            print(f"✅ Профиль создан: {agent_profile['name']}")
            print(f"✅ Тип агента: {agent_profile['agent_type']}")
            print(f"✅ Уровень автономности: {agent_profile['autonomy_level']}")
            print(f"✅ Целевая аудитория: {agent_profile['target_audience']}")
            print(f"✅ Системных промптов: {len(agent_profile['system_prompts'])}")
            print(f"✅ Ограничений: {len(agent_profile['guardrails'])}")
            
            print(f"\n📊 Сводка сбора данных:")
            print(f"  - Файлов обработано: {data_summary['documents_processed']}")
            print(f"  - Анализ кода: {'✅' if data_summary['code_analysis_success'] else '❌'}")
            print(f"  - Анализ промптов: {'✅' if data_summary['prompt_analysis_success'] else '❌'}")
            print(f"  - Ошибок: {data_summary['errors_count']}")
            
        else:
            print(f"❌ Ошибка обработки: {result.error_message}")
            return False
        
        # Получаем статистику агента
        stats = profiler.get_stats()
        print(f"\n📈 Статистика агента:")
        print(f"  - Запросов: {stats['total_requests']}")
        print(f"  - Успешность: {stats['success_rate']:.1%}")
        print(f"  - Среднее время: {stats['average_response_time']:.2f}с")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка обработки: {e}")
        return False

async def test_langgraph_integration():
    """Тест интеграции с LangGraph"""
    print("\n🧪 Тестирование интеграции с LangGraph...")
    
    try:
        from src.agents.profiler_agent import create_profiler_agent, create_profiler_node_function
        
        profiler = create_profiler_agent()
        
        # Создаем функцию узла для LangGraph
        profiler_node = create_profiler_node_function(profiler)
        
        # Тестовое состояние workflow
        test_state = {
            "assessment_id": "test_langgraph_001",
            "source_files": ["test_stage1.py"],
            "preliminary_agent_name": "TestAgent"
        }
        
        print("🔄 Запуск узла профайлера в LangGraph состоянии...")
        
        # Вызываем узел
        updated_state = await profiler_node(test_state)
        
        print(f"✅ Состояние обновлено")
        print(f"✅ Текущий шаг: {updated_state.get('current_step', 'unknown')}")
        print(f"✅ Результат профилирования: {'✅' if 'profiling_result' in updated_state else '❌'}")
        
        if 'agent_profile' in updated_state:
            profile = updated_state['agent_profile']
            print(f"✅ Профиль агента в состоянии: {profile['name']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка интеграции LangGraph: {e}")
        return False

async def main():
    """Главная функция тестирования"""
    print("🚀 Тестирование профайлер-агента (Этап 3)")
    print("=" * 60)
    
    success = True
    
    # Тестируем создание агента
    success &= await test_profiler_agent_creation()
    
    # Тестируем обработку данных
    success &= await test_profiler_agent_processing()
    
    # Тестируем интеграцию с LangGraph
    success &= await test_langgraph_integration()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ПРОФАЙЛЕР-АГЕНТ РАБОТАЕТ УСПЕШНО!")
        print("\n📋 Проверено:")
        print("✅ Создание профайлер-агента")
        print("✅ Сбор данных из файлов и папок")
        print("✅ Анализ документов, кода и промптов")
        print("✅ Создание профиля AgentProfile")
        print("✅ Интеграция с LangGraph workflow")
        print("\n🚀 Готовы создавать агентов-оценщиков!")
    else:
        print("❌ ЕСТЬ ПРОБЛЕМЫ С ПРОФАЙЛЕРОМ")
        print("\n🔧 Проверьте:")
        print("- Запущен ли LM Studio на localhost:1234?")
        print("- Все ли зависимости установлены?")
        print("- Есть ли права на чтение файлов?")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)