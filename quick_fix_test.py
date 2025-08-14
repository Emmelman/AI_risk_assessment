# quick_fix_test.py
"""
Быстрый тест исправлений для Lawdigest_bot
Проверяет только критические исправления
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

# JSON Encoder для datetime объектов
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

async def quick_test():
    """Быстрый тест только на реальных данных с диагностикой"""
    
    print("🔧 БЫСТРЫЙ ТЕСТ ИСПРАВЛЕНИЙ V2")
    print("=" * 45)
    
    # Подключаемся к системе
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from src.agents.profiler_agent import create_profiler_agent
        
        # Создаем профайлер
        profiler = create_profiler_agent(
            llm_base_url="http://127.0.0.1:1234",
            llm_model="openai/gpt-oss-20b",
            temperature=0.1
        )
        
        print("✅ Профайлер создан")
        
        # Тестовые данные
        real_data_path = Path(r"C:\Users\Nikita\Documents\Python Projects\AI_Risk_Assessment\Lawdigest_bot")
        
        if not real_data_path.exists():
            print(f"❌ Папка не найдена: {real_data_path}")
            return
        
        # Показываем что есть в папке
        files_found = []
        for pattern in ['*.py', '*.md', '*.txt', '*.json', '*.yaml', '*.yml', '*.env', '*.cfg', '*.ini', '*.docx', '*.xlsx']:
            files_found.extend(list(real_data_path.rglob(pattern)))
        
        print(f"📁 Найдено файлов для анализа: {len(files_found)}")
        
        # Входные данные
        input_data = {
            "source_files": [str(real_data_path)],
            "agent_name": "Lawdigest_bot"
        }
        
        print(f"🚀 Запуск с диагностикой...")
        
        # Запускаем
        result = await profiler.process(input_data, "quick_test_v2")
        
        if result.status.value == "completed":
            print("✅ Профилирование успешно")
            
            agent_profile_data = result.result_data["agent_profile"]
            
            # Проверяем ключевые поля
            print(f"\n📋 ОСНОВНЫЕ ПОЛЯ:")
            print(f"   Название: {agent_profile_data.get('name')}")
            print(f"   Тип: {agent_profile_data.get('agent_type')}")
            print(f"   Автономность: {agent_profile_data.get('autonomy_level')}")
            print(f"   LLM модель: {agent_profile_data.get('llm_model')}")
            
            description = agent_profile_data.get('description', '')
            print(f"   Описание: {description[:100]}{'...' if len(description) > 100 else ''}")
            
            # Проверяем массивы
            print(f"\n📊 МАССИВЫ:")
            system_prompts = agent_profile_data.get('system_prompts', [])
            guardrails = agent_profile_data.get('guardrails', [])
            print(f"   System prompts: {len(system_prompts)} шт")
            print(f"   Guardrails: {len(guardrails)} шт")
            
            # Проверяем detailed_summary
            detailed = agent_profile_data.get('detailed_summary')
            if detailed:
                print(f"\n📝 DETAILED_SUMMARY АНАЛИЗ:")
                
                all_sections_ok = True
                for section_name, content in detailed.items():
                    content_str = str(content)
                    
                    # Проверяем признаки JSON
                    is_json_like = (
                        content_str.startswith('{') or 
                        'probability_score' in content_str or
                        'risk_level' in content_str or
                        '"' in content_str[:50]
                    )
                    
                    # Проверяем что это нормальный текст
                    is_normal_text = (
                        not is_json_like and 
                        len(content_str) > 100 and
                        not content_str.startswith('{') and
                        '. ' in content_str  # Есть предложения
                    )
                    
                    if is_normal_text:
                        status = "✅ Нормальный текст"
                        print(f"   {section_name}: {status} ({len(content_str)} символов)")
                        # Показываем превью
                        preview = content_str[:200] + "..." if len(content_str) > 200 else content_str
                        print(f"      📝 Превью: {preview}")
                    else:
                        status = "❌ Все еще JSON/структура"
                        print(f"   {section_name}: {status} ({len(content_str)} символов)")
                        all_sections_ok = False
                        # Показываем начало для диагностики
                        preview = content_str[:100] + "..." if len(content_str) > 100 else content_str
                        print(f"      🔍 Начало: {preview}")
                
                print(f"\n🎯 ИТОГОВАЯ ОЦЕНКА DETAILED_SUMMARY:")
                if all_sections_ok:
                    print("   ✅ ВСЕ РАЗДЕЛЫ СОДЕРЖАТ НОРМАЛЬНЫЙ ТЕКСТ!")
                else:
                    print("   ❌ Некоторые разделы все еще содержат JSON структуры")
            else:
                print(f"\n❌ detailed_summary отсутствует!")
            
            # Сохраняем с исправленным энкодером
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"lawdigest_fixed_v2_{timestamp}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(agent_profile_data, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
            
            print(f"\n💾 Результат сохранен: {output_file}")
            
            # Показываем размер файла
            file_size = Path(output_file).stat().st_size
            print(f"📏 Размер файла: {file_size:,} байт")
            
        else:
            print(f"❌ Ошибка: {result.error_message}")
            
    except Exception as e:
        print(f"💥 Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(quick_test())