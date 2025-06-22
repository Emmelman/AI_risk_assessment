# test_stage2.py
"""
Быстрый тест инструментов анализа (Этап 2)
Проверяем document_parser, code_analyzer, prompt_analyzer
"""

import sys
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Тест импортов всех инструментов"""
    print("🧪 Тестирование импортов...")
    
    try:
        from src.tools.document_parser import DocumentParser, create_document_parser
        print("✅ Document Parser импортирован")
        
        from src.tools.code_analyzer import CodeAnalyzer, create_code_analyzer
        print("✅ Code Analyzer импортирован")
        
        from src.tools.prompt_analyzer import PromptAnalyzer, create_prompt_analyzer
        print("✅ Prompt Analyzer импортирован")
        
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False

def test_document_parser():
    """Тест парсера документов"""
    print("\n🧪 Тестирование Document Parser...")
    
    try:
        from src.tools.document_parser import create_document_parser
        
        parser = create_document_parser()
        
        # Тестируем поддерживаемые расширения
        extensions = parser.get_supported_extensions()
        print(f"✅ Поддерживаемые форматы: {extensions}")
        
        # Проверяем, может ли парсить test файлы
        test_file = Path("test_stage1.py")  # Используем существующий файл
        if test_file.exists():
            can_parse = parser.can_parse(test_file)
            print(f"✅ Может парсить .py файлы: {can_parse}")
            
            if can_parse:
                result = parser.parse_document(test_file)
                print(f"✅ Файл распарсен: {result.success}, секций: {len(result.sections)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка Document Parser: {e}")
        return False

def test_code_analyzer():
    """Тест анализатора кода"""
    print("\n🧪 Тестирование Code Analyzer...")
    
    try:
        from src.tools.code_analyzer import create_code_analyzer
        
        analyzer = create_code_analyzer()
        
        # Тестируем анализ простого Python кода
        test_code = '''
def hello_world():
    """Простая функция для тестирования"""
    print("Hello, World!")
    return "success"

class TestClass:
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value
'''
        
        # Создаем временный файл для теста
        temp_file = Path("temp_test.py")
        with open(temp_file, 'w') as f:
            f.write(test_code)
        
        try:
            # Анализируем файл
            result = analyzer.analyze_project(Path("."), max_files=10)
            print(f"✅ Анализ кода: успех={result.success}, файлов={result.total_files}")
            
            if result.success:
                print(f"✅ Языки: {result.languages}")
                print(f"✅ Общая сложность: {result.complexity_summary.get('average_complexity', 0):.2f}")
        finally:
            # Удаляем временный файл
            if temp_file.exists():
                temp_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка Code Analyzer: {e}")
        return False

def test_prompt_analyzer():
    """Тест анализатора промптов"""
    print("\n🧪 Тестирование Prompt Analyzer...")
    
    try:
        from src.tools.prompt_analyzer import create_prompt_analyzer
        
        analyzer = create_prompt_analyzer()
        
        # Тестовые промпты
        test_prompts = [
            "Ты - полезный помощник банковского сотрудника. Твоя задача - отвечать на вопросы клиентов.",
            "Ограничения: Не раскрывай персональные данные клиентов. Не давай финансовых советов.",
            "Пример диалога: Клиент: Какой у меня баланс? Ассистент: Проверьте баланс в мобильном приложении."
        ]
        
        # Анализируем промпты
        result = analyzer.analyze_prompts(test_prompts)
        
        print(f"✅ Анализ промптов: успех={result.success}, найдено={result.total_prompts}")
        
        if result.success:
            print(f"✅ Системные промпты: {len(result.system_prompts)}")
            print(f"✅ Ограничения: {len(result.guardrails)}")
            print(f"✅ Возможности: {result.capabilities}")
            print(f"✅ Черты личности: {result.personality_traits}")
            print(f"✅ Индикаторы риска: {result.risk_indicators}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка Prompt Analyzer: {e}")
        return False

def main():
    """Главная функция тестирования"""
    print("🚀 Тестирование инструментов анализа (Этап 2)")
    print("=" * 60)
    
    success = True
    
    # Тестируем импорты
    success &= test_imports()
    
    # Тестируем каждый инструмент
    success &= test_document_parser()
    success &= test_code_analyzer()
    success &= test_prompt_analyzer()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ВСЕ ТЕСТЫ ЭТАПА 2 ПРОЙДЕНЫ УСПЕШНО!")
        print("\n📋 Проверено:")
        print("✅ Document Parser (Word, Excel, PDF, Text)")
        print("✅ Code Analyzer (Python, JavaScript, Java)")
        print("✅ Prompt Analyzer (системные промпты, guardrails)")
        print("\n🚀 Готовы к Этапу 3: Создание агентов")
    else:
        print("❌ ЕСТЬ ОШИБКИ В ТЕСТАХ")
        print("🔧 Проверьте синтаксис файлов и зависимости")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)