# test_setup.py
"""Простой тест для проверки настройки"""

def test_imports():
    """Тест импорта основных библиотек"""
    try:
        import langgraph
        import langchain
        import sqlalchemy
        import rich
        import click
        import pandas
        import numpy
        print("✅ Все библиотеки импортируются успешно!")
        return True
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False

def test_env():
    """Тест загрузки переменных окружения"""
    try:
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        
        llm_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234")
        llm_model = os.getenv("LLM_MODEL", "qwen3-4b")
        
        print(f"✅ LLM URL: {llm_url}")
        print(f"✅ LLM Model: {llm_model}")
        return True
    except Exception as e:
        print(f"❌ Ошибка загрузки .env: {e}")
        return False

def test_document_parsers():
    """Тест библиотек для парсинга документов"""
    try:
        import PyPDF2
        import docx
        import openpyxl
        print("✅ Все библиотеки для парсинга документов работают!")
        return True
    except ImportError as e:
        print(f"❌ Ошибка импорта парсеров: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Тестирование настройки проекта")
    print("=" * 40)
    
    success = True
    success &= test_imports()
    success &= test_env()
    success &= test_document_parsers()
    
    print("=" * 40)
    if success:
        print("🎉 Настройка проекта успешна!")
        print("\n📋 Следующие шаги:")
        print("1. Запустите LM Studio с моделью qwen3-4b")
        print("2. Запустите сервер на localhost:1234")
        print("3. Начинайте разработку агентов!")
    else:
        print("❌ Есть проблемы с настройкой")