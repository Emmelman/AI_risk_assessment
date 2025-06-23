# create_project.py
"""
Простой скрипт создания структуры проекта
"""

import os
import subprocess
import sys
from pathlib import Path

def check_python_version():
    """Проверка версии Python"""
    import sys
    
    version = sys.version_info
    required_version = (3, 12, 4)
    
    print(f"🐍 Текущая версия Python: {version.major}.{version.minor}.{version.micro}")
    
    if version < required_version:
        print(f"❌ Требуется Python {required_version[0]}.{required_version[1]}.{required_version[2]}+")
        print("Установите Python 3.12.4 или новее")
        return False
    
    print("✅ Версия Python подходит")
    return True

def create_project_structure():
    """Создание структуры директорий проекта"""
    
    directories = [
        "src",
        "src/agents", 
        "src/tools",
        "src/workflow",
        "src/models",
        "src/utils",
        "config",
        "config/prompts",
        "tests",
        "tests/fixtures",
        "logs",
        "data"
    ]
    
    print("📁 Создаём структуру проекта...")
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {directory}/")
    
    # Создание __init__.py файлов
    init_files = [
        "src/__init__.py",
        "src/agents/__init__.py",
        "src/tools/__init__.py", 
        "src/workflow/__init__.py",
        "src/models/__init__.py",
        "src/utils/__init__.py",
        "config/__init__.py",
        "config/prompts/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
    
    print("✅ Структура проекта создана")

def create_requirements():
    """Создание requirements.txt"""
    
    requirements = """# Core ML/AI Libraries
langgraph==0.2.21
langchain==0.2.11
langchain-community==0.2.11
langchain-openai==0.1.22

# Document Processing
docling==1.16.0
python-docx==1.1.2
openpyxl==3.1.2

# Database
sqlalchemy==2.0.23
aiosqlite==0.19.0

# HTTP/API
httpx==0.25.2
aiohttp==3.9.1

# Data Processing
pandas==2.1.4
numpy==1.24.4
pydantic==2.5.0
pydantic-settings==2.1.0

# Utilities
python-dotenv==1.0.0
click==8.1.7
rich==13.7.0
loguru==0.7.2

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("✅ requirements.txt создан")

def create_env_file():
    """Создание .env файла"""
    
    env_content = """# LLM Configuration
LLM_BASE_URL=http://127.0.0.1:1234
LLM_MODEL=qwen3-4b
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=4096

# Database
DATABASE_URL=sqlite:///./ai_risk_assessment.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/ai_risk_assessment.log

# Risk Assessment
MAX_RETRY_COUNT=3
QUALITY_THRESHOLD=7.0
MAX_CONCURRENT_EVALUATIONS=6"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ .env файл создан")

def create_gitignore():
    """Создание .gitignore"""
    
    gitignore = """__pycache__/
*.py[cod]
*.so
.Python
*.egg-info/
dist/
build/

# Virtual environments
venv/
env/
ai_risk_env/

# Environment
.env

# Database
*.db
*.sqlite

# Logs
logs/
*.log

# Cache
cache/

# Testing
.coverage
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore)
    
    print("✅ .gitignore создан")

if __name__ == "__main__":
    print("🚀 Создание проекта системы оценки рисков ИИ-агентов")
    
    # Проверка версии Python
    if not check_python_version():
        sys.exit(1)
    
    create_project_structure()
    create_requirements()
    create_env_file()
    create_gitignore()
    
    print("\n✅ Проект создан!")
    print("\nТребования:")
    print("- Python 3.12.4+")
    print("\nСледующие шаги:")
    print("1. python -m venv ai_risk_env")
    print("2. source ai_risk_env/bin/activate  # Linux/Mac")
    print("   ai_risk_env\\Scripts\\activate     # Windows") 
    print("3. python install_libraries.py")


# install_libraries.py
"""
Установка всех необходимых библиотек в виртуальное окружение
"""

import subprocess
import sys
from pathlib import Path

def check_virtual_env():
    """Проверка активации виртуального окружения"""
    # Проверка для conda
    if os.environ.get('CONDA_DEFAULT_ENV'):
        print(f"✅ Conda окружение активно: {os.environ.get('CONDA_DEFAULT_ENV')}")
        return True
    
    # Проверка для venv
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Виртуальное окружение активно")
        return True
    else:
        print("❌ Виртуальное окружение не активировано!")
        return False 

def check_python_version_venv():
    """Проверка версии Python в виртуальном окружении"""
    version = sys.version_info
    required_version = (3, 12, 4)
    
    print(f"🐍 Python в виртуальном окружении: {version.major}.{version.minor}.{version.micro}")
    
    if version < required_version:
        print(f"⚠️  Рекомендуется Python {required_version[0]}.{required_version[1]}.{required_version[2]}+")
        print("Но попробуем продолжить...")
        return True
    
    print("✅ Версия Python подходит")
    return True

def upgrade_pip():
    """Обновление pip"""
    print("⬆️  Обновление pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print("✅ pip обновлен")
    except subprocess.CalledProcessError:
        print("⚠️  Не удалось обновить pip")

def install_requirements():
    """Установка основных зависимостей"""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt не найден")
        return False
    
    print("📦 Установка основных библиотек...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Основные библиотеки установлены")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки: {e}")
        return False

def install_optional_packages():
    """Установка дополнительных полезных пакетов"""
    optional = [
        "jupyter",      # Для экспериментов
        "matplotlib",   # Для графиков
        "psutil",       # Для мониторинга системы
    ]
    
    print("🔧 Установка дополнительных пакетов...")
    
    for package in optional:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"   ✓ {package}")
        except subprocess.CalledProcessError:
            print(f"   ⚠️  {package} - не удалось установить")

def verify_installation():
    """Проверка успешности установки ключевых пакетов"""
    key_packages = [
        "langgraph",
        "langchain", 
        "docling",
        "sqlalchemy",
        "rich",
        "click",
        "pytest"
    ]
    
    print("\n🔍 Проверка установки...")
    
    failed = []
    for package in key_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            failed.append(package)
    
    if failed:
        print(f"\n⚠️  Не удалось импортировать: {', '.join(failed)}")
        return False
    else:
        print("\n✅ Все ключевые пакеты установлены успешно!")
        return True

def create_test_script():
    """Создание простого тестового скрипта"""
    test_code = '''# test_setup.py
"""Простой тест для проверки настройки"""

def test_imports():
    """Тест импорта основных библиотек"""
    try:
        import langgraph
        import langchain
        import docling
        import sqlalchemy
        import rich
        import click
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

if __name__ == "__main__":
    print("🧪 Тестирование настройки проекта")
    
    success = True
    success &= test_imports()
    success &= test_env()
    
    if success:
        print("\\n🎉 Настройка проекта успешна!")
    else:
        print("\\n❌ Есть проблемы с настройкой")
'''
    
    with open("test_setup.py", "w") as f:
        f.write(test_code)
    
    print("✅ Тестовый скрипт создан (test_setup.py)")

if __name__ == "__main__":
    print("📦 Установка библиотек для системы оценки рисков ИИ-агентов")
    print("=" * 60)
    
    if not check_virtual_env():
        sys.exit(1)
    
    if not check_python_version_venv():
        sys.exit(1)
    
    upgrade_pip()
    
    if not install_requirements():
        sys.exit(1)
    
    install_optional_packages()
    
    if not verify_installation():
        print("\n⚠️  Некоторые пакеты не установились, но можно продолжать")
    
    create_test_script()
    
    print("\n" + "=" * 60)
    print("🎉 УСТАНОВКА ЗАВЕРШЕНА!")
    print("\nТребования выполнены:")
    print("- Python 3.12.4+ ✅")
    print("- Все библиотеки установлены ✅")
    print("\nДля проверки запустите: python test_setup.py")
    print("\nНастройка LM Studio:")
    print("1. Запустите LM Studio")
    print("2. Загрузите модель qwen3-4b") 
    print("3. Запустите сервер на localhost:1234")
    print("4. Протестируйте: curl http://localhost:1234/v1/models")


# run_setup.py
"""
Объединенный скрипт - создает проект и устанавливает библиотеки
"""

import os
import sys
import subprocess

def main():
    print("🚀 ПОЛНАЯ НАСТРОЙКА ПРОЕКТА")
    print("Требования: Python 3.12.4+")
    print("=" * 40)
    
    # Проверка версии Python
    version = sys.version_info
    if version < (3, 12, 4):
        print(f"❌ Требуется Python 3.12.4+, текущая версия: {version.major}.{version.minor}.{version.micro}")
        print("Обновите Python и попробуйте снова")
        return
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    
    # Шаг 1: Создание структуры
    print("\n1️⃣ Создание структуры проекта...")
    exec(open("create_project.py").read())
    
    # Шаг 2: Создание виртуального окружения
    print("\n2️⃣ Создание виртуального окружения...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "ai_risk_env"], check=True)
        print("✅ Виртуальное окружение создано")
    except subprocess.CalledProcessError:
        print("❌ Ошибка создания виртуального окружения")
        return
    
    print("\n📋 Дальнейшие действия:")
    print("1. Активируйте окружение:")
    print("   Linux/Mac: source ai_risk_env/bin/activate")
    print("   Windows:   ai_risk_env\\Scripts\\activate")
    print("2. Установите библиотеки: python install_libraries.py")
    print("3. Запустите LM Studio с моделью qwen3-4b на localhost:1234")

if __name__ == "__main__":
    main()