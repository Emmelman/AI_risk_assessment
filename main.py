# main.py
"""
Главный модуль CLI приложения для оценки рисков ИИ-агентов

ОБНОВЛЕНО: Интеграция с центральным LLM конфигуратором
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import click

# Импорты системы оценки рисков
from src.workflow.graph_builder import (
    create_risk_assessment_workflow, 
    validate_workflow_dependencies,
    print_workflow_status
)
from src.config import get_global_llm_config, LLMConfigManager
from src.utils.logger import setup_logging, get_logger
from src.models.database import init_database, get_assessment_by_id


def setup_environment():
    """Настройка окружения приложения"""
    
    # Настраиваем логирование
    setup_logging()
    
    # Инициализируем базу данных
    init_database()
    
    # Проверяем центральный конфигуратор LLM
    try:
        config_manager = get_global_llm_config()
        if not config_manager.validate_configuration():
            click.echo("⚠️  Предупреждение: Конфигурация LLM не прошла валидацию", err=True)
        
        if not config_manager.is_available():
            click.echo("⚠️  Предупреждение: LLM сервер недоступен", err=True)
            
    except Exception as e:
        click.echo(f"❌ Ошибка инициализации LLM конфигуратора: {e}", err=True)
        sys.exit(1)


@click.group()
@click.version_option(version="2.0.0", message="AI Risk Assessment System v%(version)s")
def cli():
    """
    🤖 AI Risk Assessment System
    
    Система комплексной оценки операционных рисков ИИ-агентов
    для банковской сферы с поддержкой российского регулирования.
    """
    setup_environment()


@cli.command()
@click.argument('paths', nargs=-1, required=True)
@click.option('--agent-name', '-n', default=None, help='Название агента')
@click.option('--output', '-o', type=click.Path(), help='Файл для сохранения результата')
@click.option('--quality-threshold', '-q', type=float, default=7.0, 
              help='Порог качества критика (0-10)')
@click.option('--max-retries', '-r', type=int, default=3, 
              help='Максимум повторов при неприемлемом качестве')
@click.option('--format', 'output_format', type=click.Choice(['json', 'yaml', 'txt']), 
              default='json', help='Формат вывода результата')
@click.option('--verbose', '-v', is_flag=True, help='Подробный вывод')
def assess(paths: tuple, agent_name: Optional[str], output: Optional[str], 
           quality_threshold: float, max_retries: int, output_format: str, verbose: bool):
    """
    🎯 Комплексная оценка рисков ИИ-агента
    
    PATHS: Пути к файлам или папкам с данными агента
    
    Примеры использования:
    
      # Оценка одного файла
      python main.py assess agent.py
      
      # Оценка папки с проектом
      python main.py assess ./my_agent_project/
      
      # С настройками качества
      python main.py assess ./agent/ --quality-threshold 8.0 --max-retries 5
      
      # Сохранение в файл  
      python main.py assess ./agent/ --output results.json
    """
    
    logger = get_logger()
    
    # Валидация путей
    file_paths = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            click.echo(f"❌ Путь не найден: {path_str}", err=True)
            sys.exit(1)
        file_paths.append(str(path.resolve()))
    
    # Определяем имя агента
    if not agent_name:
        if len(file_paths) == 1:
            agent_name = Path(file_paths[0]).stem
        else:
            agent_name = f"Agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Проверяем конфигурацию перед запуском
    config_manager = get_global_llm_config()
    if verbose:
        status_info = config_manager.get_status_info()
        click.echo(f"🔧 Используется провайдер: {status_info['provider']}")
        click.echo(f"🔧 Модель: {status_info['model']}")
        click.echo(f"🔧 Статус: {'✅ Доступен' if status_info['is_available'] else '❌ Недоступен'}")
    
    async def run_assessment():
        """Асинхронный запуск оценки"""
        
        try:
            # Создаем workflow (использует центральный конфигуратор)
            workflow = create_risk_assessment_workflow(
                quality_threshold=quality_threshold,
                max_retries=max_retries
            )
            
            if verbose:
                click.echo(f"🚀 Запуск оценки агента '{agent_name}'...")
                click.echo(f"📁 Анализируемые пути: {', '.join(file_paths)}")
            
            # Выполняем оценку
            result = await workflow.run_assessment(
                source_files=file_paths,
                agent_name=agent_name
            )
            
            # Форматируем результат
            if output_format == 'json':
                formatted_result = json.dumps(result, ensure_ascii=False, indent=2, default=str)
            elif output_format == 'yaml':
                import yaml
                formatted_result = yaml.dump(result, allow_unicode=True, default_flow_style=False)
            else:  # txt
                formatted_result = format_result_as_text(result)
            
            # Сохраняем или выводим результат
            if output:
                Path(output).write_text(formatted_result, encoding='utf-8')
                click.echo(f"✅ Результат сохранен в: {output}")
            else:
                click.echo(formatted_result)
            
            # Краткая сводка
            if verbose or not output:
                print_assessment_summary(result)
                
        except Exception as e:
            logger.error(f"Ошибка выполнения оценки: {e}")
            click.echo(f"❌ Ошибка: {e}", err=True)
            sys.exit(1)
    
    # Запускаем асинхронную оценку
    asyncio.run(run_assessment())


@cli.command()
@click.option('--check-llm', is_flag=True, help='Проверить доступность LLM')
@click.option('--check-db', is_flag=True, help='Проверить подключение к БД')
@click.option('--check-workflow', is_flag=True, help='Проверить готовность workflow')
@click.option('--detailed', is_flag=True, help='Подробная диагностика')
def status(check_llm: bool, check_db: bool, check_workflow: bool, detailed: bool):
    """
    📊 Проверка статуса системы
    
    Проверяет готовность всех компонентов системы оценки рисков.
    """
    
    if not any([check_llm, check_db, check_workflow]) or detailed:
        # Если не указаны конкретные проверки, делаем все
        check_llm = check_db = check_workflow = True
    
    if check_llm:
        click.echo("🔧 Проверка LLM конфигурации:")
        check_llm_status(detailed)
    
    if check_db:
        click.echo("\n💾 Проверка базы данных:")
        check_database_status(detailed)
    
    if check_workflow:
        click.echo("\n⚙️  Проверка workflow:")
        check_workflow_status(detailed)


@cli.command()
def demo():
    """
    🎬 Демонстрация системы на тестовых данных
    
    Запускает оценку рисков на встроенных тестовых данных
    для демонстрации возможностей системы.
    """
    
    click.echo("🎬 Запуск демонстрации системы оценки рисков...")
    
    # Создаем тестовые данные
    test_data = create_demo_data()
    
    async def run_demo():
        """Асинхронный запуск демо"""
        
        try:
            workflow = create_risk_assessment_workflow(
                quality_threshold=6.0,  # Сниженный порог для демо
                max_retries=2
            )
            
            result = await workflow.run_assessment(
                file_paths=test_data["file_paths"],
                agent_name="Demo Banking Assistant"
            )
            
            # Выводим красивый результат
            print_demo_result(result)
            
        except Exception as e:
            click.echo(f"❌ Ошибка демонстрации: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(run_demo())


@cli.command()
@click.argument('assessment_id')
@click.option('--format', 'output_format', type=click.Choice(['json', 'yaml', 'summary']), 
              default='summary', help='Формат вывода')
def show(assessment_id: str, output_format: str):
    """
    📋 Показать результаты конкретной оценки
    
    ASSESSMENT_ID: Идентификатор оценки для просмотра
    """
    
    try:
        assessment = get_assessment_by_id(assessment_id)
        if not assessment:
            click.echo(f"❌ Оценка с ID '{assessment_id}' не найдена", err=True)
            sys.exit(1)
        
        if output_format == 'json':
            click.echo(json.dumps(assessment, ensure_ascii=False, indent=2, default=str))
        elif output_format == 'yaml':
            import yaml
            click.echo(yaml.dump(assessment, allow_unicode=True, default_flow_style=False))
        else:  # summary
            print_assessment_summary(assessment)
            
    except Exception as e:
        click.echo(f"❌ Ошибка получения оценки: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--provider', type=click.Choice(['lm_studio', 'gigachat']), 
              help='Переключить LLM провайдера')
@click.option('--model', help='Переопределить модель LLM')
@click.option('--temperature', type=float, help='Переопределить температуру')
@click.option('--show-config', is_flag=True, help='Показать текущую конфигурацию')
def config(provider: Optional[str], model: Optional[str], temperature: Optional[float], show_config: bool):
    """
    ⚙️  Управление конфигурацией LLM
    
    Позволяет просматривать и изменять настройки LLM провайдера.
    """
    
    config_manager = get_global_llm_config()
    
    if show_config:
        status_info = config_manager.get_status_info()
        click.echo("📋 Текущая конфигурация LLM:")
        click.echo(f"  Провайдер: {status_info['provider_type']}")
        click.echo(f"  Модель: {status_info['model']}")
        click.echo(f"  URL: {status_info['base_url']}")
        click.echo(f"  Температура: {status_info['temperature']}")
        click.echo(f"  Доступен: {'✅' if status_info['is_available'] else '❌'}")
        return
    
    if provider:
        try:
            new_manager = LLMConfigManager.create_with_provider_type(provider)
            from src.config import set_global_llm_config
            set_global_llm_config(new_manager)
            click.echo(f"✅ Провайдер изменен на: {provider}")
        except Exception as e:
            click.echo(f"❌ Ошибка смены провайдера: {e}", err=True)
    
    # TODO: Реализовать переопределение модели и температуры
    # Пока что эти параметры берутся из переменных окружения
    
    if not any([provider, model, temperature, show_config]):
        click.echo("❓ Используйте --show-config для просмотра или укажите параметры для изменения")


# ===============================
# Вспомогательные функции
# ===============================

def check_llm_status(detailed: bool = False):
    """Проверка статуса LLM"""
    
    try:
        config_manager = get_global_llm_config()
        status_info = config_manager.get_status_info()
        
        # Основная информация
        click.echo(f"  Провайдер: {status_info['provider']}")
        click.echo(f"  Модель: {status_info['model']}")
        click.echo(f"  Доступность: {'✅' if status_info['is_available'] else '❌'}")
        click.echo(f"  Конфигурация: {'✅' if status_info['is_valid'] else '❌'}")
        
        if detailed:
            click.echo(f"  URL: {status_info['base_url']}")
            click.echo(f"  Температура: {status_info['temperature']}")
            click.echo(f"  Max tokens: {status_info['max_tokens']}")
            
    except Exception as e:
        click.echo(f"  ❌ Ошибка: {e}")


def check_database_status(detailed: bool = False):
    """Проверка статуса базы данных"""
    
    try:
        from src.models.database import test_db_connection
        if test_db_connection():
            click.echo("  ✅ База данных доступна")
        else:
            click.echo("  ❌ База данных недоступна")
            
    except Exception as e:
        click.echo(f"  ❌ Ошибка: {e}")


def check_workflow_status(detailed: bool = False):
    """Проверка статуса workflow"""
    
    try:
        dependencies = validate_workflow_dependencies()
        all_ready = all(dependencies.values())
        
        click.echo(f"  Готовность: {'✅' if all_ready else '❌'}")
        
        if detailed or not all_ready:
            for component, status in dependencies.items():
                status_icon = "✅" if status else "❌"
                click.echo(f"    {status_icon} {component}")
                
    except Exception as e:
        click.echo(f"  ❌ Ошибка: {e}")


def create_demo_data() -> Dict[str, Any]:
    """Создание тестовых данных для демонстрации"""
    
    import tempfile
    import os
    
    # Создаем временные файлы с тестовыми данными
    demo_files = []
    
    # Файл с кодом агента
    agent_code = '''
class BankingAssistant:
    """ИИ-помощник для банковских операций"""
    
    def __init__(self):
        self.model = "qwen3-4b"
        self.capabilities = [
            "Консультации по продуктам",
            "Помощь с операциями", 
            "Анализ финансов"
        ]
    
    def process_query(self, query: str) -> str:
        # Обработка запроса клиента
        return self.llm_call(query)
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(agent_code)
        demo_files.append(f.name)
    
    # Файл с конфигурацией
    config_data = '''
{
    "agent_name": "Banking Assistant",
    "target_audience": ["Клиенты банка", "Сотрудники"],
    "data_access": ["internal", "confidential"],
    "guardrails": [
        "Не разглашать данные клиентов",
        "Требовать аутентификацию для операций"
    ]
}
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(config_data)
        demo_files.append(f.name)
    
    return {"file_paths": demo_files}


def format_result_as_text(result: Dict[str, Any]) -> str:
    """Форматирование результата в текстовом виде"""
    
    text_parts = []
    
    text_parts.append(f"🤖 ОЦЕНКА РИСКОВ АГЕНТА: {result.get('agent_name', 'Unknown')}")
    text_parts.append("=" * 60)
    
    # Общий риск
    overall_level = result.get('overall_risk_level', 'unknown')
    overall_score = result.get('overall_risk_score', 0)
    
    risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(overall_level, "⚪")
    text_parts.append(f"{risk_emoji} Общий уровень риска: {overall_level.upper()} ({overall_score}/10)")
    text_parts.append("")
    
    # Детальные оценки
    text_parts.append("📊 ДЕТАЛЬНЫЕ ОЦЕНКИ:")
    risk_evaluations = result.get('risk_evaluations', {})
    
    for risk_type, evaluation in risk_evaluations.items():
        risk_score = evaluation.get('risk_score', 0)
        risk_level = evaluation.get('risk_level', 'unknown')
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk_level, "⚪")
        
        text_parts.append(f"  {risk_emoji} {risk_type}: {risk_level} ({risk_score}/10)")
    
    # Рекомендации
    recommendations = result.get('priority_recommendations', [])
    if recommendations:
        text_parts.append("")
        text_parts.append("💡 ПРИОРИТЕТНЫЕ РЕКОМЕНДАЦИИ:")
        for i, rec in enumerate(recommendations[:5], 1):
            text_parts.append(f"  {i}. {rec}")
    
    # Время выполнения
    processing_time = result.get('processing_time', 0)
    text_parts.append("")
    text_parts.append(f"⏱️  Время обработки: {processing_time:.1f} секунд")
    
    return "\n".join(text_parts)


def print_assessment_summary(result: Dict[str, Any]):
    """Вывод краткой сводки оценки"""
    
    agent_name = result.get('agent_name', 'Unknown')
    overall_level = result.get('overall_risk_level', 'unknown')
    overall_score = result.get('overall_risk_score', 0)
    
    risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(overall_level, "⚪")
    
    click.echo(f"\n📊 СВОДКА ОЦЕНКИ:")
    click.echo(f"  Агент: {agent_name}")
    click.echo(f"  {risk_emoji} Общий риск: {overall_level.upper()} ({overall_score}/10)")
    
    risk_evaluations = result.get('risk_evaluations', {})
    if risk_evaluations:
        high_risks = [rt for rt, ev in risk_evaluations.items() if ev.get('risk_level') == 'high']
        if high_risks:
            click.echo(f"  ⚠️  Высокие риски: {', '.join(high_risks)}")


def print_demo_result(result: Dict[str, Any]):
    """Красивый вывод результата демонстрации"""
    
    click.echo("\n" + "🎬 РЕЗУЛЬТАТЫ ДЕМОНСТРАЦИИ".center(60, "="))
    click.echo(format_result_as_text(result))
    click.echo("=" * 60)
    click.echo("✅ Демонстрация завершена успешно!")
    click.echo("\n💡 Для реальной оценки используйте: python main.py assess <путь_к_агенту>")


if __name__ == '__main__':
    cli()