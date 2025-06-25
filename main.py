# main.py
"""
CLI интерфейс для системы оценки рисков ИИ-агентов
Основная точка входа в систему
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree
from rich.json import JSON

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.workflow import create_workflow_from_env
from src.models.database import get_db_manager
from src.utils.logger import setup_logging, get_logger

# ===== НОВОЕ: Интеграция рассуждений =====
from src.utils.reasoning_integration import enable_all_reasoning, setup_reasoning_env

console = Console()


@click.group()
@click.option('--log-level', default='INFO', help='Уровень логирования')
@click.option('--log-file', default='logs/ai_risk_assessment.log', help='Файл логов')
@click.option('--show-reasoning/--no-reasoning', default=True, help='Показывать рассуждения агентов')
@click.pass_context
def cli(ctx, log_level, log_file, show_reasoning):
    """🤖 Система оценки рисков ИИ-агентов"""
    ctx.ensure_object(dict)
    
    # ===== НОВОЕ: Настройка рассуждений =====
    if show_reasoning:
        setup_reasoning_env()
        enable_all_reasoning()
    
    # Настраиваем логирование
    setup_logging(log_level=log_level, log_file=log_file)
    logger = get_logger()
    
    ctx.obj['logger'] = logger
    ctx.obj['show_reasoning'] = show_reasoning
    
    # Красивый заголовок
    console.print(Panel.fit(
        "[bold blue]🤖 Система оценки рисков ИИ-агентов[/bold blue]\n"
        "Мультиагентная система на базе LangGraph\n"
        f"{'🧠 Рассуждения агентов: ВКЛЮЧЕНЫ' if show_reasoning else '🔇 Рассуждения агентов: ВЫКЛЮЧЕНЫ'}",
        title="AI Risk Assessment System",
        border_style="blue"
    ))


@cli.command()
@click.argument('source_files', nargs=-1, required=True)
@click.option('--agent-name', '-n', help='Имя анализируемого агента')
@click.option('--output', '-o', help='Файл для сохранения результата (JSON)')
@click.option('--quality-threshold', '-q', default=7.0, help='Порог качества для критика (0-10)')
@click.option('--max-retries', '-r', default=3, help='Максимум повторов оценки')
@click.option('--model', '-m', default='qwen3-4b', help='LLM модель')
@click.pass_context
async def assess(ctx, source_files, agent_name, output, quality_threshold, max_retries, model):
    """Запуск оценки рисков ИИ-агента"""
    logger = ctx.obj['logger']
    show_reasoning = ctx.obj.get('show_reasoning', True)
    
    # Проверяем входные файлы
    validated_files = []
    for file_path in source_files:
        path = Path(file_path)
        if path.exists():
            if path.is_dir():
                # Если папка, берем все файлы
                for ext in ['*.py', '*.js', '*.java', '*.txt', '*.md', '*.json', '*.yaml']:
                    validated_files.extend([str(f) for f in path.rglob(ext)])
            else:
                validated_files.append(str(path.absolute()))
        else:
            console.print(f"[red]❌ Файл/папка не найдена: {file_path}[/red]")
            return
    
    if not validated_files:
        console.print("[red]❌ Не найдено валидных файлов для анализа[/red]")
        return
    
    console.print(f"[green]📁 Найдено файлов для анализа: {len(validated_files)}[/green]")
    
    # Показываем первые 10 файлов для подтверждения
    for i, file_path in enumerate(validated_files[:10]):
        console.print(f"  • {file_path}")
    
    if len(validated_files) > 10:
        console.print(f"  ... и еще {len(validated_files) - 10} файлов")
    
    # Создаем workflow
    try:
        console.print("\n[yellow]⚙️ Инициализация workflow...[/yellow]")
        workflow = create_workflow_from_env()
        
        # Проверяем доступность LLM
        llm_healthy = await workflow.profiler.health_check()
        if not llm_healthy:
            console.print("[red]❌ LM Studio недоступен на localhost:1234[/red]")
            console.print("Убедитесь что LM Studio запущен с моделью qwen3-4b")
            return
        
        console.print("[green]✅ LLM сервер доступен[/green]")
        
        if show_reasoning:
            console.print("[blue]🧠 Рассуждения агентов будут отображаться в реальном времени[/blue]")
        
    except Exception as e:
        console.print(f"[red]❌ Ошибка инициализации: {e}[/red]")
        return
    
    # Запускаем оценку с прогрессом
    assessment_id = f"cli_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("🔄 Выполнение оценки рисков...", total=None)
        
        try:
            result = await workflow.run_assessment(
                source_files=validated_files,
                agent_name=agent_name,
                assessment_id=assessment_id
            )
            
            progress.update(task, completed=True)
            
            if result["success"]:
                await _display_assessment_result(result, output)
                logger.bind_context(assessment_id, "cli").info(
                    f"✅ Оценка завершена успешно: {assessment_id}"
                )
            else:
                console.print(f"[red]❌ Ошибка оценки: {result.get('error', 'Неизвестная ошибка')}[/red]")
                
        except KeyboardInterrupt:
            progress.update(task, description="❌ Отменено пользователем")
            console.print("\n[yellow]⚠️ Оценка прервана пользователем[/yellow]")
            
        except Exception as e:
            progress.update(task, description="❌ Ошибка выполнения")
            console.print(f"\n[red]❌ Неожиданная ошибка: {e}[/red]")
            logger.bind_context(assessment_id, "cli").error(f"Ошибка CLI: {e}")


# ===== НОВОЕ: Команда для тестирования БД =====
@cli.command()
@click.pass_context
async def test_db(ctx):
    """Проверка состояния базы данных"""
    try:
        console.print("[blue]🗄️ Проверка базы данных...[/blue]")
        
        db_manager = await get_db_manager()
        console.print("[green]✅ Подключение к БД успешно[/green]")
        
        # Простая статистика
        from sqlalchemy import text
        async with db_manager.async_session() as session:
            
            tables = ['agent_profiles', 'risk_assessments', 'risk_evaluations']
            
            stats_table = Table(title="📊 Статистика БД")
            stats_table.add_column("Таблица", style="cyan")
            stats_table.add_column("Записей", style="white")
            
            for table in tables:
                try:
                    result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    stats_table.add_row(table, str(count))
                except Exception as e:
                    stats_table.add_row(table, f"Ошибка: {str(e)[:30]}")
            
            console.print(stats_table)
        
        await db_manager.close()
        
    except Exception as e:
        console.print(f"[red]❌ Ошибка БД: {e}[/red]")


@cli.command()
@click.argument('assessment_id')
@click.option('--output', '-o', help='Файл для сохранения результата')
@click.pass_context
async def show(ctx, assessment_id, output):
    """Показать результаты оценки по ID"""
    try:
        db_manager = await get_db_manager()
        assessment_data = await db_manager.get_risk_assessment(assessment_id)
        
        if not assessment_data:
            console.print(f"[red]❌ Оценка с ID {assessment_id} не найдена[/red]")
            return
        
        await _display_saved_assessment(assessment_data, output)
        
    except Exception as e:
        console.print(f"[red]❌ Ошибка получения оценки: {e}[/red]")


@cli.command()
@click.option('--limit', '-l', default=10, help='Количество последних оценок')
@click.pass_context
async def list_assessments(ctx, limit):
    """Список последних оценок"""
    try:
        db_manager = await get_db_manager()
        
        # Используем простой способ получения оценок
        from sqlalchemy import select, desc
        from src.models.database import RiskAssessmentDB
        
        async with db_manager.async_session() as session:
            stmt = select(RiskAssessmentDB).order_by(desc(RiskAssessmentDB.assessment_timestamp)).limit(limit)
            result = await session.execute(stmt)
            assessments = result.scalars().all()
            
            if not assessments:
                console.print("[yellow]📭 Нет сохраненных оценок[/yellow]")
                return
            
            table = Table(title=f"📋 Последние {len(assessments)} оценок")
            table.add_column("ID", style="cyan")
            table.add_column("Уровень риска", style="white")
            table.add_column("Балл", style="white")
            table.add_column("Дата", style="green")
            
            for assessment in assessments:
                risk_level = assessment.overall_risk_level
                color = {
                    "low": "green",
                    "medium": "yellow", 
                    "high": "red"
                }.get(risk_level, "white")
                
                table.add_row(
                    assessment.id[:8] + "...",
                    f"[{color}]{risk_level.upper()}[/{color}]",
                    str(assessment.overall_risk_score),
                    str(assessment.assessment_timestamp)[:19]
                )
            
            console.print(table)
        
        await db_manager.close()
        
    except Exception as e:
        console.print(f"[red]❌ Ошибка получения списка: {e}[/red]")


@cli.command()
@click.option('--check-llm', is_flag=True, help='Проверить LLM сервер')
@click.option('--check-db', is_flag=True, help='Проверить базу данных')
@click.pass_context
async def status(ctx, check_llm, check_db):
    """Проверка статуса системы"""
    results = []
    
    # Проверка LLM
    if check_llm or not (check_db):
        try:
            from src.utils.llm_client import get_llm_client
            client = await get_llm_client()
            
            if await client.health_check():
                results.append(("✅ LLM сервер", "Доступен", "green"))
                
                try:
                    models = await client.list_models()
                    results.append(("📋 Доступные модели", f"{len(models)} моделей", "blue"))
                except:
                    results.append(("📋 Доступные модели", "Недоступно", "yellow"))
            else:
                results.append(("❌ LLM сервер", "Недоступен", "red"))
                
        except Exception as e:
            results.append(("❌ LLM сервер", f"Ошибка: {str(e)[:50]}", "red"))
    
    # Проверка базы данных
    if check_db or not (check_llm):
        try:
            db_manager = await get_db_manager()
            results.append(("✅ База данных", "Доступна", "green"))
            
            # Статистика
            try:
                from sqlalchemy import text
                async with db_manager.async_session() as session:
                    result = await session.execute(text("SELECT COUNT(*) FROM risk_assessments"))
                    count = result.scalar()
                    results.append(("📊 Оценок в БД", str(count), "blue"))
            except:
                results.append(("📊 Оценок в БД", "Недоступно", "yellow"))
                
            await db_manager.close()
                
        except Exception as e:
            results.append(("❌ База данных", f"Ошибка: {str(e)[:50]}", "red"))
    
    # Отображаем результаты
    table = Table(title="Статус компонентов системы")
    table.add_column("Компонент", style="bold")
    table.add_column("Статус")
    
    for component, status, color in results:
        table.add_row(component, f"[{color}]{status}[/{color}]")
    
    console.print(table)


# Замените demo команду в main.py на эту версию:

@cli.command()
@click.pass_context
async def demo(ctx):
    """Демонстрация системы на тестовых данных"""
    console.print("[blue]🎭 Запуск демонстрации системы...[/blue]\n")
    
    # Создаем тестовые файлы
    demo_dir = Path("demo_data")
    demo_dir.mkdir(exist_ok=True)
    
    # Тестовый агент
    test_agent_code = '''# demo_agent.py
"""
Демонстрационный ИИ-агент для системы оценки рисков
"""

class DemoAgent:
    def __init__(self):
        self.name = "DemoAgent"
        self.version = "1.0"
        self.system_prompt = """
        Ты - демонстрационный помощник банка.
        Твоя задача - отвечать на вопросы клиентов о банковских услугах.
        
        Ограничения:
        - Не раскрывай персональные данные клиентов
        - Не давай финансовых советов
        - Направляй сложные вопросы к специалистам
        """
    
    def process_query(self, query: str) -> str:
        """Обработка запроса клиента"""
        # Простая логика обработки
        if "баланс" in query.lower():
            return "Проверьте баланс в мобильном приложении банка"
        elif "кредит" in query.lower():
            return "Для вопросов по кредитам обратитесь к кредитному специалисту"
        else:
            return "Как я могу помочь вам сегодня?"
'''
    
    test_description = '''Демонстрационный агент
    
Название: DemoAgent
Тип: Банковский помощник
Целевая аудитория: Клиенты банка
Автономность: Под надзором

Описание:
Простой чат-бот для ответов на базовые вопросы клиентов банка.
Имеет ограниченную функциональность и требует человеческого надзора.

Возможности:
- Ответы на часто задаваемые вопросы
- Направление к специалистам
- Простая навигация по услугам

Ограничения:
- Не обрабатывает персональные данные
- Не принимает финансовых решений
- Требует подтверждения для сложных операций
'''
    
    try:
        # Сохраняем тестовые файлы
        (demo_dir / "demo_agent.py").write_text(test_agent_code, encoding='utf-8')
        (demo_dir / "description.txt").write_text(test_description, encoding='utf-8')
        
        console.print(f"[green]📁 Созданы тестовые файлы в {demo_dir}[/green]")
        
        # Запускаем оценку напрямую через workflow (БЕЗ рекурсии!)
        demo_files = [str(demo_dir / "demo_agent.py"), str(demo_dir / "description.txt")]
        
        console.print("[blue]📊 Запускаем демонстрационную оценку...[/blue]\n")
        
        # Создаем workflow
        workflow = create_workflow_from_env()
        
        # Проверяем LLM
        llm_healthy = await workflow.profiler.health_check()
        if not llm_healthy:
            console.print("[red]❌ LM Studio недоступен. Запустите LM Studio с моделью qwen3-4b[/red]")
            return
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("🔄 Демонстрационная оценка...", total=None)
            
            # Запускаем workflow напрямую
            result = await workflow.run_assessment(
                source_files=demo_files,
                agent_name="DemoAgent",
                assessment_id=f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            progress.update(task, completed=True)
            
            if result["success"]:
                console.print("\n[green]✅ Демонстрация завершена успешно![/green]")
                await _display_assessment_result(result)
            else:
                console.print(f"\n[red]❌ Ошибка демонстрации: {result.get('error')}[/red]")
                
    except Exception as e:
        console.print(f"\n[red]❌ Ошибка демонстрации: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
    
    finally:
        # Очищаем тестовые файлы
        try:
            import shutil
            if demo_dir.exists():
                shutil.rmtree(demo_dir)
                console.print(f"[dim]🗑️ Очищены временные файлы[/dim]")
        except:
            pass


# Вспомогательные функции для отображения результатов
async def _display_assessment_result(result, output_file=None):
    """Отображение результатов оценки"""
    assessment = result.get("assessment")
    if not assessment:
        console.print("[red]❌ Нет данных для отображения[/red]")
        return
    
    # Основная информация
    console.print(Panel(
        f"[bold green]🎯 Оценка завершена успешно![/bold green]\n\n"
        f"Assessment ID: {assessment.get('id', 'unknown')}\n"
        f"Общий уровень риска: [bold]{assessment.get('overall_risk_level', 'unknown').upper()}[/bold]\n"
        f"Общий балл: {assessment.get('overall_risk_score', 0)}/25\n"
        f"Время обработки: {assessment.get('processing_time_seconds', 0):.1f} секунд",
        title="📊 Результаты оценки",
        border_style="green"
    ))
    
    # Детальные оценки
    risk_evaluations = result.get("risk_evaluations", {})
    if risk_evaluations:
        table = Table(title="🔍 Детальные оценки рисков")
        table.add_column("Тип риска", style="cyan")
        table.add_column("Балл", style="white")
        table.add_column("Уровень", style="white")
        
        for risk_type, evaluation in risk_evaluations.items():
            level = evaluation.get('risk_level', 'unknown')
            color = {
                'low': 'green',
                'medium': 'yellow',
                'high': 'red'
            }.get(level, 'white')
            
            table.add_row(
                risk_type,
                f"{evaluation.get('total_score', 0)}/25",
                f"[{color}]{level.upper()}[/{color}]"
            )
        
        console.print(table)
    
    # Рекомендации
    recommendations = assessment.get("priority_recommendations", [])
    if recommendations:
        console.print("\n[bold green]💡 Приоритетные рекомендации:[/bold green]")
        for i, rec in enumerate(recommendations[:5], 1):
            console.print(f"  {i}. {rec}")
    
    # Сохранение в файл
    if output_file:
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            
            console.print(f"\n[green]💾 Результаты сохранены в {output_file}[/green]")
        except Exception as e:
            console.print(f"\n[red]❌ Ошибка сохранения: {e}[/red]")


async def _display_saved_assessment(assessment_data, output_file=None):
    """Отображение сохраненной оценки"""
    console.print("[blue]📋 Информация из базы данных[/blue]")
    # Здесь можно добавить логику отображения сохраненных данных


def main():
    """Главная функция CLI"""
    try:
        # Создаем папки если их нет
        Path("logs").mkdir(exist_ok=True)
        Path("data").mkdir(exist_ok=True)
        
        # Запускаем CLI
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 До свидания![/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Критическая ошибка: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    
    # ИСПРАВЛЕННЫЙ патч для async команд
    def make_async(f):
        def wrapper(*args, **kwargs):
            return asyncio.run(f(*args, **kwargs))
        return wrapper
    
    # Применяем патч
    assess.callback = make_async(assess.callback)
    show.callback = make_async(show.callback)
    list_assessments.callback = make_async(list_assessments.callback)
    status.callback = make_async(status.callback)
    demo.callback = make_async(demo.callback)
    # test_db.callback = make_async(test_db.callback)  # Пока закомментировано
    
    main()