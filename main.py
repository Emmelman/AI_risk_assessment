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


console = Console()


@click.group()
@click.option('--log-level', default='INFO', help='Уровень логирования')
@click.option('--log-file', default='logs/ai_risk_assessment.log', help='Файл логов')
@click.pass_context
def cli(ctx, log_level, log_file):
    """🤖 Система оценки рисков ИИ-агентов"""
    ctx.ensure_object(dict)
    
    # Настраиваем логирование
    setup_logging(log_level=log_level, log_file=log_file)
    logger = get_logger()
    
    ctx.obj['logger'] = logger
    
    # Красивый заголовок
    console.print(Panel.fit(
        "[bold blue]🤖 Система оценки рисков ИИ-агентов[/bold blue]\n"
        "Мультиагентная система на базе LangGraph",
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
    
    # Проверяем входные файлы
    validated_files = []
    for file_path in source_files:
        path = Path(file_path)
        if path.exists():
            validated_files.append(str(path.absolute()))
        else:
            console.print(f"[red]❌ Файл не найден: {file_path}[/red]")
            return
    
    if not validated_files:
        console.print("[red]❌ Не найдено валидных файлов для анализа[/red]")
        return
    
    console.print(f"[green]📁 Найдено файлов для анализа: {len(validated_files)}[/green]")
    for file_path in validated_files:
        console.print(f"  • {file_path}")
    
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
        profiles = await db_manager.list_agent_profiles(limit=limit)
        
        if not profiles:
            console.print("[yellow]📭 Нет сохраненных оценок[/yellow]")
            return
        
        table = Table(title="🤖 Последние оценки ИИ-агентов")
        table.add_column("ID", style="cyan")
        table.add_column("Имя агента", style="green")
        table.add_column("Тип", style="blue")
        table.add_column("Дата обновления", style="dim")
        
        for profile in profiles:
            table.add_row(
                profile["id"][:8] + "...",
                profile["name"],
                profile["agent_type"],
                profile["updated_at"].strftime("%Y-%m-%d %H:%M")
            )
        
        console.print(table)
        
        # Показываем детали по первой оценке
        if profiles:
            first_profile = profiles[0]
            assessments = await db_manager.get_assessments_for_agent(first_profile["id"], limit=1)
            
            if assessments:
                latest = assessments[0]
                console.print(f"\n[dim]💡 Для подробностей используйте: python main.py show {latest['id']}[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Ошибка получения списка: {e}[/red]")


@cli.command()
@click.option('--check-llm', is_flag=True, help='Проверить доступность LLM')
@click.option('--check-db', is_flag=True, help='Проверить базу данных')
@click.pass_context
async def status(ctx, check_llm, check_db):
    """Проверка статуса системы"""
    console.print("[blue]🔍 Проверка статуса системы...[/blue]\n")
    
    results = []
    
    # Проверка LLM
    if check_llm or not (check_db):
        try:
            workflow = create_workflow_from_env()
            llm_healthy = await workflow.profiler.health_check()
            
            if llm_healthy:
                results.append(("✅ LLM сервер", "Доступен", "green"))
                
                # Получаем модели
                try:
                    models = await workflow.profiler.llm_client.get_available_models()
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
            profiles = await db_manager.list_agent_profiles(limit=1)
            results.append(("✅ База данных", "Доступна", "green"))
            
            # Статистика
            try:
                total_profiles = len(await db_manager.list_agent_profiles(limit=1000))
                results.append(("📊 Агентов в БД", str(total_profiles), "blue"))
            except:
                results.append(("📊 Агентов в БД", "Недоступно", "yellow"))
                
        except Exception as e:
            results.append(("❌ База данных", f"Ошибка: {str(e)[:50]}", "red"))
    
    # Отображаем результаты
    table = Table(title="Статус компонентов системы")
    table.add_column("Компонент", style="bold")
    table.add_column("Статус")
    
    for component, status, color in results:
        table.add_row(component, f"[{color}]{status}[/{color}]")
    
    console.print(table)


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
    
    # Сохраняем тестовые файлы
    (demo_dir / "demo_agent.py").write_text(test_agent_code, encoding='utf-8')
    (demo_dir / "description.txt").write_text(test_description, encoding='utf-8')
    
    console.print(f"[green]📁 Созданы тестовые файлы в {demo_dir}[/green]")
    
    # Запускаем оценку
    demo_files = [str(demo_dir / "demo_agent.py"), str(demo_dir / "description.txt")]
    
    try:
        workflow = create_workflow_from_env()
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("🔄 Демонстрационная оценка...", total=None)
            
            result = await workflow.run_assessment(
                source_files=demo_files,
                agent_name="DemoAgent"
            )
            
            progress.update(task, completed=True)
            
            if result["success"]:
                console.print("\n[green]✅ Демонстрация завершена успешно![/green]")
                await _display_assessment_result(result)
            else:
                console.print(f"\n[red]❌ Ошибка демонстрации: {result.get('error')}[/red]")
                
    except Exception as e:
        console.print(f"\n[red]❌ Ошибка демонстрации: {e}[/red]")
    
    # Очищаем тестовые файлы
    try:
        import shutil
        shutil.rmtree(demo_dir)
        console.print(f"[dim]🗑️ Очищены временные файлы[/dim]")
    except:
        pass


# ===============================
# Вспомогательные функции
# ===============================

async def _display_assessment_result(result: dict, output_file: Optional[str] = None):
    """Отображение результата оценки"""
    assessment = result.get("final_assessment")
    if not assessment:
        console.print("[red]❌ Отсутствуют данные оценки[/red]")
        return
    
    # Основная информация
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        f"[bold green]✅ Оценка завершена[/bold green]\n"
        f"ID: {result['assessment_id']}\n"
        f"Время: {result.get('processing_time', 0):.1f}с",
        title="Результат оценки",
        border_style="green"
    ))
    
    # Информация об агенте
    agent_info = assessment.get("agent_profile", {})
    console.print(f"\n[bold blue]🤖 Агент: {agent_info.get('name', 'Unknown')}[/bold blue]")
    console.print(f"Тип: {agent_info.get('agent_type', 'unknown')}")
    console.print(f"Описание: {agent_info.get('description', 'Не указано')[:100]}...")
    
    # Общий результат
    overall_score = assessment.get("overall_risk_score", 0)
    overall_level = assessment.get("overall_risk_level", "unknown")
    
    level_colors = {"low": "green", "medium": "yellow", "high": "red"}
    level_color = level_colors.get(overall_level, "white")
    
    console.print(f"\n[bold]📊 Общий риск: [{level_color}]{overall_level.upper()}[/{level_color}] ({overall_score}/25)[/bold]")
    
    # Таблица рисков
    risk_evaluations = assessment.get("risk_evaluations", {})
    if risk_evaluations:
        table = Table(title="Детализация по типам рисков")
        table.add_column("Тип риска", style="bold")
        table.add_column("Балл", justify="center")
        table.add_column("Уровень", justify="center")
        table.add_column("Вероятность", justify="center")
        table.add_column("Воздействие", justify="center")
        
        risk_names = {
            "ethical": "Этические",
            "stability": "Стабильность", 
            "security": "Безопасность",
            "autonomy": "Автономность",
            "regulatory": "Регуляторные",
            "social": "Социальные"
        }
        
        for risk_type, evaluation in risk_evaluations.items():
            risk_name = risk_names.get(risk_type, risk_type)
            level = evaluation.get("risk_level", "unknown")
            level_color = level_colors.get(level, "white")
            
            table.add_row(
                risk_name,
                str(evaluation.get("total_score", 0)),
                f"[{level_color}]{level}[/{level_color}]",
                str(evaluation.get("probability_score", 0)),
                str(evaluation.get("impact_score", 0))
            )
        
        console.print(table)
    
    # Рекомендации
    recommendations = assessment.get("priority_recommendations", [])
    if recommendations:
        console.print("\n[bold blue]💡 Приоритетные рекомендации:[/bold blue]")
        for i, rec in enumerate(recommendations[:5], 1):
            console.print(f"  {i}. {rec}")
    
    # Сохранение в файл
    if output_file:
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, default=str)
            
            console.print(f"\n[green]💾 Результат сохранен в {output_file}[/green]")
        except Exception as e:
            console.print(f"\n[red]❌ Ошибка сохранения: {e}[/red]")


async def _display_saved_assessment(assessment_data: dict, output_file: Optional[str] = None):
    """Отображение сохраненной оценки"""
    assessment = assessment_data.get("assessment")
    evaluations = assessment_data.get("evaluations", [])
    
    if not assessment:
        console.print("[red]❌ Данные оценки повреждены[/red]")
        return
    
    # Основная информация
    console.print(Panel.fit(
        f"[bold blue]📋 Сохраненная оценка[/bold blue]\n"
        f"ID: {assessment.id}\n"
        f"Дата: {assessment.assessment_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Общий риск: {assessment.overall_risk_level} ({assessment.overall_risk_score}/25)",
        title="Детали оценки",
        border_style="blue"
    ))
    
    # Таблица рисков из БД
    if evaluations:
        table = Table(title="Сохраненные оценки рисков")
        table.add_column("Тип риска", style="bold")
        table.add_column("Балл", justify="center")
        table.add_column("Уровень", justify="center")
        table.add_column("Агент-оценщик", style="dim")
        
        level_colors = {"low": "green", "medium": "yellow", "high": "red"}
        
        for eval_record in evaluations:
            level_color = level_colors.get(eval_record.risk_level, "white")
            table.add_row(
                eval_record.risk_type,
                str(eval_record.total_score),
                f"[{level_color}]{eval_record.risk_level}[/{level_color}]",
                eval_record.evaluator_agent
            )
        
        console.print(table)
    
    # Рекомендации
    if assessment.priority_recommendations:
        console.print("\n[bold blue]💡 Рекомендации:[/bold blue]")
        for i, rec in enumerate(assessment.priority_recommendations, 1):
            console.print(f"  {i}. {rec}")
    
    # Сохранение в файл
    if output_file:
        try:
            output_data = {
                "assessment": {
                    "id": assessment.id,
                    "timestamp": assessment.assessment_timestamp.isoformat(),
                    "overall_risk_score": assessment.overall_risk_score,
                    "overall_risk_level": assessment.overall_risk_level,
                    "recommendations": assessment.priority_recommendations
                },
                "evaluations": [
                    {
                        "risk_type": e.risk_type,
                        "total_score": e.total_score,
                        "risk_level": e.risk_level,
                        "reasoning": e.probability_reasoning + " " + e.impact_reasoning
                    }
                    for e in evaluations
                ]
            }
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            console.print(f"\n[green]💾 Оценка сохранена в {output_file}[/green]")
        except Exception as e:
            console.print(f"\n[red]❌ Ошибка сохранения: {e}[/red]")


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
    # Запускаем асинхронный CLI
    import asyncio
    
    # Патчим click для поддержки async
    def async_command(f):
        f = asyncio.coroutine(f)
        def wrapper(*args, **kwargs):
            return asyncio.run(f(*args, **kwargs))
        return wrapper
    
    # Применяем патч к командам
    for command in [assess, show, list_assessments, status, demo]:
        command.callback = async_command(command.callback)
    
    main()