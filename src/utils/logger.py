# src/utils/logger.py
"""
Система логирования для системы оценки рисков ИИ-агентов
Использует loguru для удобного и гибкого логирования
"""

import sys
import os
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from loguru import Logger


class RiskAssessmentLogger:
    """Настроенный логгер для системы оценки рисков"""
    
    def __init__(
        self,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        enable_console: bool = True,
        enable_rich: bool = True
    ):
        self.console = Console() if enable_rich else None
        
        # Удаляем стандартный обработчик loguru
        logger.remove()
        
        # Настройка консольного вывода
        if enable_console:
            if enable_rich and self.console:
                # Rich обработчик для красивого консольного вывода
                logger.add(
                    self._rich_sink,
                    level=log_level,
                    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                    colorize=True
                )
            else:
                # Стандартный консольный вывод
                logger.add(
                    sys.stderr,
                    level=log_level,
                    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                    colorize=True
                )
        
        # Настройка файлового логирования
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                log_file,
                level=log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {extra[assessment_id]} | {extra[agent_name]} | {message}",
                rotation="50 MB",
                retention="30 days",
                compression="zip",
                enqueue=True,  # Асинхронная запись
                backtrace=True,
                diagnose=True
            )
        
        # Контекст по умолчанию
        self.default_context = {
            "assessment_id": "unknown",
            "agent_name": "system"
        }
        
        self.logger = logger.bind(**self.default_context)
    
    def _rich_sink(self, message):
        """Кастомный sink для Rich вывода"""
        record = message.record
        
        # Форматирование для Rich Console
        timestamp = record["time"].strftime("%Y-%m-%d %H:%M:%S")
        level = record["level"].name
        location = f"{record['name']}:{record['function']}:{record['line']}"
        
        # Цветовая схема по уровням
        level_colors = {
            "DEBUG": "dim blue",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold red"
        }
        
        color = level_colors.get(level, "white")
        
        # Дополнительная информация из extra
        extra_info = ""
        if "assessment_id" in record["extra"]:
            assessment_id = record["extra"]["assessment_id"]
            if assessment_id != "unknown":
                extra_info += f"[dim]({assessment_id[:8]})[/dim] "
        
        if "agent_name" in record["extra"]:
            agent_name = record["extra"]["agent_name"]
            if agent_name != "system":
                extra_info += f"[cyan]{agent_name}[/cyan] "
        
        # Вывод в консоль
        self.console.print(
            f"[dim]{timestamp}[/dim] [{color}]{level: <8}[/{color}] "
            f"{extra_info}[dim]{location}[/dim] - {record['message']}"
        )
    
    def bind_context(self, assessment_id: Optional[str] = None, agent_name: Optional[str] = None):
        """Привязка контекста к логгеру"""
        context = self.default_context.copy()
        
        if assessment_id:
            context["assessment_id"] = assessment_id
        if agent_name:
            context["agent_name"] = agent_name
            
        return logger.bind(**context)
    
    def log_agent_start(self, agent_name: str, task_type: str, assessment_id: str):
        """Логирование старта работы агента"""
        bound_logger = self.bind_context(assessment_id, agent_name)
        bound_logger.info(f"🚀 Запуск агента для задачи: {task_type}")
    
    def log_agent_success(self, agent_name: str, task_type: str, assessment_id: str, execution_time: float):
        """Логирование успешного завершения работы агента"""
        bound_logger = self.bind_context(assessment_id, agent_name)
        bound_logger.info(f"✅ Агент завершил задачу: {task_type} за {execution_time:.2f}с")
    
    def log_agent_error(self, agent_name: str, task_type: str, assessment_id: str, error: Exception):
        """Логирование ошибки агента"""
        bound_logger = self.bind_context(assessment_id, agent_name)
        bound_logger.error(f"❌ Ошибка агента при выполнении {task_type}: {str(error)}")
    
    def log_agent_retry(self, agent_name: str, task_type: str, assessment_id: str, attempt: int):
        """Логирование повторной попытки агента"""
        bound_logger = self.bind_context(assessment_id, agent_name)
        bound_logger.warning(f"🔄 Повторная попытка агента: {task_type} (попытка {attempt})")
    
    def log_risk_evaluation(
        self, 
        agent_name: str, 
        assessment_id: str, 
        risk_type: str, 
        score: int, 
        level: str
    ):
        """Логирование результата оценки риска"""
        bound_logger = self.bind_context(assessment_id, agent_name)
        bound_logger.info(f"📊 Оценка риска {risk_type}: {score} баллов ({level})")
    
    def log_critic_feedback(
        self, 
        assessment_id: str, 
        risk_type: str, 
        quality_score: float, 
        is_acceptable: bool
    ):
        """Логирование обратной связи от критика"""
        bound_logger = self.bind_context(assessment_id, "critic")
        status = "✅ принято" if is_acceptable else "❌ отклонено"
        bound_logger.info(f"🔍 Критика {risk_type}: {quality_score:.1f}/10 - {status}")
    
    def log_workflow_step(self, assessment_id: str, step_name: str, details: Optional[str] = None):
        """Логирование шага workflow"""
        bound_logger = self.bind_context(assessment_id, "orchestrator")
        message = f"⚙️ Workflow шаг: {step_name}"
        if details:
            message += f" - {details}"
        bound_logger.info(message)
    
    def log_llm_request(self, agent_name: str, assessment_id: str, model: str, tokens: int):
        """Логирование запроса к LLM"""
        bound_logger = self.bind_context(assessment_id, agent_name)
        bound_logger.debug(f"🤖 LLM запрос к {model}: {tokens} токенов")
    
    def log_document_parsing(self, assessment_id: str, file_path: str, file_type: str, success: bool):
        """Логирование парсинга документов"""
        bound_logger = self.bind_context(assessment_id, "profiler")
        status = "✅" if success else "❌"
        bound_logger.info(f"📄 Парсинг {file_type}: {file_path} {status}")
    
    def log_database_operation(self, operation: str, table: str, success: bool, details: Optional[str] = None):
        """Логирование операций с базой данных"""
        bound_logger = self.bind_context()
        status = "✅" if success else "❌"
        message = f"💾 БД {operation} в {table} {status}"
        if details:
            message += f" - {details}"
        bound_logger.debug(message)
    
    def log_performance_metrics(self, assessment_id: str, metrics: Dict[str, Any]):
        """Логирование метрик производительности"""
        bound_logger = self.bind_context(assessment_id, "system")
        
        total_time = metrics.get("total_processing_time", 0)
        token_count = metrics.get("total_tokens", 0)
        agent_count = metrics.get("agents_used", 0)
        
        bound_logger.info(
            f"📈 Метрики оценки: {total_time:.2f}с, "
            f"{token_count} токенов, {agent_count} агентов"
        )
    
    def get_logger(self) -> "Logger":
        """Получение настроенного логгера"""
        return self.logger


# ===============================
# Декораторы для логирования
# ===============================

def log_agent_execution(agent_name: str, task_type: str):
    """
    Декоратор для автоматического логирования выполнения методов агентов
    
    Usage:
        @log_agent_execution("profiler", "profile_agent")
        async def run_profiler(self, ...):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            assessment_id = kwargs.get("assessment_id", "unknown")
            
            # Получаем логгер
            risk_logger = get_logger()
            risk_logger.log_agent_start(agent_name, task_type, assessment_id)
            
            try:
                result = await func(*args, **kwargs)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                risk_logger.log_agent_success(agent_name, task_type, assessment_id, execution_time)
                
                return result
                
            except Exception as e:
                risk_logger.log_agent_error(agent_name, task_type, assessment_id, e)
                raise
        
        return wrapper
    return decorator


def log_llm_call(agent_name: str):
    """
    Декоратор для логирования вызовов LLM
    
    Usage:
        @log_llm_call("evaluator_ethical")
        async def call_llm(self, ...):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            assessment_id = kwargs.get("assessment_id", "unknown")
            
            try:
                result = await func(*args, **kwargs)
                
                # Логируем успешный вызов
                risk_logger = get_logger()
                model = getattr(result, "model", "unknown")
                tokens = getattr(result, "usage", {}).get("total_tokens", 0)
                
                risk_logger.log_llm_request(agent_name, assessment_id, model, tokens)
                
                return result
                
            except Exception as e:
                risk_logger = get_logger()
                bound_logger = risk_logger.bind_context(assessment_id, agent_name)
                bound_logger.error(f"🤖 Ошибка LLM вызова: {str(e)}")
                raise
        
        return wrapper
    return decorator


# ===============================
# Глобальный экземпляр логгера
# ===============================

_global_logger: Optional[RiskAssessmentLogger] = None


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_rich: bool = True
) -> RiskAssessmentLogger:
    """
    Настройка глобального логгера
    
    Args:
        log_level: Уровень логирования
        log_file: Путь к файлу логов
        enable_console: Включить консольный вывод
        enable_rich: Использовать Rich для красивого вывода
    """
    global _global_logger
    
    _global_logger = RiskAssessmentLogger(
        log_level=log_level,
        log_file=log_file,
        enable_console=enable_console,
        enable_rich=enable_rich
    )
    
    return _global_logger


def get_logger() -> RiskAssessmentLogger:
    """Получение глобального логгера"""
    global _global_logger
    
    if _global_logger is None:
        # Настройка по умолчанию из переменных окружения
        log_level = os.getenv("LOG_LEVEL", "INFO")
        log_file = os.getenv("LOG_FILE", "logs/ai_risk_assessment.log")
        
        _global_logger = setup_logging(
            log_level=log_level,
            log_file=log_file
        )
    
    return _global_logger


# ===============================
# Контекстный менеджер для логирования
# ===============================

class LogContext:
    """Контекстный менеджер для автоматического логирования блоков кода"""
    
    def __init__(
        self, 
        operation_name: str, 
        assessment_id: str, 
        agent_name: str = "system",
        log_success: bool = True,
        log_timing: bool = True
    ):
        self.operation_name = operation_name
        self.assessment_id = assessment_id
        self.agent_name = agent_name
        self.log_success = log_success
        self.log_timing = log_timing
        self.start_time = None
        self.logger = get_logger()
    
    def __enter__(self):
        self.start_time = datetime.now()
        bound_logger = self.logger.bind_context(self.assessment_id, self.agent_name)
        bound_logger.info(f"🔄 Начало: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = (datetime.now() - self.start_time).total_seconds()
        bound_logger = self.logger.bind_context(self.assessment_id, self.agent_name)
        
        if exc_type is None:
            if self.log_success:
                timing_info = f" за {execution_time:.2f}с" if self.log_timing else ""
                bound_logger.info(f"✅ Завершено: {self.operation_name}{timing_info}")
        else:
            bound_logger.error(f"❌ Ошибка в {self.operation_name}: {exc_val}")


# ===============================
# Специальные методы для LangGraph
# ===============================

class LangGraphLogger:
    """Специализированный логгер для LangGraph workflow"""
    
    def __init__(self, base_logger: RiskAssessmentLogger):
        self.logger = base_logger
    
    def log_graph_start(self, assessment_id: str, graph_name: str):
        """Логирование запуска LangGraph"""
        bound_logger = self.logger.bind_context(assessment_id, "langgraph")
        bound_logger.info(f"🔄 Запуск LangGraph: {graph_name}")
    
    def log_node_entry(self, assessment_id: str, node_name: str, state_keys: list):
        """Логирование входа в узел графа"""
        bound_logger = self.logger.bind_context(assessment_id, f"node_{node_name}")
        bound_logger.debug(f"➡️ Вход в узел: {node_name} | Состояние: {state_keys}")
    
    def log_node_exit(self, assessment_id: str, node_name: str, next_node: str, execution_time: float):
        """Логирование выхода из узла графа"""
        bound_logger = self.logger.bind_context(assessment_id, f"node_{node_name}")
        bound_logger.debug(f"⬅️ Выход из узла: {node_name} → {next_node} ({execution_time:.2f}с)")
    
    def log_conditional_edge(self, assessment_id: str, condition: str, chosen_path: str, reason: str):
        """Логирование условного перехода"""
        bound_logger = self.logger.bind_context(assessment_id, "langgraph")
        bound_logger.info(f"🔀 Условный переход: {condition} → {chosen_path} | {reason}")
    
    def log_state_update(self, assessment_id: str, node_name: str, updated_keys: list):
        """Логирование обновления состояния"""
        bound_logger = self.logger.bind_context(assessment_id, f"node_{node_name}")
        bound_logger.debug(f"📝 Обновление состояния: {updated_keys}")
    
    def log_graph_completion(self, assessment_id: str, total_time: float, nodes_visited: int):
        """Логирование завершения графа"""
        bound_logger = self.logger.bind_context(assessment_id, "langgraph")
        bound_logger.info(f"🏁 LangGraph завершен: {nodes_visited} узлов за {total_time:.2f}с")
    
    def log_retry_logic(self, assessment_id: str, node_name: str, retry_count: int, max_retries: int):
        """Логирование логики повторов"""
        bound_logger = self.logger.bind_context(assessment_id, f"node_{node_name}")
        bound_logger.warning(f"🔄 Повтор узла: попытка {retry_count}/{max_retries}")
    
    def log_quality_check(self, assessment_id: str, risk_type: str, quality_score: float, threshold: float):
        """Логирование проверки качества критиком"""
        bound_logger = self.logger.bind_context(assessment_id, "quality_check")
        status = "✅ пройдена" if quality_score >= threshold else "❌ не пройдена"
        bound_logger.info(f"🔍 Проверка качества {risk_type}: {quality_score:.1f}/{threshold} - {status}")
    
    def log_workflow_step(self, assessment_id: str, step_name: str, details: str = ""):
        """Логирование шага workflow"""
        bound_logger = self.logger.bind_context(assessment_id, "workflow")
        message = f"⚙️ Workflow шаг: {step_name}"
        if details:
            message += f" - {details}"

# Добавляем LangGraph логгер к основному
def get_langgraph_logger() -> LangGraphLogger:
    """Получение LangGraph логгера"""
    base_logger = get_logger()
    return LangGraphLogger(base_logger)


# ===============================
# Декораторы для LangGraph узлов
# ===============================

def log_graph_node(node_name: str):
    """
    Декоратор для логирования узлов LangGraph
    
    Usage:
        @log_graph_node("profiler_node")
        def profiler_node(state: WorkflowState) -> WorkflowState:
            ...
    """
    def decorator(func):
        async def wrapper(state, *args, **kwargs):
            assessment_id = getattr(state, 'assessment_id', 'unknown')
            start_time = datetime.now()
            
            graph_logger = get_langgraph_logger()
            
            # Логируем вход в узел
            state_keys = list(state.__dict__.keys()) if hasattr(state, '__dict__') else ['unknown']
            graph_logger.log_node_entry(assessment_id, node_name, state_keys)
            
            try:
                # Выполняем функцию узла
                if asyncio.iscoroutinefunction(func):
                    result = await func(state, *args, **kwargs)
                else:
                    result = func(state, *args, **kwargs)
                
                # Логируем успешное выполнение
                execution_time = (datetime.now() - start_time).total_seconds()
                next_node = getattr(result, 'current_step', 'unknown')
                graph_logger.log_node_exit(assessment_id, node_name, next_node, execution_time)
                
                return result
                
            except Exception as e:
                # Логируем ошибку
                base_logger = get_logger()
                bound_logger = base_logger.bind_context(assessment_id, f"node_{node_name}")
                bound_logger.error(f"❌ Ошибка в узле {node_name}: {str(e)}")
                raise
        
        return wrapper
    return decorator


def log_conditional_edge_func(edge_name: str):
    """
    Декоратор для логирования условных переходов LangGraph
    
    Usage:
        @log_conditional_edge_func("quality_check_router")
        def quality_check_router(state: WorkflowState) -> str:
            ...
    """
    def decorator(func):
        def wrapper(state, *args, **kwargs):
            assessment_id = getattr(state, 'assessment_id', 'unknown')
            
            # Выполняем функцию условного перехода
            chosen_path = func(state, *args, **kwargs)
            
            # Логируем выбранный путь
            graph_logger = get_langgraph_logger()
            
            # Пытаемся определить причину выбора пути
            reason = "условие выполнено"
            if hasattr(state, 'critic_results'):
                failed_checks = [
                    risk_type for risk_type, result in state.critic_results.items()
                    if result and not result.result_data.get('is_acceptable', True)
                ]
                if failed_checks:
                    reason = f"не прошли проверку: {', '.join(failed_checks)}"
            
            graph_logger.log_conditional_edge(assessment_id, edge_name, chosen_path, reason)
            
            return chosen_path
        
        return wrapper
    return decorator


# ===============================
# Функции инициализации для разных режимов
# ===============================

def setup_logging_for_development():
    """Настройка логирования для разработки"""
    return setup_logging(
        log_level="DEBUG",
        log_file="logs/ai_risk_assessment_dev.log",
        enable_console=True,
        enable_rich=True
    )


def setup_logging_for_production():
    """Настройка логирования для продакшена"""
    return setup_logging(
        log_level="INFO",
        log_file="logs/ai_risk_assessment_prod.log",
        enable_console=False,
        enable_rich=False
    )


def setup_logging_for_testing():
    """Настройка логирования для тестирования"""
    return setup_logging(
        log_level="WARNING",
        log_file=None,  # Только в консоль
        enable_console=True,
        enable_rich=False
    )


# Автоматическая настройка на основе переменных окружения
def auto_setup_logging():
    """Автоматическая настройка логирования"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return setup_logging_for_production()
    elif env == "testing":
        return setup_logging_for_testing()
    else:
        return setup_logging_for_development()


# Экспорт основных функций
__all__ = [
    "RiskAssessmentLogger",
    "LangGraphLogger", 
    "setup_logging",
    "get_logger",
    "get_langgraph_logger",
    "log_agent_execution",
    "log_llm_call",
    "log_graph_node",
    "log_conditional_edge_func",
    "LogContext",
    "auto_setup_logging"
]