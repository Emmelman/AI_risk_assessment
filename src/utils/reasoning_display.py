# src/utils/reasoning_display.py
"""
Модуль для отображения рассуждений всех агентов
"""

import os
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def show_profiler_reasoning(agent_name: str, stage: str, content: str):
    """
    Показывает рассуждения профайлера в красивом формате
    
    Args:
        agent_name: Имя агента
        stage: Этап работы
        content: Содержание рассуждений
    """
    
    if os.getenv('SHOW_AGENT_REASONING') != 'true':
        return
        
    console = Console()
    
    stage_emojis = {
        "data_collection": "📊",
        "data_analysis": "🔍", 
        "llm_call": "🤖",
        "profile_creation": "🎯",
        "result": "✅",
        "error": "❌"
    }
    
    emoji = stage_emojis.get(stage, "🧠")
    title = f"{emoji} Рассуждения: {agent_name} - {stage.upper()}"
    
    console.print(Panel(
        content.strip(),
        title=title,
        border_style="cyan",
        padding=(1, 2)
    ))


def show_evaluator_reasoning(agent_name: str, risk_type: str, stage: str, content: str):
    """
    Показывает рассуждения агентов-оценщиков
    
    Args:
        agent_name: Имя агента
        risk_type: Тип риска
        stage: Этап оценки
        content: Содержание рассуждений
    """
    
    if os.getenv('SHOW_AGENT_REASONING') != 'true':
        return
        
    console = Console()
    
    title = f"🧠 Рассуждения: {agent_name} - {risk_type.upper()}"
    
    console.print(Panel(
        content.strip(),
        title=title,
        border_style="blue",
        padding=(1, 2)
    ))


def show_critic_reasoning(critic_name: str, evaluation_type: str, content: str):
    """
    Показывает рассуждения критика
    
    Args:
        critic_name: Имя критика
        evaluation_type: Тип оцениваемого риска
        content: Содержание рассуждений
    """
    
    if os.getenv('SHOW_CRITIC_REASONING') != 'true':
        return
        
    console = Console()
    
    title = f"👨‍⚖️ Критик: {critic_name} - {evaluation_type.upper()}"
    
    console.print(Panel(
        content.strip(),
        title=title,
        border_style="red",
        padding=(1, 2)
    ))


def enable_all_reasoning():
    """Включает отображение всех типов рассуждений"""
    os.environ["SHOW_AGENT_REASONING"] = "true"
    os.environ["SHOW_CRITIC_REASONING"] = "true"
    print("🧠 Рассуждения агентов включены")


def disable_all_reasoning():
    """Отключает отображение всех типов рассуждений"""
    os.environ["SHOW_AGENT_REASONING"] = "false"
    os.environ["SHOW_CRITIC_REASONING"] = "false"
    print("🔇 Рассуждения агентов отключены")