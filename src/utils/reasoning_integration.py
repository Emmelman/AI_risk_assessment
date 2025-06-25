# src/utils/reasoning_integration.py
"""
ПРОСТАЯ интеграция рассуждений агентов в существующий код
Минимальные изменения - максимальная польза
"""

import json
from typing import Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def show_agent_reasoning(
    agent_name: str,
    risk_type: str,
    evaluation_result: Dict[str, Any],
    agent_profile_summary: str = ""
):
    """
    Простое отображение рассуждений агента
    Вызывается ПОСЛЕ получения результата от LLM
    """
    
    # Извлекаем данные из результата
    prob_score = evaluation_result.get('probability_score', 0)
    impact_score = evaluation_result.get('impact_score', 0)
    total_score = evaluation_result.get('total_score', 0)
    risk_level = evaluation_result.get('risk_level', 'unknown')
    
    prob_reasoning = evaluation_result.get('probability_reasoning', 'Нет обоснования')
    impact_reasoning = evaluation_result.get('impact_reasoning', 'Нет обоснования')
    key_factors = evaluation_result.get('key_factors', [])
    recommendations = evaluation_result.get('recommendations', [])
    confidence = evaluation_result.get('confidence_level', 0.0)
    
    # Цвет по уровню риска
    level_color = {
        'low': 'green',
        'medium': 'yellow',
        'high': 'red'
    }.get(risk_level, 'white')
    
    # Создаем текст с рассуждениями
    reasoning_text = Text()
    
    # Заголовок
    reasoning_text.append(f"🧠 {agent_name} анализирует: {risk_type}\n\n", style="bold blue")
    
    # Рассуждение о вероятности
    reasoning_text.append("💭 Анализ вероятности:\n", style="bold cyan")
    reasoning_text.append(f"{prob_reasoning}\n", style="white")
    reasoning_text.append(f"➜ Оценка: {prob_score}/5\n\n", style="cyan")
    
    # Рассуждение о тяжести
    reasoning_text.append("🎯 Анализ тяжести последствий:\n", style="bold cyan")
    reasoning_text.append(f"{impact_reasoning}\n", style="white")
    reasoning_text.append(f"➜ Оценка: {impact_score}/5\n\n", style="cyan")
    
    # Ключевые факторы
    if key_factors:
        reasoning_text.append("🔑 Ключевые факторы риска:\n", style="bold yellow")
        for factor in key_factors[:3]:  # Показываем топ-3
            reasoning_text.append(f"• {factor}\n", style="yellow")
        reasoning_text.append("\n")
    
    # Финальная оценка
    reasoning_text.append("🎯 ИТОГОВАЯ ОЦЕНКА:\n", style="bold")
    reasoning_text.append(f"Общий балл: {total_score}/25\n", style="white")
    reasoning_text.append(f"Уровень риска: ", style="white")
    reasoning_text.append(f"{risk_level.upper()}\n", style=f"bold {level_color}")
    reasoning_text.append(f"Уверенность: {confidence:.1%}\n\n", style="blue")
    
    # Топ-3 рекомендации
    if recommendations:
        reasoning_text.append("💡 Топ-3 рекомендации:\n", style="bold green")
        for i, rec in enumerate(recommendations[:3], 1):
            reasoning_text.append(f"{i}. {rec}\n", style="green")
    
    # Отображаем панель
    console.print(Panel(
        reasoning_text,
        title=f"🔍 Рассуждения агента",
        border_style=level_color,
        width=100
    ))
    
    console.print("─" * 100)


def show_critic_reasoning(
    risk_type: str,
    critic_result: Dict[str, Any]
):
    """Отображение рассуждений критика"""
    
    quality_score = critic_result.get('quality_score', 0)
    is_acceptable = critic_result.get('is_acceptable', False)
    critic_reasoning = critic_result.get('critic_reasoning', 'Нет обоснования')
    issues = critic_result.get('issues_found', [])
    suggestions = critic_result.get('improvement_suggestions', [])
    
    # Цвет по качеству
    color = "green" if is_acceptable else "red"
    status = "✅ ПРИНЯТО" if is_acceptable else "❌ ОТКЛОНЕНО"
    
    critic_text = Text()
    critic_text.append(f"🔍 Критический анализ: {risk_type}\n", style="bold")
    critic_text.append(f"📊 Качество: {quality_score:.1f}/10 - {status}\n\n", style=f"bold {color}")
    critic_text.append(f"💭 Обоснование критика:\n{critic_reasoning}\n", style="white")
    
    if issues:
        critic_text.append(f"\n⚠️ Выявленные проблемы:\n", style="yellow")
        for issue in issues[:3]:
            critic_text.append(f"• {issue}\n", style="yellow")
    
    if suggestions:
        critic_text.append(f"\n💡 Предложения по улучшению:\n", style="blue")
        for suggestion in suggestions[:3]:
            critic_text.append(f"• {suggestion}\n", style="blue")
    
    console.print(Panel(
        critic_text,
        title="🔍 Критический анализ",
        border_style=color,
        width=100
    ))


# ==========================================
# ПАТЧИ ДЛЯ ИНТЕГРАЦИИ В СУЩЕСТВУЮЩИЙ КОД
# ==========================================

def enable_reasoning_in_evaluators():
    """
    Включает отображение рассуждений в агентах-оценщиках
    НУЖНО ВЫЗВАТЬ ОДИН РАЗ в начале main.py
    """
    
    # Патчим метод evaluate_risk в базовом агенте
    from src.agents.base_agent import AnalysisAgent
    from src.utils.llm_client import RiskAnalysisLLMClient
    
    # Сохраняем оригинальный метод
    original_evaluate_risk = getattr(RiskAnalysisLLMClient, 'evaluate_risk', None)
    
    if original_evaluate_risk:
        async def patched_evaluate_risk(self, risk_type, agent_data, evaluation_criteria, assessment_id):
            """Патченый метод с отображением рассуждений"""
            
            # Вызываем оригинальный метод
            result = await original_evaluate_risk(self, risk_type, agent_data, evaluation_criteria, assessment_id)
            
            # Показываем рассуждения, если включены
            try:
                # Проверяем, включены ли рассуждения
                import os
                if os.getenv("SHOW_AGENT_REASONING", "true").lower() == "true":
                    show_agent_reasoning(
                        agent_name=getattr(self, '_current_agent_name', 'Unknown Agent'),
                        risk_type=risk_type,
                        evaluation_result=result,
                        agent_profile_summary=str(agent_data)[:200]
                    )
            except Exception as e:
                # Игнорируем ошибки отображения
                pass
            
            return result
        
        # Применяем патч
        RiskAnalysisLLMClient.evaluate_risk = patched_evaluate_risk


def enable_reasoning_in_critic():
    """Включает отображение рассуждений критика"""
    
    from src.agents.critic_agent import CriticAgent
    
    # Патчим метод analyze_evaluation_quality
    original_analyze = getattr(CriticAgent, 'analyze_evaluation_quality', None)
    
    if original_analyze:
        async def patched_analyze(self, original_evaluation, agent_data, quality_threshold, assessment_id):
            """Патченый метод критика с отображением рассуждений"""
            
            # Вызываем оригинальный метод
            result = await original_analyze(self, original_evaluation, agent_data, quality_threshold, assessment_id)
            
            # Показываем рассуждения критика
            try:
                import os
                if os.getenv("SHOW_CRITIC_REASONING", "true").lower() == "true":
                    risk_type = original_evaluation.get('risk_type', 'Unknown')
                    show_critic_reasoning(risk_type, result)
            except Exception as e:
                pass
            
            return result
        
        # Применяем патч
        CriticAgent.analyze_evaluation_quality = patched_analyze


# ==========================================
# ПРОСТАЯ ИНТЕГРАЦИЯ В ОДИН КЛИК
# ==========================================

def enable_all_reasoning():
    """
    ПРОСТОЕ включение всех рассуждений
    Добавить ОДНУ строчку в main.py:
    
    from src.utils.reasoning_integration import enable_all_reasoning
    enable_all_reasoning()
    """
    
    print("🧠 Включаем отображение рассуждений агентов...")
    
    try:
        enable_reasoning_in_evaluators()
        print("✅ Рассуждения оценщиков включены")
    except Exception as e:
        print(f"⚠️ Ошибка патча оценщиков: {e}")
    
    try:
        enable_reasoning_in_critic()
        print("✅ Рассуждения критика включены")
    except Exception as e:
        print(f"⚠️ Ошибка патча критика: {e}")
    
    print("🎉 Рассуждения агентов активированы!")


# ==========================================
# ПЕРЕМЕННЫЕ ОКРУЖЕНИЯ ДЛЯ УПРАВЛЕНИЯ
# ==========================================

def setup_reasoning_env():
    """Настройка переменных окружения для управления рассуждениями"""
    
    import os
    
    # Устанавливаем дефолтные значения
    if "SHOW_AGENT_REASONING" not in os.environ:
        os.environ["SHOW_AGENT_REASONING"] = "true"
    
    if "SHOW_CRITIC_REASONING" not in os.environ:
        os.environ["SHOW_CRITIC_REASONING"] = "true"
    
    print(f"🔧 Рассуждения агентов: {'ВКЛ' if os.getenv('SHOW_AGENT_REASONING') == 'true' else 'ВЫКЛ'}")
    print(f"🔧 Рассуждения критика: {'ВКЛ' if os.getenv('SHOW_CRITIC_REASONING') == 'true' else 'ВЫКЛ'}")


# ==========================================
# ЭКСПОРТ
# ==========================================

__all__ = [
    'show_agent_reasoning',
    'show_critic_reasoning', 
    'enable_all_reasoning',
    'setup_reasoning_env'
]