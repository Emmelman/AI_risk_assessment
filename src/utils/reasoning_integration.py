# src/utils/reasoning_integration.py - ПОЛНОСТЬЮ ИСПРАВЛЕННАЯ ВЕРСИЯ

# Глобальная переменная для отслеживания примененных патчей
applied_patches = set()

"""
НАДЕЖНАЯ интеграция рассуждений агентов с автоматическим определением сигнатур
"""

import json
import inspect
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
    """Простое отображение рассуждений агента"""
    
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
    
    # Выводим панель с рассуждениями
    console.print(Panel(
        reasoning_text,
        title=f"🧠 Рассуждения: {agent_name}",
        border_style=level_color,
        padding=(1, 2)
    ))


def show_critic_reasoning(risk_type: str, critic_result: Dict[str, Any]):
    """Отображение рассуждений критика"""
    
    quality_score = critic_result.get('quality_score', 0)
    is_acceptable = critic_result.get('is_acceptable', False)
    issues_found = critic_result.get('issues_found', [])
    suggestions = critic_result.get('improvement_suggestions', [])
    reasoning = critic_result.get('critic_reasoning', 'Нет рассуждений')
    
    # Цвет на основе приемлемости
    border_color = "green" if is_acceptable else "red"
    status_icon = "✅" if is_acceptable else "❌"
    
    reasoning_text = Text()
    reasoning_text.append(f"🔍 Критик анализирует: {risk_type}\n\n", style="bold blue")
    reasoning_text.append(f"📊 Качество оценки: {quality_score}/10\n", style="cyan")
    reasoning_text.append(f"{status_icon} Статус: {'ПРИНЯТО' if is_acceptable else 'ОТКЛОНЕНО'}\n\n", style=f"bold {'green' if is_acceptable else 'red'}")
    
    reasoning_text.append("🤔 Рассуждения критика:\n", style="bold yellow")
    reasoning_text.append(f"{reasoning}\n\n", style="white")
    
    if issues_found:
        reasoning_text.append("⚠️ Найденные проблемы:\n", style="bold red")
        for issue in issues_found:
            reasoning_text.append(f"• {issue}\n", style="red")
        reasoning_text.append("\n")
    
    if suggestions:
        reasoning_text.append("💡 Предложения по улучшению:\n", style="bold green")
        for suggestion in suggestions:
            reasoning_text.append(f"• {suggestion}\n", style="green")
    
    console.print(Panel(
        reasoning_text,
        title="🔍 Анализ критика",
        border_style=border_color,
        padding=(1, 2)
    ))


def enable_reasoning_in_evaluators():
    """
    НАДЕЖНОЕ включение отображения рассуждений в агентах-оценщиках
    Автоматически определяет сигнатуру метода и адаптируется к ней
    """
    
    try:
        from src.utils.llm_client import RiskAnalysisLLMClient
        
        # Сохраняем оригинальный метод
        original_evaluate_risk = getattr(RiskAnalysisLLMClient, 'evaluate_risk', None)
        
        if not original_evaluate_risk:
            print("⚠️ Метод evaluate_risk не найден в RiskAnalysisLLMClient")
            return
        
        # ИНСПЕКТИРУЕМ СИГНАТУРУ ОРИГИНАЛЬНОГО МЕТОДА
        sig = inspect.signature(original_evaluate_risk)
        param_names = list(sig.parameters.keys())
        
        print(f"🔍 Найдена сигнатура: {param_names}")
        
        # СОЗДАЕМ УНИВЕРСАЛЬНУЮ ОБЕРТКУ
        async def universal_patched_evaluate_risk(self, *args, **kwargs):
            """Универсальная патченая функция с автоматической адаптацией к сигнатуре"""
            
            try:
                # Вызываем оригинальный метод с точными параметрами
                result = await original_evaluate_risk(self, *args, **kwargs)
                
                # Показываем рассуждения, если включены
                try:
                    import os
                    if os.getenv("SHOW_AGENT_REASONING", "true").lower() == "true":
                        # Извлекаем risk_type из аргументов
                        risk_type = args[0] if args else kwargs.get('risk_type', 'Unknown')
                        
                        show_agent_reasoning(
                            agent_name=getattr(self, '_current_agent_name', 'Unknown Agent'),
                            risk_type=str(risk_type),
                            evaluation_result=result,
                            agent_profile_summary=""
                        )
                except Exception as e:
                    # Игнорируем ошибки отображения
                    pass
                
                return result
                
            except Exception as e:
                # Если что-то пошло не так, логируем и пробрасываем ошибку дальше
                print(f"⚠️ Ошибка в патченом методе evaluate_risk: {e}")
                raise
        
        # Применяем патч
        RiskAnalysisLLMClient.evaluate_risk = universal_patched_evaluate_risk
        print("✅ Патч для evaluate_risk успешно применен")
        
        # ДОПОЛНИТЕЛЬНО: патчим также critique_evaluation для критика
        original_critique_evaluation = getattr(RiskAnalysisLLMClient, 'critique_evaluation', None)
        
        if original_critique_evaluation:
            async def universal_patched_critique_evaluation(self, *args, **kwargs):
                """Универсальная патченая функция для critique_evaluation"""
                
                try:
                    # Вызываем оригинальный метод
                    result = await original_critique_evaluation(self, *args, **kwargs)
                    
                    # Показываем рассуждения критика, если включены
                    try:
                        import os
                        if os.getenv("SHOW_CRITIC_REASONING", "true").lower() == "true":
                            # Извлекаем risk_type из аргументов
                            risk_type = args[0] if args else kwargs.get('risk_type', 'Unknown')
                            
                            show_critic_reasoning(str(risk_type), result)
                    except Exception as e:
                        # Игнорируем ошибки отображения
                        pass
                    
                    return result
                    
                except Exception as e:
                    print(f"⚠️ Ошибка в патченом методе critique_evaluation: {e}")
                    raise
            
            # Применяем патч
            RiskAnalysisLLMClient.critique_evaluation = universal_patched_critique_evaluation
            print("✅ Патч для critique_evaluation успешно применен")
        
    except Exception as e:
        print(f"⚠️ Ошибка патча оценщиков: {e}")


def enable_reasoning_in_critic():
    """ИСПРАВЛЕННОЕ включение отображения рассуждений критика"""
    
    try:
        from src.agents.critic_agent import CriticAgent
        
        # Патчим приватный метод _critique_evaluation (реальный метод критика)
        original_critique = getattr(CriticAgent, '_critique_evaluation', None)
        
        if not original_critique:
            print("⚠️ Метод _critique_evaluation не найден в CriticAgent")
            return
        
        # Инспектируем сигнатуру
        sig = inspect.signature(original_critique)
        param_names = list(sig.parameters.keys())
        print(f"🔍 Найдена сигнатура критика: {param_names}")
        
        async def universal_patched_critique(self, *args, **kwargs):
            """Универсальная патченая функция критика"""
            
            try:
                # Вызываем оригинальный метод
                result = await original_critique(self, *args, **kwargs)
                
                # Показываем рассуждения критика
                try:
                    import os
                    if os.getenv("SHOW_CRITIC_REASONING", "true").lower() == "true":
                        # Извлекаем risk_type из kwargs или аргументов
                        risk_type = kwargs.get('risk_type', 'Unknown')
                        if hasattr(risk_type, 'value'):
                            risk_type = risk_type.value
                        
                        show_critic_reasoning(str(risk_type), result)
                except Exception as e:
                    # Игнорируем ошибки отображения
                    pass
                
                return result
                
            except Exception as e:
                print(f"⚠️ Ошибка в патченом методе _critique_evaluation: {e}")
                raise
        
        # Применяем патч
        CriticAgent._critique_evaluation = universal_patched_critique
        print("✅ Патч для критика успешно применен")
        
    except Exception as e:
        print(f"⚠️ Ошибка патча критика: {e}")


# ==========================================
# ПОЛНОСТЬЮ ИСПРАВЛЕННАЯ ПРОСТАЯ ИНТЕГРАЦИЯ В ОДИН КЛИК
# ==========================================

def enable_all_reasoning():
    """
    ПОЛНОСТЬЮ ИСПРАВЛЕННОЕ включение всех рассуждений
    Добавить ОДНУ строчку в main.py:
    
    from src.utils.reasoning_integration import enable_all_reasoning
    enable_all_reasoning()
    """
    
    print("🧠 Включаем отображение рассуждений агентов...")
    
    try:
        enable_reasoning_in_evaluators()
        print("✅ Рассуждения оценщиков включены")
    except Exception as e:
        print(f"⚠️ Не найден метод evaluate_risk для патча")
    
    try:
        enable_reasoning_in_critic()
        print("✅ Рассуждения критика включены") 
    except Exception as e:
        print(f"⚠️ Не найден метод analyze_evaluation_quality для патча")
    
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
# ДОПОЛНИТЕЛЬНЫЕ УТИЛИТЫ ДЛЯ ОТЛАДКИ
# ==========================================

def inspect_method_signatures():
    """Отладочная функция для проверки сигнатур методов"""
    
    try:
        from src.utils.llm_client import RiskAnalysisLLMClient
        from src.agents.critic_agent import CriticAgent
        
        print("🔍 ИНСПЕКЦИЯ СИГНАТУР МЕТОДОВ:")
        
        # Проверяем RiskAnalysisLLMClient
        evaluate_risk_method = getattr(RiskAnalysisLLMClient, 'evaluate_risk', None)
        if evaluate_risk_method:
            sig = inspect.signature(evaluate_risk_method)
            print(f"📋 RiskAnalysisLLMClient.evaluate_risk: {list(sig.parameters.keys())}")
        else:
            print("❌ RiskAnalysisLLMClient.evaluate_risk не найден")
        
        # Проверяем CriticAgent
        analyze_method = getattr(CriticAgent, 'analyze_evaluation_quality', None)
        if analyze_method:
            sig = inspect.signature(analyze_method)
            print(f"📋 CriticAgent.analyze_evaluation_quality: {list(sig.parameters.keys())}")
        else:
            print("❌ CriticAgent.analyze_evaluation_quality не найден")
            
    except Exception as e:
        print(f"❌ Ошибка при инспекции: {e}")


# ==========================================
# ЭКСПОРТ
# ==========================================

__all__ = [
    'show_agent_reasoning',
    'show_critic_reasoning', 
    'enable_all_reasoning',
    'setup_reasoning_env',
    'inspect_method_signatures'
]