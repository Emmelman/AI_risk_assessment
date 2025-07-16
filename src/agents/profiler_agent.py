# src/agents/profiler_agent.py
"""
Профайлер-агент для сбора и анализа данных об ИИ-агенте

ОБНОВЛЕНО: Убраны LLM параметры, используется центральный конфигуратор
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .base_agent import AnalysisAgent, AgentConfig
from ..models.risk_models import AgentProfile, AgentTaskResult, ProcessingStatus, AgentType, AutonomyLevel, DataSensitivity
from ..tools.document_parser import create_document_parser, parse_agent_documents
from ..tools.code_analyzer import create_code_analyzer, analyze_agent_codebase
from ..tools.prompt_analyzer import create_prompt_analyzer, analyze_agent_prompts
from ..utils.logger import LogContext


class ProfilerAgent(AnalysisAgent):
    """
    Агент-профайлер для сбора данных об ИИ-агенте
    
    Функции:
    1. Парсинг документации (Word, Excel, PDF)
    2. Анализ кодовой базы (Python, JavaScript, Java)
    3. Извлечение промптов и инструкций
    4. Определение технических характеристик
    5. Создание профиля агента для оценки рисков
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Инициализируем инструменты анализа
        self.document_parser = create_document_parser()
        self.code_analyzer = create_code_analyzer()
        self.prompt_analyzer = create_prompt_analyzer()
    
    def get_system_prompt(self) -> str:
        """Системный промпт для профайлера"""
        return """Ты - эксперт-аналитик по профилированию ИИ-агентов для оценки операционных рисков.

Твоя задача: создавать детальные профили ИИ-агентов на основе доступной информации.

ПРИНЦИПЫ АНАЛИЗА:
1. Тщательно анализируй все предоставленные данные
2. Определяй назначение и возможности агента
3. Выявляй технические характеристики
4. Оценивай уровень автономности
5. Классифицируй доступ к данным
6. Извлекай ключевые промпты и ограничения

ФОРМАТ ВЫВОДА: Структурированный JSON с полным профилем агента

ОБЯЗАТЕЛЬНЫЕ ПОЛЯ:
- name: название агента
- agent_type: тип агента (chatbot, assistant, trader, scorer, analyzer, generator)
- autonomy_level: уровень автономности (supervised, semi_autonomous, autonomous)
- data_access: типы данных (public, internal, confidential, critical)
- target_audience: целевая аудитория
- llm_model: используемая модель
- system_prompts: системные промпты
- guardrails: ограничения безопасности

Будь точным и объективным в анализе."""
    
    async def process(
        self, 
        input_data: Dict[str, Any], 
        assessment_id: str = "unknown"
    ) -> AgentTaskResult:
        """
        Основная обработка: профилирование агента
        
        Args:
            input_data: Данные для профилирования (пути к файлам/папкам)
            assessment_id: ID оценки
            
        Returns:
            Результат профилирования с AgentProfile
        """
        start_time = datetime.now()
        
        try:
            # Извлекаем пути для анализа
            file_paths = input_data.get("file_paths", [])
            agent_name = input_data.get("agent_name", "Unknown Agent")
            
            if not file_paths:
                raise ValueError("Не предоставлены файлы для анализа")
            
            # Выполняем сбор данных из всех источников
            collected_data = await self._collect_agent_data(file_paths, assessment_id)
            
            # Создаем профиль агента на основе собранных данных
            agent_profile = await self._create_agent_profile(
                collected_data, 
                agent_name, 
                assessment_id
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.update_stats(execution_time, True)
            
            return AgentTaskResult(
                agent_name=self.name,
                status=ProcessingStatus.COMPLETED,
                result_data={"agent_profile": agent_profile},
                execution_time=execution_time,
                assessment_id=assessment_id
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.update_stats(execution_time, False)
            
            self.logger.error(f"Профилирование завершилось с ошибкой: {e}")
            
            return AgentTaskResult(
                agent_name=self.name,
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                execution_time=execution_time,
                assessment_id=assessment_id
            )
    
    async def _collect_agent_data(
        self, 
        file_paths: List[str], 
        assessment_id: str
    ) -> Dict[str, Any]:
        """Сбор данных из всех источников"""
        
        collected_data = {
            "documents": {},
            "code_analysis": {},
            "prompts": {},
            "file_structure": []
        }
        
        for file_path in file_paths:
            path = Path(file_path)
            
            try:
                if path.is_file():
                    await self._process_single_file(path, collected_data, assessment_id)
                elif path.is_dir():
                    await self._process_directory(path, collected_data, assessment_id)
                else:
                    self.logger.warning(f"Путь не найден: {file_path}")
                    
            except Exception as e:
                self.logger.error(f"Ошибка обработки {file_path}: {e}")
                continue
        
        return collected_data
    
    async def _process_single_file(
        self, 
        file_path: Path, 
        collected_data: Dict[str, Any], 
        assessment_id: str
    ):
        """Обработка одного файла"""
        
        file_extension = file_path.suffix.lower()
        
        # Документы (Word, Excel, PDF)
        if file_extension in ['.docx', '.xlsx', '.pdf', '.txt', '.md']:
            try:
                doc_data = await parse_agent_documents([str(file_path)])
                collected_data["documents"][str(file_path)] = doc_data
            except Exception as e:
                self.logger.error(f"Ошибка парсинга документа {file_path}: {e}")
        
        # Код (Python, JavaScript, Java, JSON, YAML)
        elif file_extension in ['.py', '.js', '.java', '.json', '.yaml', '.yml']:
            try:
                code_data = await analyze_agent_codebase([str(file_path)])
                collected_data["code_analysis"][str(file_path)] = code_data
            except Exception as e:
                self.logger.error(f"Ошибка анализа кода {file_path}: {e}")
        
        # Промпты и конфигурации
        if file_extension in ['.txt', '.md', '.json', '.yaml', '.yml'] or 'prompt' in file_path.name.lower():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                prompt_data = await analyze_agent_prompts([content])
                collected_data["prompts"][str(file_path)] = prompt_data
            except Exception as e:
                self.logger.error(f"Ошибка анализа промптов {file_path}: {e}")
        
        # Добавляем информацию о структуре файлов
        collected_data["file_structure"].append({
            "path": str(file_path),
            "name": file_path.name,
            "extension": file_extension,
            "size": file_path.stat().st_size if file_path.exists() else 0
        })
    
    async def _process_directory(
        self, 
        dir_path: Path, 
        collected_data: Dict[str, Any], 
        assessment_id: str
    ):
        """Обработка директории рекурсивно"""
        
        for file_path in dir_path.rglob('*'):
            if file_path.is_file():
                await self._process_single_file(file_path, collected_data, assessment_id)
    
    async def _create_agent_profile(
        self, 
        collected_data: Dict[str, Any], 
        agent_name: str, 
        assessment_id: str
    ) -> AgentProfile:
        """Создание профиля агента на основе собранных данных"""
        
        # Формируем сводку всех данных для анализа
        data_summary = self._prepare_data_summary(collected_data)
        
        # Анализируем с помощью LLM
        profile_prompt = f"""Создай детальный профиль ИИ-агента '{agent_name}' на основе анализа файлов.

ДОСТУПНЫЕ ДАННЫЕ:
{data_summary}

Создай структурированный профиль с классификацией по всем параметрам."""
        
        profile_data = await self.call_llm_structured(
            data_to_analyze=data_summary,
            extraction_prompt=profile_prompt,
            assessment_id=assessment_id,
            expected_format="JSON"
        )
        
        # Преобразуем в объект AgentProfile
        return self._parse_profile_data(profile_data, agent_name, collected_data)
    
    def _prepare_data_summary(self, collected_data: Dict[str, Any]) -> str:
        """Подготовка сводки данных для анализа"""
        
        summary_parts = []
        
        # Информация о файлах
        file_count = len(collected_data["file_structure"])
        file_types = {}
        for file_info in collected_data["file_structure"]:
            ext = file_info["extension"]
            file_types[ext] = file_types.get(ext, 0) + 1
        
        summary_parts.append(f"СТРУКТУРА ФАЙЛОВ ({file_count} файлов):")
        for ext, count in file_types.items():
            summary_parts.append(f"  {ext}: {count} файлов")
        
        # Документы
        if collected_data["documents"]:
            summary_parts.append(f"\nДОКУМЕНТЫ ({len(collected_data['documents'])} файлов):")
            for path, doc_data in collected_data["documents"].items():
                summary_parts.append(f"  {Path(path).name}: {str(doc_data)[:200]}...")
        
        # Анализ кода
        if collected_data["code_analysis"]:
            summary_parts.append(f"\nАНАЛИЗ КОДА ({len(collected_data['code_analysis'])} файлов):")
            for path, code_data in collected_data["code_analysis"].items():
                summary_parts.append(f"  {Path(path).name}: {str(code_data)[:200]}...")
        
        # Промпты
        if collected_data["prompts"]:
            summary_parts.append(f"\nПРОМПТЫ И КОНФИГУРАЦИИ ({len(collected_data['prompts'])} файлов):")
            for path, prompt_data in collected_data["prompts"].items():
                summary_parts.append(f"  {Path(path).name}: {str(prompt_data)[:200]}...")
        
        return "\n".join(summary_parts)
    
    def _parse_profile_data(
        self, 
        profile_data: Dict[str, Any], 
        agent_name: str, 
        collected_data: Dict[str, Any]
    ) -> AgentProfile:
        """Парсинг данных профиля в объект AgentProfile"""
        
        # Определяем тип агента
        agent_type_str = profile_data.get("agent_type", "assistant")
        try:
            agent_type = AgentType(agent_type_str)
        except ValueError:
            agent_type = AgentType.ASSISTANT
        
        # Определяем уровень автономности
        autonomy_str = profile_data.get("autonomy_level", "supervised")
        try:
            autonomy_level = AutonomyLevel(autonomy_str)
        except ValueError:
            autonomy_level = AutonomyLevel.SUPERVISED
        
        # Определяем уровни доступа к данным
        data_access_list = profile_data.get("data_access", ["public"])
        data_sensitivity = []
        for access in data_access_list:
            try:
                data_sensitivity.append(DataSensitivity(access))
            except ValueError:
                continue
        
        if not data_sensitivity:
            data_sensitivity = [DataSensitivity.PUBLIC]
        
        return AgentProfile(
            name=profile_data.get("name", agent_name),
            agent_type=agent_type,
            autonomy_level=autonomy_level,
            data_sensitivity=data_sensitivity,
            target_audience=profile_data.get("target_audience", ["Общие пользователи"]),
            capabilities=profile_data.get("capabilities", []),
            limitations=profile_data.get("limitations", []),
            llm_model=profile_data.get("llm_model", "unknown"),
            system_prompts=profile_data.get("system_prompts", []),
            guardrails=profile_data.get("guardrails", []),
            data_access=profile_data.get("data_access", ["public"]),
            integration_points=profile_data.get("integration_points", []),
            deployment_environment=profile_data.get("deployment_environment", "unknown"),
            version=profile_data.get("version", "1.0"),
            created_at=datetime.now(),
            additional_metadata=collected_data
        )


# ===============================
# Функции для LangGraph интеграции
# ===============================

def create_profiler_node_function(profiler_agent: ProfilerAgent):
    """
    Создание функции узла профайлера для LangGraph
    
    Args:
        profiler_agent: Экземпляр профайлер-агента
        
    Returns:
        Функция узла для LangGraph workflow
    """
    
    async def profiler_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Узел профилирования в LangGraph workflow"""
        
        # Получаем данные из состояния
        input_data = {
            "source_files": state.get("source_files", []),
            "agent_name": state.get("agent_name", "Unknown Agent")
        }
        assessment_id = state.get("assessment_id", "unknown")
        
        # Выполняем профилирование
        result = await profiler_agent.run(input_data, assessment_id)
        
        # Обновляем состояние
        updated_state = state.copy()
        
        if result.status == ProcessingStatus.COMPLETED:
            agent_profile_data = result.result_data["agent_profile"]
            updated_state["agent_profile"] = agent_profile_data
            updated_state["current_step"] = "evaluation_preparation"
        else:
            updated_state["current_step"] = "error"
            updated_state["error_message"] = result.error_message
        
        return updated_state
    
    return profiler_node


# ===============================
# Фабрики (ОБНОВЛЕННЫЕ)
# ===============================

def create_profiler_agent(
    max_retries: int = 3,
    timeout_seconds: int = 1800
) -> ProfilerAgent:
    """
    Создание профайлер-агента (новая версия без LLM параметров)
    
    Args:
        max_retries: Максимум повторов
        timeout_seconds: Тайм-аут в секундах (увеличен для анализа больших объемов)
        
    Returns:
        Настроенный профайлер-агент
    """
    config = AgentConfig(
        name="profiler_agent",
        description="Агент для профилирования ИИ-агентов и сбора данных для оценки рисков",
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        use_risk_analysis_client=False  # Профайлер использует стандартный клиент
    )
    
    return ProfilerAgent(config)


def create_profiler_from_env() -> ProfilerAgent:
    """Создание профайлер-агента из переменных окружения"""
    import os
    
    return create_profiler_agent(
        max_retries=int(os.getenv("MAX_RETRY_COUNT", "3")),
        timeout_seconds=1800  # Фиксированный большой тайм-аут для профайлера
    )


# Legacy функция для обратной совместимости (DEPRECATED)
def create_profiler_agent_legacy(
    llm_base_url: str = "http://127.0.0.1:1234",
    llm_model: str = "qwen3-4b",
    temperature: float = 0.1
) -> ProfilerAgent:
    """
    DEPRECATED: Создание профайлер-агента (старая версия)
    Используйте create_profiler_agent() без LLM параметров
    """
    import warnings
    from ..utils.llm_client import LLMConfig
    
    warnings.warn(
        "create_profiler_agent_legacy deprecated. Use create_profiler_agent() without LLM params.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Создаем переопределение для legacy кода
    llm_override = LLMConfig(
        base_url=llm_base_url,
        model=llm_model,
        temperature=temperature,
        timeout=1800
    )
    
    config = AgentConfig(
        name="profiler_agent",
        description="Агент для профилирования ИИ-агентов и сбора данных для оценки рисков",
        max_retries=3,
        timeout_seconds=1800,
        use_risk_analysis_client=False,
        llm_override=llm_override
    )
    
    return ProfilerAgent(config)
def create_profiler_agent_legacy(**kwargs) -> ProfilerAgent:
    """
    DEPRECATED: Legacy функция создания профайлера
    
    Используйте create_profiler_agent() без параметров.
    """
    import warnings
    warnings.warn(
        "create_profiler_agent_legacy deprecated. Use create_profiler_agent() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return create_profiler_agent()

# Экспорт
__all__ = [
    "ProfilerAgent",
    "create_profiler_agent",
    "create_profiler_from_env",
    "create_profiler_node_function",
    # Legacy exports (deprecated)
    "create_profiler_agent_legacy"
]