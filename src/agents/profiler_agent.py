# src/agents/profiler_agent.py
"""
Профайлер-агент для сбора и анализа данных об ИИ-агенте
Собирает информацию из кодовой базы, документации, промптов и конфигураций
ОБНОВЛЕНО: Поэтапное создание профиля (6 этапов) для работы с ограниченными моделями
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
    
    ОБНОВЛЕНО: Поэтапное профилирование для решения проблем с большими промптами
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Инициализируем инструменты анализа
        self.document_parser = create_document_parser()
        self.code_analyzer = create_code_analyzer()
        self.prompt_analyzer = create_prompt_analyzer()
        
        # Состояние для поэтапного анализа
        self.analysis_state = {}
    
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
        assessment_id: str
    ) -> AgentTaskResult:
        """
        Основная обработка профайлинга агента
        
        Args:
            input_data: Содержит пути к файлам и папкам для анализа
                - source_files: List[str] - список файлов/папок
                - agent_name: Optional[str] - предварительное имя агента
            assessment_id: ID оценки
            
        Returns:
            Результат с профилем агента
        """
        start_time = datetime.now()
        
        try:
            with LogContext("profile_agent", assessment_id, self.name):
                # Извлекаем входные данные
                source_files = input_data.get("source_files", [])
                preliminary_name = input_data.get("agent_name", "Unknown_Agent")
                
                if not source_files:
                    raise ValueError("Не предоставлены файлы для анализа")
                
                # Собираем данные из всех источников
                collected_data = await self._collect_all_data(source_files, assessment_id)
                
                # НОВОЕ: Анализируем собранные данные ПОЭТАПНО
                agent_profile = await self._analyze_and_create_profile(
                    collected_data, preliminary_name, assessment_id
                )
                
                # Создаем результат
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                return AgentTaskResult(
                    agent_name=self.name,
                    task_type="profiling",
                    status=ProcessingStatus.COMPLETED,
                    result_data={
                        "agent_profile": agent_profile.dict(),
                        "collected_data_summary": self._create_data_summary(collected_data)
                    },
                    start_time=start_time,
                    end_time=end_time,
                    execution_time_seconds=execution_time
                )
                
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return AgentTaskResult(
                agent_name=self.name,
                task_type="profiling",
                status=ProcessingStatus.FAILED,
                result_data={},
                error_message=str(e),
                start_time=start_time,
                end_time=end_time,
                execution_time_seconds=execution_time
            )

    # ==========================================
    # НОВЫЙ ПОЭТАПНЫЙ МЕТОД (заменяет старый)
    # ==========================================

    async def _analyze_and_create_profile(
        self,
        collected_data: Dict[str, Any],
        preliminary_name: str,
        assessment_id: str
    ) -> AgentProfile:
        """
        ПОЭТАПНЫЙ анализ собранных данных и создание профиля агента
        
        ЭТАПЫ:
        1. Структурирование данных по категориям  
        2. Извлечение базовых полей
        3. Техническая архитектура  
        4. Операционная модель
        5. Безопасность и ограничения
        6. Генерация detailed_summary
        """
        
        bound_logger = self.logger.bind_context(assessment_id, self.name)
        bound_logger.info("🚀 Начинаем ПОЭТАПНОЕ создание профиля (6 этапов)")
        
        # Инициализируем состояние для хранения промежуточных результатов
        self.analysis_state = {
            "categorized_data": {},      # Этап 1
            "basic_profile": {},         # Этап 2  
            "technical_analysis": {},    # Этап 3
            "operational_analysis": {},  # Этап 4
            "security_analysis": {},     # Этап 5
            "detailed_sections": {}      # Этап 6
        }
        
        try:
            # ЭТАП 1: Структурирование данных по категориям
            bound_logger.info("📁 ЭТАП 1: Структурирование данных по категориям")
            self.analysis_state["categorized_data"] = await self._stage1_categorize_data(
                collected_data, assessment_id
            )
            
            # ЭТАП 2: Извлечение базовых полей
            bound_logger.info("🔧 ЭТАП 2: Извлечение базовых полей")
            self.analysis_state["basic_profile"] = await self._stage2_extract_basic_fields(
                self.analysis_state["categorized_data"], preliminary_name, assessment_id
            )
            
            # ЭТАП 3: Техническая архитектура
            bound_logger.info("⚙️ ЭТАП 3: Анализ технической архитектуры")
            self.analysis_state["technical_analysis"] = await self._stage3_technical_architecture(
                self.analysis_state["categorized_data"], assessment_id
            )
            
            # ЭТАП 4: Операционная модель
            bound_logger.info("📊 ЭТАП 4: Анализ операционной модели")
            self.analysis_state["operational_analysis"] = await self._stage4_operational_model(
                self.analysis_state["categorized_data"], assessment_id
            )
            
            # ЭТАП 5: Безопасность и ограничения
            bound_logger.info("🔒 ЭТАП 5: Анализ безопасности и ограничений")
            self.analysis_state["security_analysis"] = await self._stage5_security_analysis(
                self.analysis_state["categorized_data"], assessment_id
            )
            
            # ЭТАП 6: Генерация detailed_summary
            bound_logger.info("📝 ЭТАП 6: Генерация детального саммари")
            self.analysis_state["detailed_sections"] = await self._stage6_detailed_summary(
                self.analysis_state, assessment_id
            )
            
            # Объединяем все результаты в финальный профиль
            bound_logger.info("🔗 Объединяем результаты всех этапов")
            final_profile_data = self._merge_all_stages()
            
            # Валидируем итоговые данные
            profile_data = self._validate_and_fix_profile_data(final_profile_data, preliminary_name)
            
            # Создаем объект AgentProfile
            agent_profile = AgentProfile(
                name=profile_data["name"],
                version=profile_data.get("version", "1.0"),
                description=profile_data["description"],
                agent_type=AgentType(profile_data["agent_type"]),
                llm_model=profile_data["llm_model"],
                autonomy_level=AutonomyLevel(profile_data["autonomy_level"]),
                data_access=[DataSensitivity(da) for da in profile_data.get("data_access", [])],
                external_apis=profile_data.get("external_apis", []),
                target_audience=profile_data["target_audience"],
                operations_per_hour=profile_data.get("operations_per_hour"),
                revenue_per_operation=profile_data.get("revenue_per_operation"),
                system_prompts=profile_data.get("system_prompts", []),
                guardrails=profile_data.get("guardrails", []),
                source_files=collected_data.get("source_files", []),
                detailed_summary=profile_data.get("detailed_summary")
            )
            
            bound_logger.info("✅ ПОЭТАПНОЕ профилирование завершено успешно")
            return agent_profile
            
        except Exception as e:
            bound_logger.error(f"❌ Ошибка в поэтапном профилировании: {e}")
            raise

    # ==========================================
    # ЭТАП 1: Структурирование данных
    # ==========================================

    async def _stage1_categorize_data(
        self, 
        collected_data: Dict[str, Any], 
        assessment_id: str
    ) -> Dict[str, Any]:
        """
        ЭТАП 1: Разбор и категоризация собранных данных
        
        Цель: Структурировать данные по категориям для последующего анализа
        """
        
        bound_logger = self.logger.bind_context(assessment_id, self.name)
        
        # ДИАГНОСТИКА: Смотрим что пришло в collected_data
        bound_logger.info(f"🔍 ДИАГНОСТИКА collected_data:")
        bound_logger.info(f"   Ключи: {list(collected_data.keys())}")
        
        for key, value in collected_data.items():
            if isinstance(value, list):
                bound_logger.info(f"   {key}: список из {len(value)} элементов")
            elif isinstance(value, dict):
                bound_logger.info(f"   {key}: словарь с ключами {list(value.keys())}")
            else:
                bound_logger.info(f"   {key}: {type(value)} = {str(value)[:100]}")
        
        categorized = {
            "technical": [],      # Код, архитектура, технологии
            "documentation": [],  # README, инструкции, описания  
            "configuration": [],  # .env, JSON, настройки
            "prompts": [],       # Системные промпты, инструкции для LLM
            "security": [],      # Токены, ограничения, guardrails
            "business": []       # Метрики, аудитория, операции
        }
        
        # Документы - ИСПРАВЛЕННАЯ ЛОГИКА
        documents = collected_data.get("documents", [])
        bound_logger.info(f"🔍 Найдено документов: {len(documents)}")
        
        if documents:
            for i, doc in enumerate(documents):
                bound_logger.info(f"🔍 Документ {i}: ключи = {list(doc.keys()) if isinstance(doc, dict) else 'not dict'}")
                
                # Проверяем разные варианты структуры
                success = doc.get("success", True)  # По умолчанию True если нет поля
                file_path = doc.get("file_path", "") or doc.get("path", "") or doc.get("name", "")
                content = doc.get("content", "") or doc.get("text", "")
                sections = doc.get("sections", {})
                
                bound_logger.info(f"   Файл: {file_path}, success: {success}, content: {len(content)} символов")
                
                if success and file_path:  # Если файл успешно обработан
                    file_type = doc.get("file_type", "unknown")
                    
                    # Классифицируем по типу и содержимому
                    if any(keyword in file_path.lower() for keyword in ['readme', 'doc', 'manual']):
                        categorized["documentation"].append({
                            "source": "document",
                            "file": file_path,
                            "type": file_type,
                            "content": content,
                            "sections": sections
                        })
                    elif any(keyword in file_path.lower() for keyword in ['config', '.env', 'setting']):
                        categorized["configuration"].append({
                            "source": "document", 
                            "file": file_path,
                            "content": content
                        })
                    else:
                        # Анализируем содержимое для более точной категоризации
                        content_lower = content.lower()
                        if any(keyword in content_lower for keyword in ['prompt', 'instruction', 'system']):
                            categorized["prompts"].append({
                                "source": "document",
                                "file": file_path,
                                "content": content,
                                "sections": sections
                            })
                        elif any(keyword in content_lower for keyword in ['security', 'guardrail', 'limitation']):
                            categorized["security"].append({
                                "source": "document",
                                "file": file_path,
                                "content": content
                            })
                        else:
                            categorized["documentation"].append({
                                "source": "document",
                                "file": file_path,
                                "content": content,
                                "sections": sections
                            })
        
        # Код
        if collected_data.get("code_analysis"):
            code_data = collected_data["code_analysis"]
            categorized["technical"].append({
                "source": "code_analysis",
                "files_analyzed": code_data.get("files_analyzed", 0),
                "languages": code_data.get("languages", []),
                "frameworks": code_data.get("frameworks", []),
                "dependencies": code_data.get("dependencies", []),
                "main_functions": code_data.get("main_functions", []),
                "llm_integrations": code_data.get("llm_integrations", [])
            })
        
        # Промпты
        if collected_data.get("prompt_analysis"):
            prompt_data = collected_data["prompt_analysis"]
            categorized["prompts"].append({
                "source": "prompt_analysis",
                "system_prompts": prompt_data.get("system_prompts", []),
                "capabilities": prompt_data.get("capabilities", []),
                "personality_traits": prompt_data.get("personality_traits", []),
                "restrictions": prompt_data.get("restrictions", []),
                "risk_indicators": prompt_data.get("risk_indicators", [])
            })
            
            # Guardrails идут в security
            if prompt_data.get("guardrails"):
                categorized["security"].append({
                    "source": "prompt_analysis",
                    "guardrails": prompt_data["guardrails"]
                })
        
        bound_logger.info(f"📁 Структурировано: technical={len(categorized['technical'])}, "
                         f"docs={len(categorized['documentation'])}, "
                         f"config={len(categorized['configuration'])}, "
                         f"prompts={len(categorized['prompts'])}, "
                         f"security={len(categorized['security'])}, "
                         f"business={len(categorized['business'])}")
        
        return categorized

    # ==========================================
    # ЭТАП 2: Базовые поля
    # ==========================================

    async def _stage2_extract_basic_fields(
        self,
        categorized_data: Dict[str, Any],
        preliminary_name: str, 
        assessment_id: str
    ) -> Dict[str, Any]:
        """
        ЭТАП 2: Извлечение базовых полей профиля
        
        Цель: Заполнить основную структуру JSON профиля
        """
        
        # Формируем компактные данные для анализа
        analysis_data = self._format_categorized_data_for_basic_analysis(categorized_data)
        
        extraction_prompt = f"""Проанализируй данные об ИИ-агенте и извлеки БАЗОВЫЕ поля профиля.

СТРУКТУРА ОТВЕТА (JSON):
{{
    "name": "string - название агента",
    "version": "string - версия (по умолчанию 1.0)", 
    "description": "string - краткое описание назначения агента",
    "agent_type": "string - один из: chatbot, assistant, trader, scorer, analyzer, generator, other",
    "llm_model": "string - используемая LLM модель",
    "autonomy_level": "string - один из: supervised, semi_autonomous, autonomous",
    "data_access": ["array of strings - типы данных: public, internal, confidential, critical"],
    "external_apis": ["array of strings - внешние API"],
    "target_audience": "string - целевая аудитория"
}}

ПРАВИЛА:
- Если информация неясна, используй разумные значения по умолчанию
- name по умолчанию: "{preliminary_name}"
- Отвечай ТОЛЬКО валидным JSON без дополнительного текста
"""

        bound_logger = self.logger.bind_context(assessment_id, self.name)
        bound_logger.info(f"📏 Промпт для базовых полей: {len(extraction_prompt)} символов")
        bound_logger.info(f"📏 Данные для анализа: {len(analysis_data)} символов")
        
        # Вызываем LLM
        basic_fields = await self.call_llm_structured(
            data_to_analyze=analysis_data,
            extraction_prompt=extraction_prompt,
            assessment_id=assessment_id,
            expected_format="JSON"
        )
        
        bound_logger.info(f"✅ Этап 2 завершен: извлечены базовые поля для агента '{basic_fields.get('name', 'Unknown')}'")
        
        return basic_fields

    def _format_categorized_data_for_basic_analysis(self, categorized_data: Dict[str, Any]) -> str:
        """Форматирование данных для анализа базовых полей (компактно)"""
        
        parts = []
        
        # Техническая информация (кратко)
        if categorized_data.get("technical"):
            for tech in categorized_data["technical"]:
                if tech["source"] == "code_analysis":
                    parts.append(f"ТЕХНОЛОГИИ: {', '.join(tech.get('languages', []))}")
                    parts.append(f"ФРЕЙМВОРКИ: {', '.join(tech.get('frameworks', []))}")
                    if tech.get("llm_integrations"):
                        parts.append(f"LLM ИНТЕГРАЦИИ: {', '.join(tech['llm_integrations'])}")
        
        # Документация (краткие выдержки)
        if categorized_data.get("documentation"):
            for doc in categorized_data["documentation"][:2]:  # Только первые 2 документа
                content = doc["content"][:1000]  # Только первые 1000 символов
                parts.append(f"ДОКУМЕНТ: {content}")
        
        # Промпты (краткая информация)
        if categorized_data.get("prompts"):
            for prompt_data in categorized_data["prompts"]:
                if prompt_data.get("capabilities"):
                    parts.append(f"ВОЗМОЖНОСТИ: {', '.join(prompt_data['capabilities'][:5])}")  # Первые 5
        
        return "\n\n".join(parts)

    # ==========================================
    # ЭТАП 3: Техническая архитектура  
    # ==========================================

    async def _stage3_technical_architecture(
        self,
        categorized_data: Dict[str, Any],
        assessment_id: str
    ) -> Dict[str, Any]:
        """
        ЭТАП 3: Анализ технической архитектуры
        
        Цель: Определить технические детали, возможности и ограничения
        """
        
        # Фокусируемся на технических данных
        tech_data = self._format_technical_data(categorized_data)
        
        extraction_prompt = """Проанализируй техническую архитектуру ИИ-агента.

СТРУКТУРА ОТВЕТА (JSON):
{
    "technical_details": "string - подробности технической реализации",
    "capabilities": ["array of strings - список возможностей агента"],
    "limitations": ["array of strings - известные ограничения"],
    "operations_per_hour": "number or null - оценка операций в час",
    "revenue_per_operation": "number or null - оценка дохода с операции в рублях"
}

ЗАДАЧИ:
1. Анализируй используемые технологии и архитектурные решения
2. Определи функциональные возможности агента
3. Выяви технические ограничения
4. Оцени производительность (если данные есть)

Отвечай ТОЛЬКО валидным JSON."""

        bound_logger = self.logger.bind_context(assessment_id, self.name)
        bound_logger.info(f"📏 Промпт технической архитектуры: {len(extraction_prompt)} символов")
        bound_logger.info(f"📏 Технические данные: {len(tech_data)} символов")
        
        technical_analysis = await self.call_llm_structured(
            data_to_analyze=tech_data,
            extraction_prompt=extraction_prompt,
            assessment_id=assessment_id,
            expected_format="JSON"
        )
        
        bound_logger.info(f"✅ Этап 3 завершен: проанализирована техническая архитектура")
        
        return technical_analysis

    def _format_technical_data(self, categorized_data: Dict[str, Any]) -> str:
        """Форматирование технических данных"""
        
        parts = []
        
        # Технический анализ кода
        if categorized_data.get("technical"):
            for tech in categorized_data["technical"]:
                if tech["source"] == "code_analysis":
                    parts.append("=== АНАЛИЗ КОДА ===")
                    parts.append(f"Файлов проанализировано: {tech.get('files_analyzed', 0)}")
                    parts.append(f"Языки программирования: {', '.join(tech.get('languages', []))}")
                    parts.append(f"Фреймворки: {', '.join(tech.get('frameworks', []))}")
                    parts.append(f"Зависимости: {', '.join(tech.get('dependencies', []))}")
                    
                    if tech.get("main_functions"):
                        parts.append("Основные функции:")
                        for func in tech["main_functions"][:10]:  # Первые 10 функций
                            parts.append(f"  - {func}")
                    
                    if tech.get("llm_integrations"):
                        parts.append("LLM интеграции:")
                        for integration in tech["llm_integrations"]:
                            parts.append(f"  - {integration}")
        
        # Конфигурационные файлы
        if categorized_data.get("configuration"):
            parts.append("\n=== КОНФИГУРАЦИЯ ===")
            for config in categorized_data["configuration"]:
                content = config["content"][:2000]  # Первые 2000 символов
                parts.append(f"Файл: {config['file']}")
                parts.append(content)
        
        return "\n".join(parts)

    # ==========================================  
    # ЭТАП 4: Операционная модель
    # ==========================================

    async def _stage4_operational_model(
        self,
        categorized_data: Dict[str, Any], 
        assessment_id: str
    ) -> Dict[str, Any]:
        """
        ЭТАП 4: Анализ операционной модели
        
        Цель: Понять бизнес-процессы и взаимодействие с пользователями
        """
        
        # Фокусируемся на бизнес и документационных данных
        operational_data = self._format_operational_data(categorized_data)
        
        extraction_prompt = """Проанализируй операционную модель ИИ-агента.

СТРУКТУРА ОТВЕТА (JSON):
{
    "enhanced_target_audience": "string - детальное описание целевой аудитории",
    "business_processes": ["array of strings - бизнес-процессы"],
    "interaction_patterns": ["array of strings - паттерны взаимодействия"],
    "performance_metrics": ["array of strings - ключевые метрики"]
}

ЗАДАЧИ:
1. Определи как агент взаимодействует с пользователями
2. Выяви основные бизнес-процессы
3. Найди метрики производительности и KPI
4. Опиши операционную модель

Отвечай ТОЛЬКО валидным JSON."""

        bound_logger = self.logger.bind_context(assessment_id, self.name)
        bound_logger.info(f"📏 Промпт операционной модели: {len(extraction_prompt)} символов")
        bound_logger.info(f"📏 Операционные данные: {len(operational_data)} символов")
        
        operational_analysis = await self.call_llm_structured(
            data_to_analyze=operational_data,
            extraction_prompt=extraction_prompt,
            assessment_id=assessment_id,
            expected_format="JSON"
        )
        
        bound_logger.info(f"✅ Этап 4 завершен: проанализирована операционная модель")
        
        return operational_analysis

    def _format_operational_data(self, categorized_data: Dict[str, Any]) -> str:
        """Форматирование операционных данных"""
        
        parts = []
        
        # Документация с фокусом на бизнес-процессы
        if categorized_data.get("documentation"):
            parts.append("=== БИЗНЕС ДОКУМЕНТАЦИЯ ===")
            for doc in categorized_data["documentation"]:
                parts.append(f"\nДокумент: {doc['file']}")
                
                # Ищем бизнес-информацию в секциях
                for section_name, section_content in doc.get("sections", {}).items():
                    if any(keyword in section_name.lower() for keyword in 
                           ['business', 'process', 'workflow', 'user', 'audience', 'metric']):
                        parts.append(f"[{section_name}]")
                        parts.append(section_content[:1500])  # Первые 1500 символов
        
        # Промпты с фокусом на взаимодействие
        if categorized_data.get("prompts"):
            parts.append("\n=== МОДЕЛЬ ВЗАИМОДЕЙСТВИЯ ===")
            for prompt_data in categorized_data["prompts"]:
                if prompt_data.get("capabilities"):
                    parts.append("Возможности агента:")
                    parts.append(", ".join(prompt_data["capabilities"]))
                
                if prompt_data.get("personality_traits"):
                    parts.append("Характеристики личности:")
                    parts.append(", ".join(prompt_data["personality_traits"]))
        
        return "\n".join(parts)

    # ==========================================
    # ЭТАП 5: Безопасность и ограничения
    # ==========================================

    async def _stage5_security_analysis(
        self,
        categorized_data: Dict[str, Any],
        assessment_id: str
    ) -> Dict[str, Any]:
        """
        ЭТАП 5: Анализ безопасности и ограничений
        
        Цель: Выявить меры безопасности, ограничения и риски
        """
        
        # Фокусируемся на данных безопасности и промптах
        security_data = self._format_security_data(categorized_data)
        
        extraction_prompt = """Проанализируй аспекты безопасности ИИ-агента.

СТРУКТУРА ОТВЕТА (JSON):
{
    "system_prompts": ["array of strings - системные промпты"],
    "guardrails": ["array of strings - ограничения безопасности"],
    "risk_factors": ["array of strings - выявленные факторы риска"],
    "security_measures": ["array of strings - меры безопасности"]
}

ЗАДАЧИ:
1. Извлеки все системные промпты и инструкции
2. Найди ограничения и guardrails
3. Выяви потенциальные риски и уязвимости
4. Определи меры безопасности

Отвечай ТОЛЬКО валидным JSON."""

        bound_logger = self.logger.bind_context(assessment_id, self.name)
        bound_logger.info(f"📏 Промпт анализа безопасности: {len(extraction_prompt)} символов")
        bound_logger.info(f"📏 Данные безопасности: {len(security_data)} символов")
        
        security_analysis = await self.call_llm_structured(
            data_to_analyze=security_data,
            extraction_prompt=extraction_prompt,
            assessment_id=assessment_id,
            expected_format="JSON"
        )
        
        bound_logger.info(f"✅ Этап 5 завершен: проанализированы аспекты безопасности")
        
        return security_analysis

    def _format_security_data(self, categorized_data: Dict[str, Any]) -> str:
        """Форматирование данных безопасности"""
        
        parts = []
        
        # Данные безопасности
        if categorized_data.get("security"):
            parts.append("=== БЕЗОПАСНОСТЬ ===")
            for security in categorized_data["security"]:
                if security.get("guardrails"):
                    parts.append("Guardrails:")
                    for guardrail in security["guardrails"]:
                        parts.append(f"  - {guardrail}")
                
                if security.get("content"):
                    parts.append(f"Содержимое: {security['content'][:1000]}")
        
        # Промпты с фокусом на безопасность
        if categorized_data.get("prompts"):
            parts.append("\n=== СИСТЕМНЫЕ ПРОМПТЫ ===")
            for prompt_data in categorized_data["prompts"]:
                if prompt_data.get("system_prompts"):
                    for prompt in prompt_data["system_prompts"]:
                        parts.append(f"Промпт: {prompt[:1000]}")  # Первые 1000 символов
                
                if prompt_data.get("restrictions"):
                    parts.append("Ограничения:")
                    parts.append(", ".join(prompt_data["restrictions"]))
                
                if prompt_data.get("risk_indicators"):
                    parts.append("Индикаторы риска:")
                    parts.append(", ".join(prompt_data["risk_indicators"]))
        
        # Конфигурационные файлы (могут содержать токены/ключи)
        if categorized_data.get("configuration"):
            parts.append("\n=== КОНФИГУРАЦИЯ (анализ на секреты) ===")
            for config in categorized_data["configuration"]:
                content = config["content"][:500]  # Краткий анализ
                parts.append(f"Файл: {config['file']}")
                parts.append(content)
        
        return "\n".join(parts)

    # ==========================================
    # ЭТАП 6: Detailed Summary
    # ==========================================

    async def _stage6_detailed_summary(
        self,
        analysis_state: Dict[str, Any],
        assessment_id: str
    ) -> Dict[str, Any]:
        """
        ЭТАП 6: Генерация детального саммари
        
        Цель: Создать подробные описания для каждого раздела detailed_summary
        """
        
        bound_logger = self.logger.bind_context(assessment_id, self.name)
        
        detailed_sections = {}
        
        # Генерируем каждый раздел отдельными запросами
        sections_to_generate = [
            ("overview", "Напиши подробный обзор агента (назначение, функции, область применения) - минимум 200 слов"),
            ("technical_architecture", "Напиши детальный анализ технической реализации и архитектуры - минимум 150 слов"), 
            ("operational_model", "Опиши процессы работы и взаимодействия агента с пользователями - минимум 150 слов"),
            ("conclusions", "Напиши итоговую оценку агента и ключевые выводы - минимум 100 слов")
        ]
        
        for section_name, section_prompt in sections_to_generate:
            bound_logger.info(f"📝 Генерация раздела: {section_name}")
            
            # Подготавливаем данные для конкретного раздела
            section_data = self._prepare_data_for_section(analysis_state, section_name)
            
            # ИСПРАВЛЕННЫЙ ПРОМПТ - четко требуем текст
            full_prompt = f"""
{section_prompt}

КРИТИЧЕСКИ ВАЖНО:
- Отвечай ТОЛЬКО обычным текстом
- НЕ используй JSON формат
- НЕ используй структурированные данные
- Пиши как обычное описание/эссе
- Никаких {{}} скобок или "ключ": "значение"

Пример ПРАВИЛЬНОГО ответа:
Lawdigest_bot представляет собой комплексную систему для автоматического анализа правовых документов. Система разработана для помощи юристам и правовым консультантам в обработке больших объемов юридической информации...

ТВОЙ ОТВЕТ (только текст):"""
            
            section_content = await self.call_llm_structured(
                data_to_analyze=section_data,
                extraction_prompt=full_prompt,
                assessment_id=assessment_id,
                expected_format="TEXT"
            )
            
            # Если все же получили словарь - извлекаем любой текстовый контент
            if isinstance(section_content, dict):
                # Пытаемся найти текстовое поле
                possible_text_fields = ["text", "content", "response", "answer", "description", "result"]
                extracted_text = None
                
                for field in possible_text_fields:
                    if field in section_content and isinstance(section_content[field], str):
                        extracted_text = section_content[field]
                        break
                
                # Если не нашли текстовые поля, берем все строковые значения
                if not extracted_text:
                    string_values = [str(v) for v in section_content.values() if isinstance(v, str) and len(str(v)) > 50]
                    if string_values:
                        extracted_text = string_values[0]
                    else:
                        # В крайнем случае конвертируем весь словарь в текст
                        extracted_text = f"Сгенерированный анализ для раздела {section_name} на основе доступных данных."
                
                section_content = extracted_text
            
            detailed_sections[section_name] = section_content
            
            bound_logger.info(f"✅ Раздел {section_name}: {len(str(section_content))} символов")
        
        bound_logger.info(f"✅ Этап 6 завершен: сгенерировано {len(detailed_sections)} разделов")
        
        return detailed_sections

    def _prepare_data_for_section(self, analysis_state: Dict[str, Any], section_name: str) -> str:
        """Подготавливает данные для генерации конкретного раздела"""
        
        parts = []
        
        if section_name == "overview":
            # Для обзора используем базовый профиль и операционную модель
            basic = analysis_state.get("basic_profile", {})
            operational = analysis_state.get("operational_analysis", {})
            
            parts.append(f"АГЕНТ: {basic.get('name', 'Unknown')}")
            parts.append(f"ТИП: {basic.get('agent_type', 'unknown')}")
            parts.append(f"ОПИСАНИЕ: {basic.get('description', 'N/A')}")
            parts.append(f"АУДИТОРИЯ: {basic.get('target_audience', 'N/A')}")
            
            if operational.get("business_processes"):
                parts.append("ПРОЦЕССЫ: " + ", ".join(operational["business_processes"]))
                
        elif section_name == "technical_architecture":
            # Для технической архитектуры используем технический анализ
            technical = analysis_state.get("technical_analysis", {})
            
            parts.append(f"ТЕХНИЧЕСКИЕ ДЕТАЛИ: {technical.get('technical_details', 'N/A')}")
            
            if technical.get("capabilities"):
                parts.append("ВОЗМОЖНОСТИ: " + ", ".join(technical["capabilities"]))
            
            if technical.get("limitations"):
                parts.append("ОГРАНИЧЕНИЯ: " + ", ".join(technical["limitations"]))
                
        elif section_name == "operational_model":
            # Для операционной модели используем операционный анализ
            operational = analysis_state.get("operational_analysis", {})
            
            if operational.get("interaction_patterns"):
                parts.append("ВЗАИМОДЕЙСТВИЕ: " + ", ".join(operational["interaction_patterns"]))
            
            if operational.get("performance_metrics"):
                parts.append("МЕТРИКИ: " + ", ".join(operational["performance_metrics"]))
                
        elif section_name == "conclusions":
            # Для выводов используем данные из всех этапов
            basic = analysis_state.get("basic_profile", {})
            technical = analysis_state.get("technical_analysis", {})
            security = analysis_state.get("security_analysis", {})
            
            parts.append(f"АГЕНТ: {basic.get('name')} ({basic.get('agent_type')})")
            parts.append(f"АВТОНОМНОСТЬ: {basic.get('autonomy_level')}")
            
            if security.get("risk_factors"):
                parts.append("РИСКИ: " + ", ".join(security["risk_factors"][:3]))  # Первые 3 риска
        
        return "\n".join(parts)

    # ==========================================
    # Объединение результатов
    # ==========================================

    def _merge_all_stages(self) -> Dict[str, Any]:
        """Объединяет результаты всех этапов в финальный профиль"""
        
        final_profile = {}
        
        # Базовые поля (этап 2)
        basic = self.analysis_state.get("basic_profile", {})
        final_profile.update(basic)
        
        # Технические поля (этап 3)
        technical = self.analysis_state.get("technical_analysis", {})
        final_profile.update({
            "technical_details": technical.get("technical_details"),
            "capabilities": technical.get("capabilities", []),
            "limitations": technical.get("limitations", []),
            "operations_per_hour": technical.get("operations_per_hour"),
            "revenue_per_operation": technical.get("revenue_per_operation")
        })
        
        # Операционные поля (этап 4)
        operational = self.analysis_state.get("operational_analysis", {})
        if operational.get("enhanced_target_audience"):
            final_profile["target_audience"] = operational["enhanced_target_audience"]
        
        # Поля безопасности (этап 5)
        security = self.analysis_state.get("security_analysis", {})
        final_profile.update({
            "system_prompts": security.get("system_prompts", []),
            "guardrails": security.get("guardrails", []),
            "risk_factors": security.get("risk_factors", [])
        })
        
        # Детальное саммари (этап 6)
        detailed_sections = self.analysis_state.get("detailed_sections", {})
        if detailed_sections:
            final_profile["detailed_summary"] = detailed_sections
        
        return final_profile

    # ==========================================
    # СУЩЕСТВУЮЩИЕ МЕТОДЫ (без изменений)
    # ==========================================

    async def _collect_all_data(self, source_files: List[str], assessment_id: str) -> Dict[str, Any]:
        """Сбор данных из всех источников (существующий метод без изменений)"""
        
        collected_data = {
            "source_files": source_files,
            "documents": [],
            "code_analysis": None,
            "prompt_analysis": None,
            "errors": []
        }
        
        # Преобразуем пути в Path объекты для анализа
        files_to_parse = []
        for file_path in source_files:
            path_obj = Path(file_path)
            if path_obj.exists():
                if path_obj.is_file():
                    files_to_parse.append(path_obj)
                elif path_obj.is_dir():
                    # Для папки добавляем все файлы рекурсивно
                    for file in path_obj.rglob("*"):
                        if file.is_file():
                            files_to_parse.append(file)
            else:
                collected_data["errors"].append(f"Файл не найден: {file_path}")
        
        # 1. Парсинг документов
        self.logger.bind_context(assessment_id, self.name).info(
            "📄 Парсинг документов"
        )
        
        try:
            documents_result = parse_agent_documents(files_to_parse, self.document_parser)
            if documents_result.success:
                collected_data["documents"] = documents_result.documents
        except Exception as e:
            collected_data["errors"].append(f"Ошибка парсинга документов: {e}")
        
        # 2. Анализ кода
        self.logger.bind_context(assessment_id, self.name).info(
            "💻 Анализ кода"
        )
        
        try:
            # Ищем корневую директорию проекта
            project_root = None
            for file_path in files_to_parse:
                if file_path.is_dir():
                    project_root = file_path
                    break
                else:
                    project_root = file_path.parent
                    break
            
            if project_root:
                code_analysis_result = analyze_agent_codebase(project_root, self.code_analyzer)
                if code_analysis_result.success:
                    collected_data["code_analysis"] = code_analysis_result.analysis_data
        except Exception as e:
            collected_data["errors"].append(f"Ошибка анализа кода: {e}")
        
        # 3. Анализ промптов
        self.logger.bind_context(assessment_id, self.name).info(
            "🔍 Анализ промптов и инструкций"
        )
        
        try:
            # Извлекаем тексты для анализа промптов
            prompt_sources = []
            
            # Из документов
            for doc in collected_data["documents"]:
                if doc["success"]:
                    # Добавляем содержимое секций с промптами
                    for section_name, section_content in doc["sections"].items():
                        if any(keyword in section_name.lower() for keyword in 
                               ['prompt', 'instruction', 'system', 'guardrail']):
                            prompt_sources.append(section_content)
                    
                    # Добавляем общий контент если он небольшой
                    if len(doc["content"]) < 5000:
                        prompt_sources.append(doc["content"])
            
            # Из кода (комментарии и строки)
            if collected_data["code_analysis"]:
                # Анализируем сами файлы кода
                for file_path in files_to_parse:
                    if file_path.suffix.lower() in ['.py', '.js', '.java']:
                        prompt_sources.append(str(file_path))
            
            if prompt_sources:
                prompt_analysis = analyze_agent_prompts(prompt_sources)
                
                if prompt_analysis.success:
                    collected_data["prompt_analysis"] = {
                        "total_prompts": prompt_analysis.total_prompts,
                        "system_prompts": [p.content for p in prompt_analysis.system_prompts],
                        "guardrails": [p.content for p in prompt_analysis.guardrails],
                        "capabilities": prompt_analysis.capabilities,
                        "personality_traits": prompt_analysis.personality_traits,
                        "restrictions": prompt_analysis.restrictions,
                        "risk_indicators": prompt_analysis.risk_indicators,
                        "complexity_score": prompt_analysis.complexity_score
                    }
                
        except Exception as e:
            collected_data["errors"].append(f"Ошибка анализа промптов: {e}")
        
        return collected_data

    def _validate_and_fix_profile_data(
        self, 
        llm_result: Dict[str, Any], 
        preliminary_name: str
    ) -> Dict[str, Any]:
        """Валидация и исправление данных профиля от LLM (существующий метод)"""
        
        # Значения по умолчанию
        defaults = {
            "name": preliminary_name,
            "version": "1.0",
            "description": "ИИ-агент (описание не найдено)",
            "agent_type": "other",
            "llm_model": "unknown",
            "autonomy_level": "supervised",
            "data_access": ["internal"],
            "external_apis": [],
            "target_audience": "Пользователи системы",
            "operations_per_hour": None,
            "revenue_per_operation": None,
            "system_prompts": [],
            "guardrails": []
        }
        
        # Применяем значения по умолчанию
        for key, default_value in defaults.items():
            if key not in llm_result or llm_result[key] is None:
                llm_result[key] = default_value
        
        # Валидация енумов
        valid_agent_types = [e.value for e in AgentType]
        if llm_result["agent_type"] not in valid_agent_types:
            llm_result["agent_type"] = "other"
        
        valid_autonomy_levels = [e.value for e in AutonomyLevel]
        if llm_result["autonomy_level"] not in valid_autonomy_levels:
            llm_result["autonomy_level"] = "supervised"
        
        valid_data_sensitivities = [e.value for e in DataSensitivity]
        validated_data_access = []
        for da in llm_result["data_access"]:
            if da in valid_data_sensitivities:
                validated_data_access.append(da)
        if not validated_data_access:
            validated_data_access = ["internal"]
        llm_result["data_access"] = validated_data_access
        
        return llm_result

    def _create_data_summary(self, collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Создание сводки собранных данных (существующий метод)"""
        
        summary = {
            "files_processed": len(collected_data.get("source_files", [])),
            "documents_parsed": 0,
            "documents_success": 0,
            "code_analysis_success": False,
            "prompt_analysis_success": False,
            "total_errors": len(collected_data.get("errors", []))
        }
        
        # Документы
        if collected_data.get("documents"):
            summary["documents_parsed"] = len(collected_data["documents"])
            summary["documents_success"] = sum(1 for doc in collected_data["documents"] if doc["success"])
        
        # Код
        if collected_data.get("code_analysis"):
            summary["code_analysis_success"] = True
            summary["total_files_analyzed"] = collected_data["code_analysis"]["files_analyzed"]
            summary["languages_found"] = list(collected_data["code_analysis"]["languages"].keys())
        
        # Промпты
        if collected_data.get("prompt_analysis"):
            summary["prompt_analysis_success"] = True
            summary["total_prompts_found"] = collected_data["prompt_analysis"]["total_prompts"]
            summary["system_prompts_found"] = len(collected_data["prompt_analysis"]["system_prompts"])
            summary["guardrails_found"] = len(collected_data["prompt_analysis"]["guardrails"])
        
        return summary
    
    def _get_required_result_fields(self) -> List[str]:
        """Обязательные поля результата профайлера"""
        return ["agent_profile", "collected_data_summary"]


# ===============================
# Интеграция с LangGraph (без изменений)
# ===============================

def create_profiler_node_function(profiler_agent: ProfilerAgent):
    """
    Создает функцию узла для LangGraph workflow
    
    Args:
        profiler_agent: Экземпляр профайлер-агента
        
    Returns:
        Функция для использования в LangGraph
    """
    async def profiler_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Узел профайлера в LangGraph workflow"""
        
        # Извлекаем данные из состояния
        assessment_id = state.get("assessment_id", "unknown")
        source_files = state.get("source_files", [])
        agent_name = state.get("preliminary_agent_name", "Unknown_Agent")
        
        # Подготавливаем входные данные
        input_data = {
            "source_files": source_files,
            "agent_name": agent_name
        }
        
        # Запускаем профайлер
        result = await profiler_agent.run(input_data, assessment_id)
        
        # Обновляем состояние - преобразуем AgentTaskResult в словарь
        updated_state = state.copy()
        updated_state["profiling_result"] = result.dict()  # Преобразуем в словарь
        
        if result.status == ProcessingStatus.COMPLETED:
            # Добавляем профиль агента в состояние для дальнейшего использования
            agent_profile_data = result.result_data["agent_profile"]
            updated_state["agent_profile"] = agent_profile_data
            updated_state["current_step"] = "evaluation_preparation"
        else:
            updated_state["current_step"] = "error"
            updated_state["error_message"] = result.error_message
        
        return updated_state
    
    return profiler_node


# ===============================
# Фабрики (без изменений)
# ===============================

def create_profiler_agent(
    llm_base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    temperature: Optional[float] = None
) -> ProfilerAgent:
    """
    Создание профайлер-агента
    ОБНОВЛЕНО: Использует центральный конфигуратор
    """
    from .base_agent import create_agent_config
    
    config = create_agent_config(
        name="profiler_agent",
        description="Агент для профилирования ИИ-агентов и сбора данных для оценки рисков",
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        temperature=temperature,
        max_retries=3,
        timeout_seconds=1800,  # Увеличенный тайм-аут для анализа больших объемов данных
        use_risk_analysis_client=False  # Профайлер использует стандартный клиент
    )
    
    return ProfilerAgent(config)

def create_profiler_from_env() -> ProfilerAgent:
    """
    Создание профайлер-агента из переменных окружения
    ОБНОВЛЕНО: Использует центральный конфигуратор
    """
    # ИЗМЕНЕНО: Используем центральный конфигуратор, убираем дублирование чтения env
    return create_profiler_agent()


# Экспорт
__all__ = [
    "ProfilerAgent",
    "create_profiler_agent",
    "create_profiler_from_env",
    "create_profiler_node_function"
]