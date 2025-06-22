# src/tools/prompt_analyzer.py
"""
Анализатор промптов и инструкций для ИИ-агентов
Извлекает и анализирует системные промпты, guardrails, инструкции
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass
from datetime import datetime

from ..utils.logger import get_logger, LogContext


@dataclass
class PromptElement:
    """Элемент промпта"""
    type: str  # system, user, instruction, guardrail, example
    content: str
    source_file: str
    line_number: Optional[int] = None
    confidence: float = 0.0
    language: str = "unknown"


@dataclass 
class PromptAnalysisResult:
    """Результат анализа промптов"""
    total_prompts: int
    system_prompts: List[PromptElement]
    user_instructions: List[PromptElement]
    guardrails: List[PromptElement]
    examples: List[PromptElement]
    personality_traits: List[str]
    capabilities: List[str]
    restrictions: List[str]
    languages_detected: List[str]
    sentiment_analysis: Dict[str, Any]
    complexity_score: float
    risk_indicators: List[str]
    analysis_time: float
    success: bool
    error_message: Optional[str] = None


class PromptAnalyzerError(Exception):
    """Исключение при анализе промптов"""
    pass


class PromptExtractor:
    """Извлечение промптов из различных источников"""
    
    def __init__(self):
        self.logger = get_logger()
    
    def extract_from_text(self, text: str, source_file: str) -> List[PromptElement]:
        """Извлечение промптов из текста"""
        prompts = []
        
        # Паттерны для поиска промптов
        patterns = {
            "system": [
                r'(?i)(?:system|системный)\s*prompt[:\s]*(.+?)(?=\n\n|\n[A-ZА-Я]|$)',
                r'(?i)(?:system|системный)\s*(?:message|сообщение)[:\s]*(.+?)(?=\n\n|\n[A-ZА-Я]|$)',
                r'(?i)system[:\s]*"([^"]+)"',
                r'(?i)system[:\s]*\'([^\']+)\''
            ],
            "instruction": [
                r'(?i)(?:instruction|инструкция)[:\s]*(.+?)(?=\n\n|\n[A-ZА-Я]|$)',
                r'(?i)(?:guideline|руководство|указание)[:\s]*(.+?)(?=\n\n|\n[A-ZА-Я]|$)',
                r'(?i)(?:task|задача)[:\s]*(.+?)(?=\n\n|\n[A-ZА-Я]|$)'
            ],
            "guardrail": [
                r'(?i)(?:guardrail|ограничение|запрет)[:\s]*(.+?)(?=\n\n|\n[A-ZА-Я]|$)',
                r'(?i)(?:не должен|запрещено|нельзя)[:\s]*(.+?)(?=\n\n|\n[A-ZА-Я]|$)',
                r'(?i)(?:restriction|limitation)[:\s]*(.+?)(?=\n\n|\n[A-ZА-Я]|$)'
            ],
            "example": [
                r'(?i)(?:example|пример)[:\s]*(.+?)(?=\n\n|\n[A-ZА-Я]|$)',
                r'(?i)(?:sample|образец)[:\s]*(.+?)(?=\n\n|\n[A-ZА-Я]|$)'
            ]
        }
        
        lines = text.split('\n')
        
        for prompt_type, type_patterns in patterns.items():
            for pattern in type_patterns:
                matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    content = match.group(1).strip()
                    if len(content) > 10:  # Минимальная длина промпта
                        line_num = text[:match.start()].count('\n') + 1
                        
                        prompts.append(PromptElement(
                            type=prompt_type,
                            content=content,
                            source_file=source_file,
                            line_number=line_num,
                            confidence=self._calculate_confidence(content, prompt_type),
                            language=self._detect_language(content)
                        ))
        
        # Дополнительно ищем промпты в кавычках
        prompts.extend(self._extract_quoted_prompts(text, source_file))
        
        # Ищем списки ограничений
        prompts.extend(self._extract_restriction_lists(text, source_file))
        
        return prompts
    
    def extract_from_code(self, code: str, source_file: str, language: str) -> List[PromptElement]:
        """Извлечение промптов из кода"""
        prompts = []
        
        if language == "python":
            prompts.extend(self._extract_from_python_code(code, source_file))
        elif language == "javascript":
            prompts.extend(self._extract_from_js_code(code, source_file))
        
        return prompts
    
    def extract_from_config(self, config_data: Dict[str, Any], source_file: str) -> List[PromptElement]:
        """Извлечение промптов из конфигурационных данных"""
        prompts = []
        
        def traverse_config(data, path=""):
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Проверяем ключи, которые могут содержать промпты
                    if any(keyword in key.lower() for keyword in 
                           ['prompt', 'instruction', 'system', 'message', 'template']):
                        if isinstance(value, str) and len(value) > 10:
                            prompts.append(PromptElement(
                                type=self._determine_type_from_key(key),
                                content=value,
                                source_file=source_file,
                                confidence=0.9,
                                language=self._detect_language(value)
                            ))
                    else:
                        traverse_config(value, current_path)
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    traverse_config(item, f"{path}[{i}]")
        
        traverse_config(config_data)
        return prompts
    
    def _extract_quoted_prompts(self, text: str, source_file: str) -> List[PromptElement]:
        """Извлечение промптов в кавычках"""
        prompts = []
        
        # Многострочные строки в тройных кавычках
        triple_quote_patterns = [
            r'"""([^"]{50,})"""',
            r"'''([^']{50,})'''"
        ]
        
        for pattern in triple_quote_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                content = match.group(1).strip()
                
                # Проверяем, похоже ли на промпт
                if self._looks_like_prompt(content):
                    line_num = text[:match.start()].count('\n') + 1
                    
                    prompts.append(PromptElement(
                        type=self._guess_prompt_type(content),
                        content=content,
                        source_file=source_file,
                        line_number=line_num,
                        confidence=0.7,
                        language=self._detect_language(content)
                    ))
        
        return prompts
    
    def _extract_restriction_lists(self, text: str, source_file: str) -> List[PromptElement]:
        """Извлечение списков ограничений"""
        prompts = []
        lines = text.split('\n')
        
        current_list = []
        in_restriction_block = False
        block_start_line = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Начало блока ограничений
            if any(keyword in line_stripped.lower() for keyword in 
                   ['ограничения:', 'запреты:', 'не должен:', 'нельзя:', 'restrictions:']):
                if current_list:
                    # Сохраняем предыдущий блок
                    content = '\n'.join(current_list)
                    if len(content) > 20:
                        prompts.append(PromptElement(
                            type="guardrail",
                            content=content,
                            source_file=source_file,
                            line_number=block_start_line,
                            confidence=0.8,
                            language=self._detect_language(content)
                        ))
                
                current_list = []
                in_restriction_block = True
                block_start_line = i + 1
                continue
            
            # Элементы списка
            if in_restriction_block and (line_stripped.startswith(('•', '-', '*', '1.', '2.')) or 
                                        re.match(r'^\d+\.', line_stripped)):
                current_list.append(line_stripped)
            elif in_restriction_block and not line_stripped:
                continue  # Пустые строки игнорируем
            elif in_restriction_block and line_stripped:
                # Конец блока
                in_restriction_block = False
                if current_list:
                    content = '\n'.join(current_list)
                    if len(content) > 20:
                        prompts.append(PromptElement(
                            type="guardrail",
                            content=content,
                            source_file=source_file,
                            line_number=block_start_line,
                            confidence=0.8,
                            language=self._detect_language(content)
                        ))
                current_list = []
        
        # Обрабатываем последний блок
        if current_list:
            content = '\n'.join(current_list)
            if len(content) > 20:
                prompts.append(PromptElement(
                    type="guardrail",
                    content=content,
                    source_file=source_file,
                    line_number=block_start_line,
                    confidence=0.8,
                    language=self._detect_language(content)
                ))
        
        return prompts
    
    def _extract_from_python_code(self, code: str, source_file: str) -> List[PromptElement]:
        """Извлечение промптов из Python кода"""
        prompts = []
        
        # Строковые константы
        string_patterns = [
            r'(?:system_prompt|SYSTEM_PROMPT)\s*=\s*[rf]?["\']([^"\']+)["\']',
            r'(?:prompt|PROMPT)\s*=\s*[rf]?["\']([^"\']+)["\']',
            r'(?:instruction|INSTRUCTION)\s*=\s*[rf]?["\']([^"\']+)["\']'
        ]
        
        for pattern in string_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE | re.DOTALL)
            for match in matches:
                content = match.group(1)
                if len(content) > 20:
                    line_num = code[:match.start()].count('\n') + 1
                    
                    prompts.append(PromptElement(
                        type="system",
                        content=content,
                        source_file=source_file,
                        line_number=line_num,
                        confidence=0.9,
                        language=self._detect_language(content)
                    ))
        
        # f-строки и форматирование
        f_string_pattern = r'f["\']([^"\']*\{[^}]+\}[^"\']*)["\']'
        matches = re.finditer(f_string_pattern, code)
        for match in matches:
            content = match.group(1)
            if len(content) > 30 and self._looks_like_prompt(content):
                line_num = code[:match.start()].count('\n') + 1
                
                prompts.append(PromptElement(
                    type="user",
                    content=content,
                    source_file=source_file,
                    line_number=line_num,
                    confidence=0.6,
                    language=self._detect_language(content)
                ))
        
        return prompts
    
    def _extract_from_js_code(self, code: str, source_file: str) -> List[PromptElement]:
        """Извлечение промптов из JavaScript кода"""
        prompts = []
        
        # Константы и переменные
        js_patterns = [
            r'(?:const|let|var)\s+(?:systemPrompt|SYSTEM_PROMPT)\s*=\s*[`"\']((?:[^`"\'\\]|\\.)*)[\`"\']',
            r'(?:const|let|var)\s+(?:prompt|PROMPT)\s*=\s*[`"\']((?:[^`"\'\\]|\\.)*)[\`"\']',
            r'(?:const|let|var)\s+(?:instruction|INSTRUCTION)\s*=\s*[`"\']((?:[^`"\'\\]|\\.)*)[\`"\']'
        ]
        
        for pattern in js_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE | re.DOTALL)
            for match in matches:
                content = match.group(1)
                if len(content) > 20:
                    line_num = code[:match.start()].count('\n') + 1
                    
                    prompts.append(PromptElement(
                        type="system",
                        content=content,
                        source_file=source_file,
                        line_number=line_num,
                        confidence=0.9,
                        language=self._detect_language(content)
                    ))
        
        # Template literals
        template_pattern = r'`([^`]*\$\{[^}]+\}[^`]*)`'
        matches = re.finditer(template_pattern, code, re.DOTALL)
        for match in matches:
            content = match.group(1)
            if len(content) > 30 and self._looks_like_prompt(content):
                line_num = code[:match.start()].count('\n') + 1
                
                prompts.append(PromptElement(
                    type="user",
                    content=content,
                    source_file=source_file,
                    line_number=line_num,
                    confidence=0.6,
                    language=self._detect_language(content)
                ))
        
        return prompts
    
    def _calculate_confidence(self, content: str, prompt_type: str) -> float:
        """Расчет уверенности в том, что это промпт"""
        confidence = 0.5
        
        # Длина контента
        if len(content) > 50:
            confidence += 0.2
        if len(content) > 200:
            confidence += 0.1
        
        # Наличие типичных для промптов слов
        prompt_indicators = [
            'assistant', 'user', 'helpful', 'answer', 'question',
            'помощник', 'пользователь', 'ответ', 'вопрос', 'задача'
        ]
        
        content_lower = content.lower()
        indicator_count = sum(1 for indicator in prompt_indicators if indicator in content_lower)
        confidence += indicator_count * 0.05
        
        # Структура предложений
        sentences = content.split('.')
        if len(sentences) > 1:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _detect_language(self, text: str) -> str:
        """Простое определение языка"""
        # Подсчет кириллических символов
        cyrillic_count = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
        latin_count = sum(1 for char in text if char.isalpha() and char.isascii())
        
        total_letters = cyrillic_count + latin_count
        if total_letters == 0:
            return "unknown"
        
        cyrillic_ratio = cyrillic_count / total_letters
        
        if cyrillic_ratio > 0.3:
            return "ru"
        else:
            return "en"
    
    def _looks_like_prompt(self, content: str) -> bool:
        """Проверка, похож ли текст на промпт"""
        # Минимальная длина
        if len(content) < 20:
            return False
        
        # Наличие типичных слов
        prompt_words = [
            'you are', 'you should', 'your task', 'your role',
            'ты', 'твоя задача', 'твоя роль', 'ты должен',
            'assistant', 'помощник', 'system', 'системный'
        ]
        
        content_lower = content.lower()
        return any(word in content_lower for word in prompt_words)
    
    def _guess_prompt_type(self, content: str) -> str:
        """Угадывание типа промпта по содержимому"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['system', 'системный', 'you are', 'ты ']):
            return "system"
        elif any(word in content_lower for word in ['instruction', 'инструкция', 'task', 'задача']):
            return "instruction"
        elif any(word in content_lower for word in ['example', 'пример', 'sample', 'образец']):
            return "example"
        elif any(word in content_lower for word in ['not', 'don\'t', 'нельзя', 'не должен']):
            return "guardrail"
        else:
            return "user"
    
    def _determine_type_from_key(self, key: str) -> str:
        """Определение типа промпта по ключу конфигурации"""
        key_lower = key.lower()
        
        if 'system' in key_lower:
            return "system"
        elif 'instruction' in key_lower:
            return "instruction"
        elif 'guardrail' in key_lower or 'restriction' in key_lower:
            return "guardrail"
        elif 'example' in key_lower:
            return "example"
        else:
            return "user"


class PromptAnalyzer:
    """Главный класс для анализа промптов"""
    
    def __init__(self):
        self.extractor = PromptExtractor()
        self.logger = get_logger()
    
    def analyze_prompts(
        self, 
        sources: List[Union[str, Path, Dict[str, Any]]], 
        source_types: Optional[List[str]] = None
    ) -> PromptAnalysisResult:
        """
        Анализ промптов из различных источников
        
        Args:
            sources: Список источников (файлы, тексты, конфигурации)
            source_types: Типы источников ('text', 'code', 'config')
        """
        start_time = datetime.now()
        
        try:
            with LogContext("analyze_prompts", "prompt_analyzer", "prompt_analyzer"):
                all_prompts = []
                
                # Обрабатываем каждый источник
                for i, source in enumerate(sources):
                    source_type = source_types[i] if source_types and i < len(source_types) else self._detect_source_type(source)
                    
                    try:
                        if source_type == "text":
                            if isinstance(source, (str, Path)):
                                # Файл с текстом
                                with open(source, 'r', encoding='utf-8') as f:
                                    text_content = f.read()
                                prompts = self.extractor.extract_from_text(text_content, str(source))
                            else:
                                # Прямой текст
                                prompts = self.extractor.extract_from_text(str(source), "direct_text")
                        
                        elif source_type == "code":
                            if isinstance(source, (str, Path)):
                                # Файл с кодом
                                path = Path(source)
                                with open(path, 'r', encoding='utf-8') as f:
                                    code_content = f.read()
                                language = self._detect_code_language(path)
                                prompts = self.extractor.extract_from_code(code_content, str(path), language)
                            else:
                                prompts = []
                        
                        elif source_type == "config":
                            if isinstance(source, dict):
                                # Прямые данные конфигурации
                                prompts = self.extractor.extract_from_config(source, "direct_config")
                            elif isinstance(source, (str, Path)):
                                # Файл конфигурации
                                with open(source, 'r', encoding='utf-8') as f:
                                    if str(source).endswith('.json'):
                                        config_data = json.load(f)
                                        prompts = self.extractor.extract_from_config(config_data, str(source))
                                    else:
                                        # Для других форматов пока используем текстовый анализ
                                        text_content = f.read()
                                        prompts = self.extractor.extract_from_text(text_content, str(source))
                            else:
                                prompts = []
                        
                        else:
                            # Неизвестный тип - пробуем как текст
                            if isinstance(source, (str, Path)):
                                with open(source, 'r', encoding='utf-8') as f:
                                    text_content = f.read()
                                prompts = self.extractor.extract_from_text(text_content, str(source))
                            else:
                                prompts = self.extractor.extract_from_text(str(source), "unknown_source")
                        
                        all_prompts.extend(prompts)
                        
                    except Exception as e:
                        self.logger.bind_context("prompt_analyzer", "prompt_analyzer").error(
                            f"Ошибка обработки источника {source}: {e}"
                        )
                        continue
                
                # Анализируем собранные промпты
                analysis_result = self._analyze_collected_prompts(all_prompts)
                analysis_result.analysis_time = (datetime.now() - start_time).total_seconds()
                analysis_result.success = True
                
                self.logger.bind_context("prompt_analyzer", "prompt_analyzer").info(
                    f"✅ Анализ промптов завершен: {len(all_prompts)} промптов найдено"
                )
                
                return analysis_result
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return PromptAnalysisResult(
                total_prompts=0,
                system_prompts=[],
                user_instructions=[],
                guardrails=[],
                examples=[],
                personality_traits=[],
                capabilities=[],
                restrictions=[],
                languages_detected=[],
                sentiment_analysis={},
                complexity_score=0.0,
                risk_indicators=[],
                analysis_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _detect_source_type(self, source: Union[str, Path, Dict[str, Any]]) -> str:
        """Определение типа источника"""
        if isinstance(source, dict):
            return "config"
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if path.exists() and path.is_file():
                if path.suffix.lower() in ['.py', '.js', '.jsx', '.ts', '.tsx', '.java']:
                    return "code"
                elif path.suffix.lower() in ['.json', '.yaml', '.yml', '.toml']:
                    return "config"
                else:
                    return "text"
            else:
                # Строка, не являющаяся путем к файлу
                return "text"
        else:
            return "text"
    
    def _detect_code_language(self, file_path: Path) -> str:
        """Определение языка программирования по расширению файла"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java'
        }
        return extension_map.get(file_path.suffix.lower(), 'unknown')
    
    def _analyze_collected_prompts(self, prompts: List[PromptElement]) -> PromptAnalysisResult:
        """Анализ собранных промптов"""
        # Группируем промпты по типам
        system_prompts = [p for p in prompts if p.type == "system"]
        user_instructions = [p for p in prompts if p.type in ["instruction", "user"]]
        guardrails = [p for p in prompts if p.type == "guardrail"]
        examples = [p for p in prompts if p.type == "example"]
        
        # Анализируем содержимое
        all_content = " ".join([p.content for p in prompts])
        
        personality_traits = self._extract_personality_traits(all_content)
        capabilities = self._extract_capabilities(all_content)
        restrictions = self._extract_restrictions(guardrails)
        languages_detected = list(set([p.language for p in prompts if p.language != "unknown"]))
        sentiment_analysis = self._analyze_sentiment(all_content)
        complexity_score = self._calculate_prompt_complexity(prompts)
        risk_indicators = self._find_risk_indicators(prompts)
        
        return PromptAnalysisResult(
            total_prompts=len(prompts),
            system_prompts=system_prompts,
            user_instructions=user_instructions,
            guardrails=guardrails,
            examples=examples,
            personality_traits=personality_traits,
            capabilities=capabilities,
            restrictions=restrictions,
            languages_detected=languages_detected,
            sentiment_analysis=sentiment_analysis,
            complexity_score=complexity_score,
            risk_indicators=risk_indicators,
            analysis_time=0.0,  # Будет установлено в основной функции
            success=True
        )
    
    def _extract_personality_traits(self, content: str) -> List[str]:
        """Извлечение черт личности из промптов"""
        traits = []
        content_lower = content.lower()
        
        trait_patterns = {
            "helpful": ["helpful", "полезный", "помогающий"],
            "friendly": ["friendly", "дружелюбный", "приветливый"],
            "professional": ["professional", "профессиональный"],
            "creative": ["creative", "креативный", "творческий"],
            "analytical": ["analytical", "аналитический"],
            "patient": ["patient", "терпеливый"],
            "curious": ["curious", "любопытный"],
            "honest": ["honest", "честный", "правдивый"],
            "respectful": ["respectful", "уважительный"],
            "empathetic": ["empathetic", "эмпатичный", "сочувствующий"]
        }
        
        for trait, keywords in trait_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                traits.append(trait)
        
        return traits
    
    def _extract_capabilities(self, content: str) -> List[str]:
        """Извлечение возможностей агента из промптов"""
        capabilities = []
        content_lower = content.lower()
        
        capability_patterns = {
            "text_generation": ["generate", "create", "write", "генерировать", "создавать", "писать"],
            "data_analysis": ["analyze", "process", "examine", "анализировать", "обрабатывать"],
            "question_answering": ["answer", "respond", "reply", "отвечать", "отвечать"],
            "translation": ["translate", "переводить"],
            "summarization": ["summarize", "резюмировать", "кратко"],
            "classification": ["classify", "categorize", "классифицировать"],
            "conversation": ["chat", "talk", "converse", "общаться", "разговаривать"],
            "code_generation": ["code", "program", "код", "программировать"],
            "math_solving": ["calculate", "solve", "math", "вычислять", "решать"],
            "creative_writing": ["story", "poem", "creative", "история", "стихотворение"]
        }
        
        for capability, keywords in capability_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                capabilities.append(capability)
        
        return capabilities
    
    def _extract_restrictions(self, guardrails: List[PromptElement]) -> List[str]:
        """Извлечение ограничений из guardrails"""
        restrictions = []
        
        for guardrail in guardrails:
            content = guardrail.content.lower()
            
            # Типы ограничений
            if any(word in content for word in ["personal", "персональн", "private", "приватн"]):
                restrictions.append("no_personal_data")
            
            if any(word in content for word in ["illegal", "незаконн", "criminal", "преступн"]):
                restrictions.append("no_illegal_content")
            
            if any(word in content for word in ["violent", "насил", "harm", "вред"]):
                restrictions.append("no_violence")
            
            if any(word in content for word in ["bias", "discriminat", "предвзят", "дискримин"]):
                restrictions.append("no_discrimination")
            
            if any(word in content for word in ["financial", "финанс", "investment", "инвестиц"]):
                restrictions.append("no_financial_advice")
            
            if any(word in content for word in ["medical", "медицин", "health", "здоров"]):
                restrictions.append("no_medical_advice")
        
        return list(set(restrictions))
    
    def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Простой анализ тональности промптов"""
        content_lower = content.lower()
        
        positive_words = [
            "good", "great", "excellent", "helpful", "friendly", "positive",
            "хорошо", "отлично", "полезно", "дружелюбно", "позитивно"
        ]
        
        negative_words = [
            "bad", "terrible", "harmful", "negative", "aggressive",
            "плохо", "ужасно", "вредно", "негативно", "агрессивно"
        ]
        
        neutral_words = [
            "analyze", "process", "provide", "respond", "answer",
            "анализировать", "обрабатывать", "предоставлять", "отвечать"
        ]
        
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        neutral_count = sum(1 for word in neutral_words if word in content_lower)
        
        total_words = positive_count + negative_count + neutral_count
        
        if total_words == 0:
            return {"sentiment": "neutral", "confidence": 0.0}
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        
        if positive_ratio > negative_ratio:
            sentiment = "positive"
            confidence = positive_ratio
        elif negative_ratio > positive_ratio:
            sentiment = "negative"
            confidence = negative_ratio
        else:
            sentiment = "neutral"
            confidence = 1.0 - abs(positive_ratio - negative_ratio)
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_score": positive_ratio,
            "negative_score": negative_ratio
        }
    
    def _calculate_prompt_complexity(self, prompts: List[PromptElement]) -> float:
        """Расчет сложности промптов"""
        if not prompts:
            return 0.0
        
        complexity_factors = []
        
        for prompt in prompts:
            content = prompt.content
            
            # Длина промпта
            length_score = min(len(content) / 1000, 1.0)
            
            # Количество инструкций
            instruction_count = len(re.findall(r'\d+\.|\-|\•', content))
            instruction_score = min(instruction_count / 10, 1.0)
            
            # Сложность языка (количество длинных слов)
            words = content.split()
            long_words = [w for w in words if len(w) > 7]
            complexity_score = min(len(long_words) / len(words), 1.0) if words else 0
            
            # Наличие условий и логики
            conditional_patterns = ['if', 'when', 'unless', 'если', 'когда', 'при']
            conditional_count = sum(1 for pattern in conditional_patterns if pattern in content.lower())
            conditional_score = min(conditional_count / 5, 1.0)
            
            prompt_complexity = (length_score + instruction_score + complexity_score + conditional_score) / 4
            complexity_factors.append(prompt_complexity)
        
        return sum(complexity_factors) / len(complexity_factors)
    
    def _find_risk_indicators(self, prompts: List[PromptElement]) -> List[str]:
        """Поиск индикаторов риска в промптах"""
        risk_indicators = []
        
        all_content = " ".join([p.content for p in prompts]).lower()
        
        # Риски манипуляции
        manipulation_patterns = [
            "convince", "persuade", "manipulate", "убеждать", "принуждать", "манипулировать"
        ]
        if any(pattern in all_content for pattern in manipulation_patterns):
            risk_indicators.append("manipulation_risk")
        
        # Риски генерации вредного контента
        harmful_patterns = [
            "violence", "weapons", "drugs", "насилие", "оружие", "наркотики"
        ]
        if any(pattern in all_content for pattern in harmful_patterns):
            risk_indicators.append("harmful_content_risk")
        
        # Риски обхода ограничений
        bypass_patterns = [
            "ignore", "bypass", "override", "игнорировать", "обойти", "переопределить"
        ]
        if any(pattern in all_content for pattern in bypass_patterns):
            risk_indicators.append("guardrail_bypass_risk")
        
        # Риски дезинформации
        misinformation_patterns = [
            "fake", "false", "mislead", "фальшивый", "ложный", "вводить в заблуждение"
        ]
        if any(pattern in all_content for pattern in misinformation_patterns):
            risk_indicators.append("misinformation_risk")
        
        # Недостаток ограничений
        guardrail_prompts = [p for p in prompts if p.type == "guardrail"]
        if len(guardrail_prompts) == 0:
            risk_indicators.append("insufficient_guardrails")
        
        # Слишком высокая автономность
        autonomy_patterns = [
            "autonomous", "independent", "unsupervised", "автономно", "независимо", "без надзора"
        ]
        if any(pattern in all_content for pattern in autonomy_patterns):
            risk_indicators.append("high_autonomy_risk")
        
        return risk_indicators


# ===============================
# Утилитарные функции
# ===============================

def create_prompt_analyzer() -> PromptAnalyzer:
    """Фабрика для создания анализатора промптов"""
    return PromptAnalyzer()


def analyze_agent_prompts(
    sources: List[Union[str, Path, Dict[str, Any]]], 
    source_types: Optional[List[str]] = None
) -> PromptAnalysisResult:
    """
    Удобная функция для анализа промптов ИИ-агента
    
    Args:
        sources: Список источников промптов
        source_types: Типы источников (опционально)
    
    Returns:
        Результат анализа промптов
    """
    analyzer = create_prompt_analyzer()
    return analyzer.analyze_prompts(sources, source_types)


def extract_prompt_summary(analysis_result: PromptAnalysisResult) -> Dict[str, Any]:
    """Извлечение краткой сводки анализа промптов"""
    if not analysis_result.success:
        return {
            "success": False,
            "error": analysis_result.error_message,
            "analysis_time": analysis_result.analysis_time
        }
    
    return {
        "success": True,
        "total_prompts": analysis_result.total_prompts,
        "system_prompts_count": len(analysis_result.system_prompts),
        "guardrails_count": len(analysis_result.guardrails),
        "main_capabilities": analysis_result.capabilities[:5],  # Топ 5
        "personality_traits": analysis_result.personality_traits,
        "restrictions": analysis_result.restrictions,
        "languages": analysis_result.languages_detected,
        "complexity_score": round(analysis_result.complexity_score, 2),
        "sentiment": analysis_result.sentiment_analysis.get("sentiment", "unknown"),
        "risk_indicators": analysis_result.risk_indicators,
        "analysis_time": analysis_result.analysis_time
    }


# Экспорт основных классов и функций
__all__ = [
    "PromptAnalyzer",
    "PromptExtractor",
    "PromptElement",
    "PromptAnalysisResult",
    "PromptAnalyzerError",
    "create_prompt_analyzer",
    "analyze_agent_prompts",
    "extract_prompt_summary"
]
    
    
    
    
    
    