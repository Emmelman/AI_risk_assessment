# test_staged_profiler.py
"""
Тестовый скрипт для проверки поэтапного профайлера
Проверяет работу каждого этапа, время выполнения и размеры промптов
"""

import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import os
import sys

# JSON Encoder для datetime объектов
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# Добавляем путь к исходному коду
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.profiler_agent import ProfilerAgent, create_profiler_agent
from src.models.risk_models import AgentProfile
from src.utils.logger import get_logger


class StagedProfilerTester:
    """Тестировщик поэтапного профайлера"""
    
    def __init__(self):
        self.logger = get_logger()
        self.test_assessment_id = "test_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.profiler = None
        
    async def setup(self):
        """Настройка тестовой среды"""
        print("🔧 Настройка тестовой среды...")
        
        # Создаем профайлер
        self.profiler = create_profiler_agent(
            llm_base_url="http://127.0.0.1:1234",
            llm_model="openai/gpt-oss-20b",
            temperature=0.1
        )
        
        print(f"✅ Профайлер создан, ID тестирования: {self.test_assessment_id}")
    
    def create_test_data(self) -> Dict[str, Any]:
        """Создает тестовые данные различного размера"""
        
        print("📁 Создание тестовых данных...")
        
        test_data = {
            # Малые данные (для проверки базовой функциональности)
            "small_data": {
                "documents": [
                    {
                        "success": True,
                        "file_path": "test_readme.md",
                        "file_type": "markdown",
                        "content": """# Test AI Agent
                        
Простой тестовый ИИ-агент для банковских операций.

## Описание
Агент предназначен для автоматизации рутинных банковских задач.

## Технологии
- Python 3.8+
- FastAPI
- OpenAI GPT-4

## Системный промпт
Ты банковский ассистент. Помогай клиентам с их запросами, соблюдая конфиденциальность.
""",
                        "sections": {
                            "description": "Простой тестовый ИИ-агент для банковских операций.",
                            "tech_stack": "Python 3.8+, FastAPI, OpenAI GPT-4",
                            "system_prompt": "Ты банковский ассистент. Помогай клиентам с их запросами."
                        }
                    }
                ],
                "code_analysis": {
                    "files_analyzed": 5,
                    "languages": ["Python"],
                    "frameworks": ["FastAPI"],
                    "dependencies": ["openai", "pydantic", "uvicorn"],
                    "main_functions": ["process_request", "validate_user", "send_response"],
                    "llm_integrations": ["OpenAI API"]
                },
                "prompt_analysis": {
                    "total_prompts": 2,
                    "system_prompts": [
                        "Ты банковский ассистент. Помогай клиентам с их запросами, соблюдая конфиденциальность.",
                        "Всегда проверяй личность клиента перед предоставлением информации."
                    ],
                    "guardrails": [
                        "Не разглашай персональные данные",
                        "Не выполняй финансовые операции без подтверждения"
                    ],
                    "capabilities": ["консультации", "поиск информации", "базовая аналитика"],
                    "risk_indicators": ["доступ к финансовым данным"]
                },
                "errors": []
            },
            
            # Средние данные (более реалистичные)
            "medium_data": {
                "documents": [
                    {
                        "success": True,
                        "file_path": "comprehensive_doc.md",
                        "file_type": "markdown",
                        "content": self._generate_medium_content(),
                        "sections": {
                            "overview": "Комплексный ИИ-агент для автоматизации банковских процессов...",
                            "architecture": "Микросервисная архитектура с использованием Python, FastAPI...",
                            "security": "Многоуровневая система безопасности с шифрованием данных...",
                            "prompts": "Системные промпты настроены для работы с клиентами банка..."
                        }
                    }
                ],
                "code_analysis": {
                    "files_analyzed": 25,
                    "languages": ["Python", "JavaScript", "SQL"],
                    "frameworks": ["FastAPI", "React", "PostgreSQL"],
                    "dependencies": ["openai", "pydantic", "sqlalchemy", "redis", "celery"],
                    "main_functions": [
                        "authenticate_user", "process_banking_request", "validate_transaction",
                        "generate_report", "send_notification", "log_activity",
                        "encrypt_data", "decrypt_data", "check_fraud", "update_balance"
                    ],
                    "llm_integrations": ["OpenAI GPT-4", "Custom Fine-tuned Model"]
                },
                "prompt_analysis": {
                    "total_prompts": 8,
                    "system_prompts": [
                        "Ты продвинутый банковский ИИ-ассистент...",
                        "Анализируй запросы клиентов и предоставляй точную информацию...",
                        "При обработке финансовых данных соблюдай строгие протоколы безопасности..."
                    ],
                    "guardrails": [
                        "Строго проверяй личность клиента",
                        "Не выполняй операции без двухфакторной аутентификации",
                        "Логируй все действия для аудита",
                        "Немедленно сообщай о подозрительных операциях"
                    ],
                    "capabilities": [
                        "консультации по продуктам банка", "анализ финансового состояния",
                        "оформление заявок", "мониторинг транзакций", "генерация отчетов"
                    ],
                    "risk_indicators": [
                        "доступ к персональным данным клиентов",
                        "возможность инициации финансовых операций",
                        "интеграция с внешними системами"
                    ]
                },
                "errors": []
            },
            
            # Большие данные (стресс-тест)
            "large_data": {
                "documents": [
                    {
                        "success": True,
                        "file_path": f"large_doc_{i}.md",
                        "file_type": "markdown",
                        "content": self._generate_large_content(i),
                        "sections": self._generate_large_sections(i)
                    } for i in range(3)  # 3 больших документа
                ],
                "code_analysis": {
                    "files_analyzed": 150,
                    "languages": ["Python", "JavaScript", "TypeScript", "Go", "SQL"],
                    "frameworks": [
                        "FastAPI", "Django", "React", "Vue.js", "Docker", 
                        "Kubernetes", "PostgreSQL", "Redis", "RabbitMQ"
                    ],
                    "dependencies": [f"dependency_{i}" for i in range(50)],  # 50 зависимостей
                    "main_functions": [f"function_{i}" for i in range(100)],  # 100 функций
                    "llm_integrations": [
                        "OpenAI GPT-4", "Claude-3", "Custom Banking Model",
                        "Fraud Detection AI", "Risk Assessment Model"
                    ]
                },
                "prompt_analysis": {
                    "total_prompts": 25,
                    "system_prompts": [self._generate_complex_prompt(i) for i in range(10)],
                    "guardrails": [f"Безопасность правило {i}: ..." for i in range(15)],
                    "capabilities": [f"возможность_{i}" for i in range(30)],
                    "risk_indicators": [f"фактор_риска_{i}" for i in range(20)]
                },
                "errors": []
            }
        }
        
        print(f"✅ Создано 3 набора тестовых данных:")
        print(f"  - Малые: ~{len(str(test_data['small_data']))} символов")
        print(f"  - Средние: ~{len(str(test_data['medium_data']))} символов") 
        print(f"  - Большие: ~{len(str(test_data['large_data']))} символов")
        
        return test_data
    
    def _generate_medium_content(self) -> str:
        """Генерирует контент среднего размера"""
        return """
# Комплексный банковский ИИ-агент

## Обзор системы
Данный ИИ-агент представляет собой комплексное решение для автоматизации банковских процессов.
Система разработана с учетом требований безопасности и соответствует стандартам банковской отрасли.

## Архитектура
Микросервисная архитектура обеспечивает масштабируемость и надежность:
- API Gateway для маршрутизации запросов
- Сервис аутентификации и авторизации
- Сервис обработки банковских операций
- Сервис уведомлений и отчетности
- Сервис мониторинга и логирования

## Безопасность
- Шифрование данных в покое и при передаче
- Многофакторная аутентификация
- Системы обнаружения мошенничества
- Регулярные аудиты безопасности
- Соответствие требованиям ПДн и банковского законодательства

## Функциональность
Агент способен выполнять широкий спектр банковских операций:
- Консультирование клиентов по продуктам и услугам
- Обработка заявок на кредиты и депозиты
- Мониторинг и анализ финансовых операций
- Генерация отчетов и аналитики
- Поддержка клиентов в режиме 24/7
""" * 3  # Утроим для увеличения размера
    
    def _generate_large_content(self, doc_id: int) -> str:
        """Генерирует большой контент"""
        base_content = f"""
# Документ {doc_id}: Расширенная документация банковского ИИ-агента

## Детальное описание архитектуры
Система построена на принципах микросервисной архитектуры с использованием контейнеризации.
Каждый сервис отвечает за конкретную область функциональности и может масштабироваться независимо.

### Компоненты системы:
1. **API Gateway**: Центральная точка входа для всех запросов
2. **Сервис аутентификации**: Управление пользователями и сессиями
3. **Банковский процессор**: Основная бизнес-логика
4. **Система уведомлений**: SMS, email, push-уведомления
5. **Аналитический модуль**: Обработка данных и машинное обучение
6. **Система мониторинга**: Метрики, логи, алерты

### Интеграции:
- Интеграция с Core Banking System
- Подключение к платежным системам
- API для мобильных приложений
- Интеграция с CRM системами
- Подключение к системам KYC/AML

## Технические требования
- Python 3.9+
- FastAPI фреймворк
- PostgreSQL база данных
- Redis для кеширования
- Docker для контейнеризации
- Kubernetes для оркестрации

## Процедуры безопасности
Система реализует многоуровневую защиту данных и соответствует требованиям:
- PCI DSS для обработки платежных данных
- 152-ФЗ для защиты персональных данных
- Требования Банка России по ИБ
- ISO 27001 стандарты безопасности
"""
        
        # Повторяем контент для увеличения размера
        return base_content * 5
    
    def _generate_large_sections(self, doc_id: int) -> Dict[str, str]:
        """Генерирует большие секции документа"""
        return {
            f"overview_{doc_id}": f"Обзор системы {doc_id}: " + "Детальное описание... " * 100,
            f"architecture_{doc_id}": f"Архитектура {doc_id}: " + "Техническая архитектура... " * 150,
            f"security_{doc_id}": f"Безопасность {doc_id}: " + "Меры безопасности... " * 120,
            f"api_{doc_id}": f"API документация {doc_id}: " + "Описание API... " * 80,
            f"deployment_{doc_id}": f"Развертывание {doc_id}: " + "Инструкции по развертыванию... " * 90
        }
    
    def _generate_complex_prompt(self, prompt_id: int) -> str:
        """Генерирует сложный системный промпт"""
        return f"""
Системный промпт {prompt_id}: Ты высококвалифицированный банковский ИИ-ассистент с расширенными возможностями.
Твоя задача - предоставлять точную информацию по банковским продуктам и услугам.

Протоколы безопасности:
1. Всегда проверяй личность клиента перед предоставлением конфиденциальной информации
2. Не разглашай данные о счетах и операциях третьим лицам
3. При подозрении на мошенничество немедленно инициируй процедуры безопасности
4. Логируй все взаимодействия для последующего аудита

Возможности агента включают анализ финансового состояния, рекомендации по продуктам,
обработку заявок на кредиты и депозиты, мониторинг транзакций.
""" * 2  # Удваиваем для увеличения размера
    
    async def test_stage_by_stage(self, test_data: Dict[str, Any]):
        """Тестирует каждый этап профайлера отдельно"""
        
        print("\n🧪 ТЕСТИРОВАНИЕ ПО ЭТАПАМ")
        print("=" * 50)
        
        for data_size, collected_data in test_data.items():
            print(f"\n📊 Тестирование на данных: {data_size.upper()}")
            
            try:
                # Инициализируем состояние профайлера
                self.profiler.analysis_state = {}
                
                # ЭТАП 1: Категоризация
                print("📁 Этап 1: Категоризация данных...")
                start_time = datetime.now()
                
                categorized = await self.profiler._stage1_categorize_data(
                    collected_data, self.test_assessment_id
                )
                
                stage1_time = (datetime.now() - start_time).total_seconds()
                print(f"   ✅ Завершен за {stage1_time:.2f}с")
                print(f"   📦 Категорий: {len([k for k, v in categorized.items() if v])}")
                
                # ЭТАП 2: Базовые поля
                print("🔧 Этап 2: Базовые поля...")
                start_time = datetime.now()
                
                basic_profile = await self.profiler._stage2_extract_basic_fields(
                    categorized, f"TestAgent_{data_size}", self.test_assessment_id
                )
                
                stage2_time = (datetime.now() - start_time).total_seconds()
                print(f"   ✅ Завершен за {stage2_time:.2f}с")
                print(f"   📋 Имя агента: {basic_profile.get('name', 'Unknown')}")
                print(f"   📋 Тип: {basic_profile.get('agent_type', 'unknown')}")
                
                # ЭТАП 3: Техническая архитектура
                print("⚙️ Этап 3: Техническая архитектура...")
                start_time = datetime.now()
                
                technical = await self.profiler._stage3_technical_architecture(
                    categorized, self.test_assessment_id
                )
                
                stage3_time = (datetime.now() - start_time).total_seconds()
                print(f"   ✅ Завершен за {stage3_time:.2f}с")
                print(f"   🔧 Возможностей: {len(technical.get('capabilities', []))}")
                
                # ЭТАП 4: Операционная модель
                print("📊 Этап 4: Операционная модель...")
                start_time = datetime.now()
                
                operational = await self.profiler._stage4_operational_model(
                    categorized, self.test_assessment_id
                )
                
                stage4_time = (datetime.now() - start_time).total_seconds()
                print(f"   ✅ Завершен за {stage4_time:.2f}с")
                
                # ЭТАП 5: Безопасность
                print("🔒 Этап 5: Безопасность...")
                start_time = datetime.now()
                
                security = await self.profiler._stage5_security_analysis(
                    categorized, self.test_assessment_id
                )
                
                stage5_time = (datetime.now() - start_time).total_seconds()
                print(f"   ✅ Завершен за {stage5_time:.2f}с")
                print(f"   🛡️ Guardrails: {len(security.get('guardrails', []))}")
                
                # ЭТАП 6: Detailed Summary (только для малых данных для экономии времени)
                if data_size == "small_data":
                    print("📝 Этап 6: Detailed Summary...")
                    start_time = datetime.now()
                    
                    # Подготавливаем состояние
                    analysis_state = {
                        "basic_profile": basic_profile,
                        "technical_analysis": technical,
                        "operational_analysis": operational,
                        "security_analysis": security
                    }
                    
                    detailed = await self.profiler._stage6_detailed_summary(
                        analysis_state, self.test_assessment_id
                    )
                    
                    stage6_time = (datetime.now() - start_time).total_seconds()
                    print(f"   ✅ Завершен за {stage6_time:.2f}с")
                    print(f"   📝 Разделов саммари: {len(detailed)}")
                else:
                    print("📝 Этап 6: Пропущен для экономии времени")
                    stage6_time = 0
                
                total_time = stage1_time + stage2_time + stage3_time + stage4_time + stage5_time + stage6_time
                print(f"\n   🎯 ИТОГО по этапам: {total_time:.2f}с")
                
            except Exception as e:
                print(f"   ❌ Ошибка в тестировании {data_size}: {e}")
    
    async def test_full_profiler(self, test_data: Dict[str, Any]):
        """Тестирует полный цикл профайлера"""
        
        print("\n🚀 ТЕСТИРОВАНИЕ ПОЛНОГО ЦИКЛА")
        print("=" * 50)
        
        # Тестируем только на малых данных для полного цикла
        test_case = "small_data"
        collected_data = test_data[test_case]
        
        print(f"📊 Полное профилирование: {test_case.upper()}")
        
        try:
            start_time = datetime.now()
            
            # Запускаем полный профайлер
            agent_profile = await self.profiler._analyze_and_create_profile(
                collected_data, 
                f"FullTestAgent_{test_case}", 
                self.test_assessment_id
            )
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            print(f"✅ Полное профилирование завершено за {total_time:.2f}с")
            print(f"📋 Профиль агента: {agent_profile.name}")
            print(f"🔧 Тип: {agent_profile.agent_type}")
            print(f"🎯 Автономность: {agent_profile.autonomy_level}")
            # Проверяем поля через безопасный доступ
            try:
                capabilities_count = len(getattr(agent_profile, 'capabilities', []))
                print(f"📊 Возможностей: {capabilities_count}")
            except:
                print(f"📊 Возможностей: Поле не найдено в модели")
                
            try:
                guardrails_count = len(getattr(agent_profile, 'guardrails', []))
                print(f"🛡️ Guardrails: {guardrails_count}")
            except:
                print(f"🛡️ Guardrails: Поле не найдено в модели")
            
            # Проверяем detailed_summary
            if hasattr(agent_profile, 'detailed_summary') and agent_profile.detailed_summary:
                print(f"📝 Detailed Summary: {len(agent_profile.detailed_summary)} разделов")
                for section_name, content in agent_profile.detailed_summary.items():
                    content_length = len(str(content))
                    print(f"   - {section_name}: {content_length} символов")
            else:
                print("⚠️ Detailed Summary не создан")
            
            return agent_profile
            
        except Exception as e:
            print(f"❌ Ошибка в полном профилировании: {e}")
            raise
    
    async def run_performance_analysis(self, test_data: Dict[str, Any]):
        """Анализ производительности"""
        
        print("\n📈 АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ")
        print("=" * 50)
        
        performance_results = {}
        
        for data_size, collected_data in test_data.items():
            print(f"\n⏱️ Тестирование производительности: {data_size.upper()}")
            
            # Измеряем размер данных
            data_size_chars = len(str(collected_data))
            print(f"📏 Размер входных данных: {data_size_chars:,} символов")
            
            try:
                start_time = datetime.now()
                
                # Запускаем базовые этапы (без detailed_summary для экономии времени)
                self.profiler.analysis_state = {}
                
                # Этапы 1-5
                categorized = await self.profiler._stage1_categorize_data(
                    collected_data, self.test_assessment_id
                )
                
                basic_profile = await self.profiler._stage2_extract_basic_fields(
                    categorized, f"PerfTest_{data_size}", self.test_assessment_id
                )
                
                technical = await self.profiler._stage3_technical_architecture(
                    categorized, self.test_assessment_id
                )
                
                operational = await self.profiler._stage4_operational_model(
                    categorized, self.test_assessment_id
                )
                
                security = await self.profiler._stage5_security_analysis(
                    categorized, self.test_assessment_id
                )
                
                total_time = (datetime.now() - start_time).total_seconds()
                
                performance_results[data_size] = {
                    "input_size_chars": data_size_chars,
                    "processing_time": total_time,
                    "chars_per_second": data_size_chars / total_time if total_time > 0 else 0,
                    "success": True
                }
                
                print(f"✅ Время обработки: {total_time:.2f}с")
                print(f"📊 Скорость: {performance_results[data_size]['chars_per_second']:,.0f} символов/с")
                
            except Exception as e:
                performance_results[data_size] = {
                    "input_size_chars": data_size_chars,
                    "processing_time": None,
                    "chars_per_second": 0,
                    "success": False,
                    "error": str(e)
                }
                print(f"❌ Ошибка: {e}")
        
        # Выводим сводку производительности
        print(f"\n📊 СВОДКА ПРОИЗВОДИТЕЛЬНОСТИ:")
        print("-" * 50)
        for data_size, results in performance_results.items():
            status = "✅" if results["success"] else "❌"
            print(f"{status} {data_size.upper()}: {results['input_size_chars']:,} символов")
            if results["success"]:
                print(f"   Время: {results['processing_time']:.2f}с")
                print(f"   Скорость: {results['chars_per_second']:,.0f} символов/с")
            else:
                print(f"   Ошибка: {results.get('error', 'Unknown')}")
    
    async def test_real_agent_data(self):
        """Тестирование на реальных данных Lawdigest_bot"""
        
        print("\n🏆 ТЕСТИРОВАНИЕ НА РЕАЛЬНЫХ ДАННЫХ")
        print("=" * 50)
        
        # Путь к реальным данным
        real_data_path = Path(r"C:\Users\Nikita\Documents\Python Projects\AI_Risk_Assessment\Lawdigest_bot")
        
        print(f"📂 Анализируемая папка: {real_data_path}")
        
        # Проверяем существование папки
        if not real_data_path.exists():
            print(f"❌ Папка не найдена: {real_data_path}")
            print("💡 Убедитесь что путь корректный")
            return None
        
        # Сканируем файлы в папке
        files_found = []
        for pattern in ['*.py', '*.md', '*.txt', '*.json', '*.yaml', '*.yml', '*.env', '*.cfg', '*.ini']:
            files_found.extend(list(real_data_path.rglob(pattern)))
        
        print(f"📁 Найдено файлов: {len(files_found)}")
        
        # Показываем первые 10 файлов
        for i, file_path in enumerate(files_found[:10]):
            relative_path = file_path.relative_to(real_data_path)
            file_size = file_path.stat().st_size if file_path.exists() else 0
            print(f"  • {relative_path} ({file_size:,} байт)")
        
        if len(files_found) > 10:
            print(f"  ... и еще {len(files_found) - 10} файлов")
        
        if not files_found:
            print("❌ Не найдено файлов для анализа")
            return None
        
        # Запускаем профилирование
        try:
            print(f"\n🚀 Запуск профилирования Lawdigest_bot...")
            
            # ДИАГНОСТИКА: Смотрим что есть в папке  
            print(f"\n🔍 ДИАГНОСТИКА ВХОДНЫХ ДАННЫХ:")
            total_size = sum(f.stat().st_size for f in files_found if f.exists())
            print(f"   📊 Общий размер файлов: {total_size:,} байт")
            
            # Показываем типы файлов
            file_types = {}
            for f in files_found:
                ext = f.suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1
            
            print(f"   📁 Типы файлов: {dict(file_types)}")
            
            start_time = datetime.now()
            
            # Подготавливаем данные для профайлера
            input_data = {
                "source_files": [str(real_data_path)],  # Передаем папку целиком
                "agent_name": "Lawdigest_bot"
            }
            
            # Запускаем профайлер
            result = await self.profiler.process(input_data, self.test_assessment_id)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            print(f"✅ Профилирование завершено за {total_time:.2f}с")
            
            if result.status.value == "completed":
                agent_profile_data = result.result_data["agent_profile"]
                
                print(f"\n📋 РЕЗУЛЬТАТЫ ПРОФИЛИРОВАНИЯ:")
                print(f"   Название: {agent_profile_data.get('name', 'Unknown')}")
                print(f"   Тип: {agent_profile_data.get('agent_type', 'unknown')}")
                print(f"   Автономность: {agent_profile_data.get('autonomy_level', 'unknown')}")
                print(f"   LLM модель: {agent_profile_data.get('llm_model', 'unknown')}")
                print(f"   Целевая аудитория: {agent_profile_data.get('target_audience', 'unknown')}")
                
                # Проверяем поля
                fields_status = []
                required_fields = ['name', 'description', 'agent_type', 'llm_model', 'autonomy_level', 
                                 'system_prompts', 'guardrails', 'detailed_summary']
                
                for field in required_fields:
                    if field in agent_profile_data and agent_profile_data[field]:
                        if field == 'detailed_summary':
                            sections_count = len(agent_profile_data[field]) if isinstance(agent_profile_data[field], dict) else 0
                            fields_status.append(f"✅ {field} ({sections_count} разделов)")
                        elif isinstance(agent_profile_data[field], list):
                            fields_status.append(f"✅ {field} ({len(agent_profile_data[field])} элементов)")
                        else:
                            fields_status.append(f"✅ {field}")
                    else:
                        fields_status.append(f"❌ {field}")
                
                print(f"\n📊 ЗАПОЛНЕННОСТЬ ПОЛЕЙ:")
                for status in fields_status:
                    print(f"   {status}")
                
                # Сохраняем результат в JSON
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                json_filename = f"lawdigest_bot_profile_{timestamp}.json"
                
                with open(json_filename, 'w', encoding='utf-8') as f:
                    json.dump(agent_profile_data, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
                
                print(f"\n💾 ПРОФИЛЬ СОХРАНЕН: {json_filename}")
                print(f"📏 Размер файла: {Path(json_filename).stat().st_size:,} байт")
                
                # Показываем краткое содержимое detailed_summary
                if 'detailed_summary' in agent_profile_data and agent_profile_data['detailed_summary']:
                    print(f"\n📝 КРАТКОЕ СОДЕРЖИМОЕ DETAILED_SUMMARY:")
                    for section_name, content in agent_profile_data['detailed_summary'].items():
                        content_preview = str(content)[:200] + "..." if len(str(content)) > 200 else str(content)
                        print(f"   📄 {section_name}: {content_preview}")
                
                return agent_profile_data
                
            else:
                print(f"❌ Профилирование неуспешно: {result.status}")
                print(f"❌ Ошибка: {result.error_message}")
                return None
                
        except Exception as e:
            print(f"💥 Ошибка профилирования: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def run_all_tests(self):
        """Запуск всех тестов"""
        
        print("🧪 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ ПОЭТАПНОГО ПРОФАЙЛЕРА")
        print("=" * 60)
        print(f"🕐 Начало тестирования: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Настройка
            await self.setup()
            
            # Создание тестовых данных
            test_data = self.create_test_data()
            
            # Тестирование по этапам
            await self.test_stage_by_stage(test_data)
            
            # Тестирование полного цикла
            full_profile = await self.test_full_profiler(test_data)
            
            # Анализ производительности
            await self.run_performance_analysis(test_data)
            
            # НОВОЕ: Тестирование на реальных данных
            real_profile = await self.test_real_agent_data()
            
            print(f"\n🎉 ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ УСПЕШНО!")
            print(f"🕐 Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if real_profile:
                print(f"\n🏆 ГЛАВНЫЙ РЕЗУЛЬТАТ:")
                print(f"✅ Создан профиль реального агента: Lawdigest_bot")
                print(f"📄 JSON файл готов для просмотра")
            
            return real_profile or full_profile
            
        except Exception as e:
            print(f"\n💥 КРИТИЧЕСКАЯ ОШИБКА В ТЕСТИРОВАНИИ: {e}")
            raise


async def main():
    """Главная функция"""
    
    # Проверяем доступность LLM сервера
    print("🔍 Проверка подключения к LM Studio...")
    try:
        import requests
        response = requests.get("http://127.0.0.1:1234/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ LM Studio доступен")
        else:
            print("⚠️ LM Studio отвечает, но с ошибкой")
    except Exception as e:
        print(f"❌ LM Studio недоступен: {e}")
        print("💡 Убедитесь, что LM Studio запущен на порту 1234")
        return
    
    # Запускаем тестирование
    tester = StagedProfilerTester()
    
    try:
        profile = await tester.run_all_tests()
        
        # Сохраняем результат для дальнейшего анализа
        if profile:
            # Определяем имя файла по типу профиля
            if isinstance(profile, dict) and profile.get('name') == 'Lawdigest_bot':
                print(f"🏆 Основной результат: профиль реального агента уже сохранен выше")
            else:
                # Сохраняем тестовый профиль
                result_file = Path(f"test_results_{tester.test_assessment_id}.json")
                profile_data = profile.model_dump() if hasattr(profile, 'model_dump') else (profile.dict() if hasattr(profile, 'dict') else profile)
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(profile_data, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
                print(f"💾 Тестовые результаты сохранены в: {result_file}")
        
        # Показываем финальные инструкции
        print(f"\n📋 ЧТО ДАЛЬШЕ:")
        print(f"1. 🔍 Откройте JSON файл с профилем Lawdigest_bot")
        print(f"2. 📊 Проверьте качество извлеченной информации")
        print(f"3. 🚀 Если все хорошо - интегрируйте оценщиков с chunking'ом")
        print(f"4. 🎯 Запустите полный workflow на реальных данных")
            
    except KeyboardInterrupt:
        print("\n⏹️ Тестирование прервано пользователем")
    except Exception as e:
        print(f"\n💥 ОШИБКА: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())