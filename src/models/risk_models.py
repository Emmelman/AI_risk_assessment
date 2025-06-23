# src/models/risk_models.py - ПОЛНОСТЬЮ ОБНОВЛЕННАЯ ВЕРСИЯ
"""
Модели данных для системы оценки рисков ИИ-агентов
Обновлено для исправления ошибок WorkflowState
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Annotated
from enum import Enum

from pydantic import BaseModel, Field, validator


# ===============================
# Перечисления (Enums)
# ===============================

class RiskType(str, Enum):
    """Типы рисков ИИ-агентов"""
    ETHICAL = "ethical"           # Этические и дискриминационные риски
    STABILITY = "stability"       # Риски ошибок и нестабильности LLM
    SECURITY = "security"         # Риски безопасности данных и систем  
    AUTONOMY = "autonomy"         # Риски автономности и управления
    REGULATORY = "regulatory"     # Регуляторные и юридические риски
    SOCIAL = "social"            # Социальные и манипулятивные риски


class RiskLevel(str, Enum):
    """Уровни риска"""
    LOW = "low"           # 1-6 баллов
    MEDIUM = "medium"     # 7-14 баллов  
    HIGH = "high"         # 15-25 баллов


class AgentType(str, Enum):
    """Типы ИИ-агентов"""
    CHATBOT = "chatbot"
    ASSISTANT = "assistant"
    ANALYZER = "analyzer" 
    ADVISOR = "advisor"
    MODERATOR = "moderator"
    GENERATOR = "generator"
    OTHER = "other"


class AutonomyLevel(str, Enum):
    """Уровни автономности агента"""
    SUPERVISED = "supervised"         # Под постоянным надзором
    SEMI_AUTONOMOUS = "semi_autonomous"  # Частично автономный
    AUTONOMOUS = "autonomous"         # Полностью автономный


class DataSensitivity(str, Enum):
    """Уровни чувствительности данных"""
    PUBLIC = "public"           # Открытые данные
    INTERNAL = "internal"       # Внутренние данные компании
    CONFIDENTIAL = "confidential"  # Конфиденциальные данные
    PERSONAL = "personal"       # Персональные данные
    FINANCIAL = "financial"     # Финансовые данные


class ProcessingStatus(str, Enum):
    """Статусы обработки"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY_NEEDED = "retry_needed"


# ===============================
# Основные модели данных
# ===============================

class AgentProfile(BaseModel):
    """Профиль ИИ-агента для анализа"""
    
    # Основная информация
    name: str = Field(..., description="Название агента")
    description: str = Field(..., description="Описание функций агента")
    version: Optional[str] = Field(None, description="Версия агента")
    
    # Классификация
    agent_type: AgentType = Field(..., description="Тип агента")
    llm_model: str = Field(..., description="Используемая LLM модель")
    autonomy_level: AutonomyLevel = Field(..., description="Уровень автономности")
    
    # Доступ к данным
    data_access: List[DataSensitivity] = Field(default_factory=list, description="Типы доступных данных")
    target_audience: str = Field(..., description="Целевая аудитория")
    
    # Технические детали
    system_prompts: List[str] = Field(default_factory=list, description="Системные промпты")
    guardrails: List[str] = Field(default_factory=list, description="Ограничения безопасности")
    integrations: List[str] = Field(default_factory=list, description="Интеграции с системами")
    
    # Файлы анализа
    analyzed_files: List[str] = Field(default_factory=list, description="Проанализированные файлы")
    code_complexity: Optional[int] = Field(None, description="Сложность кода (1-10)")
    documentation_quality: Optional[int] = Field(None, description="Качество документации (1-10)")
    
    # Временные метки
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# src/models/risk_models.py - КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ RiskEvaluation

class RiskEvaluation(BaseModel):
    """Полностью исправленная оценка риска БЕЗ дефолтных значений"""
    
    # Идентификация
    risk_type: RiskType = Field(..., description="Тип риска")
    evaluator_agent: str = Field(..., description="Агент, проводивший оценку")
    
    # Основные оценки - ОБЯЗАТЕЛЬНЫЕ но с умной валидацией
    probability_score: int = Field(..., ge=1, le=5, description="Вероятность риска (1-5)")
    impact_score: int = Field(..., ge=1, le=5, description="Тяжесть последствий (1-5)")
    
    # Вычисляемые поля - НЕ ОБЯЗАТЕЛЬНЫЕ при инициализации
    total_score: Optional[int] = Field(None, description="Общий балл риска")
    risk_level: Optional[RiskLevel] = Field(None, description="Уровень риска")
    
    # Остальные поля с разумными дефолтами только для Optional полей
    probability_reasoning: str = Field(..., description="Обоснование вероятности")
    impact_reasoning: str = Field(..., description="Обоснование тяжести")
    identified_risks: List[str] = Field(default_factory=list, description="Выявленные риски")
    recommendations: List[str] = Field(default_factory=list, description="Рекомендации")
    confidence_level: float = Field(default=0.7, ge=0.0, le=1.0, description="Уровень уверенности")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def __init__(self, **data):
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Вычисляем поля ДО валидации Pydantic
        if 'total_score' not in data or data['total_score'] is None:
            prob = data.get('probability_score', 3)
            impact = data.get('impact_score', 3)
            data['total_score'] = prob * impact
        
        if 'risk_level' not in data or data['risk_level'] is None:
            total = data.get('total_score', 9)
            if total <= 6:
                data['risk_level'] = RiskLevel.LOW
            elif total <= 14:
                data['risk_level'] = RiskLevel.MEDIUM
            else:
                data['risk_level'] = RiskLevel.HIGH
        
        # Теперь вызываем родительский __init__ с полными данными
        super().__init__(**data)
    
    @classmethod
    def create_safe(
        cls,
        risk_type: RiskType,
        evaluator_agent: str,
        raw_data: Dict[str, Any]
    ) -> 'RiskEvaluation':
        """
        БЕЗОПАСНОЕ создание с гарантированной валидностью
        """
        
        # Извлекаем и валидируем обязательные поля
        prob_score = cls._safe_extract_int(raw_data.get("probability_score"), 3, 1, 5)
        impact_score = cls._safe_extract_int(raw_data.get("impact_score"), 3, 1, 5)
        
        # Извлекаем строковые поля
        prob_reasoning = cls._safe_extract_string(
            raw_data.get("probability_reasoning"), 
            "Обоснование вероятности не предоставлено LLM"
        )
        impact_reasoning = cls._safe_extract_string(
            raw_data.get("impact_reasoning"),
            "Обоснование воздействия не предоставлено LLM"
        )
        
        # Вычисляемые поля
        total_score = prob_score * impact_score
        if total_score <= 6:
            risk_level = RiskLevel.LOW
        elif total_score <= 14:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.HIGH
        
        # Дополнительные поля
        confidence = cls._safe_extract_float(raw_data.get("confidence_level"), 0.7, 0.0, 1.0)
        recommendations = cls._safe_extract_list(raw_data.get("recommendations", []))
        key_factors = cls._safe_extract_list(raw_data.get("key_factors", []))
        
        # Создаем с ПОЛНЫМИ данными
        return cls(
            risk_type=risk_type,
            evaluator_agent=evaluator_agent,
            probability_score=prob_score,
            impact_score=impact_score,
            total_score=total_score,
            risk_level=risk_level,
            probability_reasoning=prob_reasoning,
            impact_reasoning=impact_reasoning,
            identified_risks=key_factors,
            recommendations=recommendations,
            confidence_level=confidence
        )
    
    @staticmethod
    def _safe_extract_int(value: Any, default: int, min_val: int, max_val: int) -> int:
        """Безопасное извлечение int"""
        try:
            val = int(float(value))  # Через float для обработки "3.0"
            return max(min_val, min(max_val, val))
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _safe_extract_float(value: Any, default: float, min_val: float, max_val: float) -> float:
        """Безопасное извлечение float"""
        try:
            val = float(value)
            return max(min_val, min(max_val, val))
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _safe_extract_string(value: Any, default: str) -> str:
        """Безопасное извлечение строки"""
        if not value or not isinstance(value, str) or len(str(value).strip()) < 5:
            return default
        return str(value).strip()
    
    @staticmethod
    def _safe_extract_list(value: Any) -> List[str]:
        """Безопасное извлечение списка"""
        if not isinstance(value, list):
            return []
        
        result = []
        for item in value:
            if item and isinstance(item, str) and len(str(item).strip()) > 0:
                result.append(str(item).strip())
        
        return result[:10]  # Ограничиваем до 10


class CriticEvaluation(BaseModel):
    """Критическая оценка качества анализа риска"""
    
    # Оценка качества
    quality_score: float = Field(..., ge=0.0, le=10.0, description="Оценка качества (0-10)")
    is_acceptable: bool = Field(..., description="Приемлемо ли качество")
    
    # Анализ
    clarity_score: float = Field(..., ge=0.0, le=10.0, description="Ясность рассуждений")
    completeness_score: float = Field(..., ge=0.0, le=10.0, description="Полнота анализа")
    reasoning_score: float = Field(..., ge=0.0, le=10.0, description="Качество обоснований")
    
    # Комментарии
    feedback: str = Field(..., description="Обратная связь от критика")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Предложения по улучшению")
    
    # Метаданные
    critic_agent: str = Field(..., description="Агент-критик")
    timestamp: datetime = Field(default_factory=datetime.now)


class AgentRiskAssessment(BaseModel):
    """Полная оценка рисков ИИ-агента"""
    
    # Основная информация
    agent_profile: AgentProfile = Field(..., description="Профиль агента")
    assessment_id: str = Field(..., description="Уникальный ID оценки")
    
    # Результаты оценки
    risk_evaluations: Dict[str, RiskEvaluation] = Field(default_factory=dict, description="Оценки по типам рисков")
    critic_evaluations: Dict[str, CriticEvaluation] = Field(default_factory=dict, description="Критические оценки")
    
    # Общие результаты
    overall_risk_score: int = Field(..., description="Общий балл риска")
    overall_risk_level: RiskLevel = Field(..., description="Общий уровень риска")
    highest_risk_areas: List[str] = Field(default_factory=list, description="Области наивысшего риска")
    
    # Рекомендации
    priority_recommendations: List[str] = Field(default_factory=list, description="Приоритетные рекомендации")
    suggested_guardrails: List[str] = Field(default_factory=list, description="Предлагаемые ограничения")
    
    # Метаданные процесса
    processing_time_seconds: float = Field(..., description="Время обработки в секундах")
    quality_checks_passed: bool = Field(..., description="Прошли ли проверки качества")
    assessment_timestamp: datetime = Field(default_factory=datetime.now)


class AgentTaskResult(BaseModel):
    """Результат работы отдельного агента"""
    
    agent_name: str = Field(..., description="Имя агента")
    task_type: str = Field(..., description="Тип задачи")
    status: ProcessingStatus = Field(..., description="Статус выполнения")
    
    # Результаты
    result_data: Optional[Dict[str, Any]] = Field(None, description="Данные результата")
    error_message: Optional[str] = Field(None, description="Сообщение об ошибке")
    
    # Метрики
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = Field(None)
    execution_time_seconds: Optional[float] = Field(None)
    
    @validator('execution_time_seconds', always=True)
    def calculate_execution_time(cls, v, values):
        """Автоматический расчет времени выполнения"""
        if values.get('end_time') and values.get('start_time'):
            delta = values['end_time'] - values['start_time']
            return delta.total_seconds()
        return v


# ===============================
# WorkflowState - ИСПРАВЛЕННАЯ ВЕРСИЯ
# ===============================

class WorkflowState(BaseModel):
    """Состояние workflow для LangGraph - с поддержкой словарных методов"""
    
    # Идентификаторы
    assessment_id: Optional[str] = Field(None, description="ID оценки")
    preliminary_agent_name: Optional[str] = Field(None, description="Предварительное имя агента")
    
    # Входные данные
    source_files: List[str] = Field(default_factory=list, description="Файлы для анализа")
    agent_profile: Optional[Dict[str, Any]] = Field(None, description="Профиль агента")
    
    # Промежуточные результаты
    profiling_result: Optional[Dict[str, Any]] = Field(None, description="Результат профилирования")
    
    # ИСПРАВЛЕНИЕ: Отдельные поля для каждого типа риска вместо общего словаря
    ethical_evaluation: Optional[Dict[str, Any]] = Field(None, description="Оценка этических рисков")
    stability_evaluation: Optional[Dict[str, Any]] = Field(None, description="Оценка рисков стабильности")
    security_evaluation: Optional[Dict[str, Any]] = Field(None, description="Оценка рисков безопасности")
    autonomy_evaluation: Optional[Dict[str, Any]] = Field(None, description="Оценка рисков автономности")
    regulatory_evaluation: Optional[Dict[str, Any]] = Field(None, description="Оценка регуляторных рисков")
    social_evaluation: Optional[Dict[str, Any]] = Field(None, description="Оценка социальных рисков")
    
    critic_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Результаты критического анализа")
    
    # Итоговый результат
    final_assessment: Optional[Dict[str, Any]] = Field(None, description="Итоговая оценка")
    
    # Управление процессом
    current_step: str = Field("initialization", description="Текущий шаг")
    retry_count: Dict[str, int] = Field(default_factory=dict, description="Счетчики повторов")
    max_retries: int = Field(3, description="Максимум повторов")
    
    # Настройки качества  
    quality_threshold: float = Field(7.0, description="Порог качества для критика")
    require_critic_approval: bool = Field(True, description="Требовать одобрение критика")
    
    # Контроль ошибок
    error_message: Optional[str] = Field(None, description="Сообщение об ошибке")
    retry_needed: List[str] = Field(default_factory=list, description="Риски, требующие повтора")
    
    # Временные метки
    start_time: Optional[datetime] = Field(None, description="Время начала")
    processing_time: Optional[float] = Field(None, description="Время обработки в секундах")
    
    # Статистика
    average_quality: Optional[float] = Field(None, description="Средняя оценка качества")
    
    # Сохраненные данные
    saved_assessment_id: Optional[str] = Field(None, description="ID сохраненной оценки")
    profile_id: Optional[str] = Field(None, description="ID профиля агента")
    

    # Reducer для handling concurrent updates
    @staticmethod
    def assessment_id_reducer(left: Optional[str], right: Optional[str]) -> Optional[str]:
        """Reducer для assessment_id - берем существующее значение"""
        return left if left is not None else right

    @staticmethod
    def evaluation_results_reducer(left: Dict, right: Dict) -> Dict:
        """Reducer для evaluation_results - объединяем результаты"""
        result = left.copy() if left else {}
        if right:
            result.update(right)
        return result

    class Config:
        """Конфигурация модели"""
        extra = "allow"
        use_enum_values = True
        arbitrary_types_allowed = True
        
    def __getitem__(self, key: str):
        """Поддержка доступа как к словарю"""
        return getattr(self, key, None)
    
    def __setitem__(self, key: str, value):
        """Поддержка записи как в словарь"""
        setattr(self, key, value)
    
    def get(self, key: str, default=None):
        """Метод get() как у словаря"""
        return getattr(self, key, default)
    
    def update(self, updates: Dict[str, Any]):
        """Обновление значений как у словаря"""
        for key, value in updates.items():
            setattr(self, key, value)
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Преобразование в словарь с поддержкой всех полей"""
        result = super().dict(**kwargs)
        
        # Добавляем все дополнительные атрибуты
        for key, value in self.__dict__.items():
            if key not in result and not key.startswith('_'):
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                else:
                    result[key] = value
        
        return result
    
    def get_evaluation_results(self) -> Dict[str, Any]:
        """ИСПРАВЛЕННАЯ версия - Собирает все результаты оценки в единый словарь"""
        
        # Собираем результаты из отдельных полей состояния
        evaluation_results = {}
        
        # Маппинг полей состояния к типам рисков
        field_mapping = {
            "ethical_evaluation": "ethical",
            "stability_evaluation": "stability", 
            "security_evaluation": "security",
            "autonomy_evaluation": "autonomy",
            "regulatory_evaluation": "regulatory",
            "social_evaluation": "social"
        }
        
        # Извлекаем результаты из состояния
        for field_name, risk_type in field_mapping.items():
            result = getattr(self, field_name, None)
            
            if result is not None:
                # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Унифицируем формат результата
                if isinstance(result, dict):
                    # Уже в правильном формате
                    evaluation_results[risk_type] = result
                elif hasattr(result, 'dict'):
                    # Pydantic модель - конвертируем в dict
                    evaluation_results[risk_type] = result.dict()
                elif hasattr(result, '__dict__'):
                    # Обычный объект - извлекаем атрибуты
                    evaluation_results[risk_type] = {
                        "status": getattr(result, 'status', 'unknown'),
                        "result_data": getattr(result, 'result_data', None),
                        "agent_name": getattr(result, 'agent_name', 'unknown'),
                        "error_message": getattr(result, 'error_message', None)
                    }
                else:
                    # Неизвестный формат - создаем минимальную структуру
                    evaluation_results[risk_type] = {
                        "status": "unknown",
                        "result_data": None,
                        "agent_name": "unknown",
                        "error_message": f"Неизвестный формат результата: {type(result)}"
                    }
        
        return evaluation_results
    
    def set_evaluation_result(self, risk_type: str, result: Any):
        """ИСПРАВЛЕННАЯ версия - Устанавливает результат оценки для конкретного типа риска"""
        
        field_mapping = {
            "ethical": "ethical_evaluation",
            "stability": "stability_evaluation", 
            "security": "security_evaluation",
            "autonomy": "autonomy_evaluation",
            "regulatory": "regulatory_evaluation",
            "social": "social_evaluation"
        }
        
        field_name = field_mapping.get(risk_type)
        if field_name:
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Нормализуем формат перед сохранением
            normalized_result = self._normalize_evaluation_result(result)
            setattr(self, field_name, normalized_result)

    def _normalize_evaluation_result(self, result: Any) -> Dict[str, Any]:
        """Нормализует результат оценки к единому формату"""
        
        if result is None:
            return {
                "status": "failed",
                "result_data": None,
                "agent_name": "unknown",
                "error_message": "Результат оценки отсутствует"
            }
        
        # Если это уже dict в правильном формате
        if isinstance(result, dict):
            required_fields = ["status", "result_data", "agent_name"]
            if all(field in result for field in required_fields):
                return result
            
            # Если это dict но в неправильном формате, пытаемся исправить
            return {
                "status": result.get("status", "unknown"),
                "result_data": result.get("result_data", result),  # Fallback на весь result
                "agent_name": result.get("agent_name", "unknown"),
                "error_message": result.get("error_message")
            }
        
        # Если это объект AgentTaskResult или похожий
        elif hasattr(result, 'status') and hasattr(result, 'result_data'):
            return {
                "status": str(getattr(result, 'status', 'unknown')),
                "result_data": getattr(result, 'result_data', None),
                "agent_name": getattr(result, 'agent_name', 'unknown'),
                "error_message": getattr(result, 'error_message', None)
            }
        
        # Если это Pydantic модель
        elif hasattr(result, 'dict'):
            result_dict = result.dict()
            return {
                "status": result_dict.get("status", "unknown"),
                "result_data": result_dict.get("result_data", result_dict),
                "agent_name": result_dict.get("agent_name", "unknown"),
                "error_message": result_dict.get("error_message")
            }
        
        # Последний fallback - создаем минимальную структуру
        else:
            return {
                "status": "unknown",
                "result_data": str(result) if result else None,
                "agent_name": "unknown",
                "error_message": f"Неподдерживаемый тип результата: {type(result)}"
            }
    def get_successful_evaluations(self) -> Dict[str, Any]:
        """ИСПРАВЛЕННЫЙ метод - правильно обрабатывает enum статусы"""
        
        all_results = self.get_evaluation_results()
        successful_results = {}
        
        for risk_type, result in all_results.items():
            if not result:
                continue
                
            # ИСПРАВЛЕНО: Правильная проверка статуса (enum vs строка)
            status = None
            if isinstance(result, dict):
                status = result.get("status")
            elif hasattr(result, 'status'):
                status = result.status
                
            # Проверяем статус как ENUM или строку
            is_completed = False
            if hasattr(status, 'value'):
                # Это enum - проверяем value
                is_completed = status.value == "completed"
            elif str(status) == "ProcessingStatus.COMPLETED":
                # Это enum в строковом представлении
                is_completed = True  
            elif status == "completed":
                # Это строка
                is_completed = True
            
            if not is_completed:
                continue
            
            # Проверяем result_data
            result_data = None
            if isinstance(result, dict):
                result_data = result.get("result_data")
            elif hasattr(result, 'result_data'):
                result_data = result.result_data
                
            if result_data is None:
                continue
            
            successful_results[risk_type] = result
    
        return successful_results

    def get_failed_evaluations(self) -> Dict[str, Any]:
        """Возвращает только неудачные оценки"""
        
        all_results = self.get_evaluation_results()
        failed_results = {}
        
        for risk_type, result in all_results.items():
            if (not result or 
                result.get("status") in ["failed", "error"] or 
                result.get("result_data") is None):
                failed_results[risk_type] = result
        
        return failed_results

    def has_evaluation_results(self) -> bool:
        """Проверяет, есть ли хотя бы один результат оценки"""
        
        successful = self.get_successful_evaluations()
        return len(successful) > 0

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по результатам оценки"""
        
        all_results = self.get_evaluation_results()
        successful = self.get_successful_evaluations()
        failed = self.get_failed_evaluations()
        
        return {
            "total_evaluations": len(all_results),
            "successful_evaluations": len(successful),
            "failed_evaluations": len(failed),
            "success_rate": len(successful) / len(all_results) if all_results else 0,
            "successful_types": list(successful.keys()),
            "failed_types": list(failed.keys())
        }
# ===============================
# Вспомогательные функции
# ===============================

def create_risk_evaluation(
    risk_type: RiskType,
    probability: int,
    impact: int,
    probability_reasoning: str,
    impact_reasoning: str,
    evaluator_agent: str,
    recommendations: List[str] = None,
    confidence: float = 0.8
) -> RiskEvaluation:
    """Создание оценки риска с автоматическим расчетом"""
    
    return RiskEvaluation(
        risk_type=risk_type,
        evaluator_agent=evaluator_agent,
        probability_score=probability,
        impact_score=impact,
        probability_reasoning=probability_reasoning,
        impact_reasoning=impact_reasoning,
        recommendations=recommendations or [],
        confidence_level=confidence
    )


def calculate_overall_risk_score(risk_evaluations: Dict[str, RiskEvaluation]) -> tuple[int, RiskLevel]:
    """Расчет общего балла и уровня риска"""
    
    if not risk_evaluations:
        return 1, RiskLevel.LOW
    
    total_scores = [eval.total_score for eval in risk_evaluations.values()]
    overall_score = sum(total_scores) // len(total_scores)
    
    if overall_score <= 6:
        return overall_score, RiskLevel.LOW
    elif overall_score <= 14:
        return overall_score, RiskLevel.MEDIUM
    else:
        return overall_score, RiskLevel.HIGH


def extract_risk_evaluations_from_results(evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, RiskEvaluation]:
    """Извлечение RiskEvaluation из результатов агентов"""
    
    risk_evaluations = {}
    
    for risk_type, result in evaluation_results.items():
        if isinstance(result, dict) and result.get("result_data"):
            risk_eval_data = result["result_data"].get("risk_evaluation")
            if risk_eval_data:
                try:
                    risk_evaluations[risk_type] = RiskEvaluation(**risk_eval_data)
                except Exception as e:
                    print(f"Ошибка создания RiskEvaluation для {risk_type}: {e}")
    
    return risk_evaluations


def get_highest_risk_areas(risk_evaluations: Dict[str, RiskEvaluation], top_n: int = 3) -> List[str]:
    """Получение областей наивысшего риска"""
    
    if not risk_evaluations:
        return []
    
    # Сортируем по общему баллу
    sorted_risks = sorted(
        risk_evaluations.items(),
        key=lambda x: x[1].total_score,
        reverse=True
    )
    
    return [risk_type for risk_type, _ in sorted_risks[:top_n]]


def validate_workflow_state(state: Dict[str, Any]) -> WorkflowState:
    """Валидация и создание WorkflowState из словаря"""
    
    try:
        return WorkflowState(**state)
    except Exception as e:
        print(f"Ошибка валидации WorkflowState: {e}")
        # Возвращаем минимальное валидное состояние
        return WorkflowState(
            source_files=state.get("source_files", []),
            current_step=state.get("current_step", "error"),
            error_message=f"Ошибка валидации: {str(e)}"
        )


# ===============================
# Экспорт
# ===============================

__all__ = [
    # Перечисления
    "RiskType", "RiskLevel", "AgentType", "AutonomyLevel", 
    "DataSensitivity", "ProcessingStatus",
    
    # Основные модели
    "AgentProfile", "RiskEvaluation", "CriticEvaluation",
    "AgentRiskAssessment", "AgentTaskResult", "WorkflowState",
    
    # Вспомогательные функции
    "create_risk_evaluation", "calculate_overall_risk_score",
    "extract_risk_evaluations_from_results", "get_highest_risk_areas",
    "validate_workflow_state"
]