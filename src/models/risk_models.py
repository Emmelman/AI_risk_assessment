# src/models/risk_models.py
"""
Модели данных для системы оценки рисков ИИ-агентов
Основано на методике ПАО Сбербанк
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


class RiskType(str, Enum):
    """Типы рисков согласно методике Сбербанк"""
    ETHICAL = "ethical"           # Этические и дискриминационные
    STABILITY = "stability"       # Ошибки и нестабильность LLM
    SECURITY = "security"         # Безопасность данных и систем
    AUTONOMY = "autonomy"         # Автономность и управление
    REGULATORY = "regulatory"     # Регуляторные и юридические
    SOCIAL = "social"            # Социальные и манипулятивные


class RiskLevel(str, Enum):
    """Уровни риска"""
    LOW = "low"          # 1-6 баллов
    MEDIUM = "medium"    # 7-14 баллов
    HIGH = "high"        # 15+ баллов


class AgentType(str, Enum):
    """Типы ИИ-агентов"""
    CHATBOT = "chatbot"
    ASSISTANT = "assistant"
    TRADER = "trader"
    SCORER = "scorer"
    ANALYZER = "analyzer"
    GENERATOR = "generator"
    OTHER = "other"


class DataSensitivity(str, Enum):
    """Уровни чувствительности данных"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    CRITICAL = "critical"


class AutonomyLevel(str, Enum):
    """Уровни автономности агента"""
    MANUAL = "manual"          # Полный контроль человека
    ASSISTED = "assisted"      # Помощь человеку
    SUPERVISED = "supervised"  # Под надзором человека
    AUTONOMOUS = "autonomous"  # Полная автономия


# ===============================
# Базовые модели для профайлинга
# ===============================

class AgentProfile(BaseModel):
    """Профиль ИИ-агента для оценки рисков"""
    
    # Основная информация
    name: str = Field(..., description="Название агента")
    version: str = Field("1.0", description="Версия агента")
    description: str = Field(..., description="Описание назначения")
    agent_type: AgentType = Field(..., description="Тип агента")
    
    # Техническая спецификация
    llm_model: str = Field(..., description="Используемая LLM модель")
    autonomy_level: AutonomyLevel = Field(..., description="Уровень автономности")
    data_access: List[DataSensitivity] = Field(default_factory=list, description="Доступ к типам данных")
    external_apis: List[str] = Field(default_factory=list, description="Внешние API")
    
    # Бизнес-контекст
    target_audience: str = Field(..., description="Целевая аудитория")
    operations_per_hour: Optional[int] = Field(None, description="Операций в час")
    revenue_per_operation: Optional[float] = Field(None, description="Доход с операции, руб")
    
    # Промпты и ограничения
    system_prompts: List[str] = Field(default_factory=list, description="Системные промпты")
    guardrails: List[str] = Field(default_factory=list, description="Ограничения безопасности")
    
    # Метаданные
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    source_files: List[str] = Field(default_factory=list, description="Исходные файлы")


class RiskEvaluation(BaseModel):
    """Оценка одного типа риска"""
    
    risk_type: RiskType = Field(..., description="Тип риска")
    probability_score: int = Field(..., ge=1, le=5, description="Вероятность (1-5)")
    impact_score: int = Field(..., ge=1, le=5, description="Тяжесть последствий (1-5)")
    total_score: Optional[int] = Field(None, ge=1, le=25, description="Итоговый балл")
    risk_level: Optional[RiskLevel] = Field(None, description="Уровень риска")
    
    # Обоснование оценки
    probability_reasoning: str = Field(..., description="Обоснование вероятности")
    impact_reasoning: str = Field(..., description="Обоснование тяжести")
    key_factors: List[str] = Field(default_factory=list, description="Ключевые факторы риска")
    recommendations: List[str] = Field(default_factory=list, description="Рекомендации")
    
    # Метаданные
    evaluator_agent: str = Field(..., description="Агент-оценщик")
    evaluation_time: datetime = Field(default_factory=datetime.now)
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Уверенность в оценке")
    
    def __init__(self, **data):
        # Вычисляем total_score и risk_level перед валидацией
        if 'total_score' not in data and 'probability_score' in data and 'impact_score' in data:
            data['total_score'] = data['probability_score'] * data['impact_score']
        
        if 'risk_level' not in data and 'total_score' in data:
            score = data['total_score']
            if score <= 6:
                data['risk_level'] = RiskLevel.LOW
            elif score <= 14:
                data['risk_level'] = RiskLevel.MEDIUM
            else:
                data['risk_level'] = RiskLevel.HIGH
        
        super().__init__(**data)
    
    @validator('total_score', always=True)
    def calculate_total_score(cls, v, values):
        """Автоматический расчет итогового балла"""
        if v is None and 'probability_score' in values and 'impact_score' in values:
            return values['probability_score'] * values['impact_score']
        return v
    
    @validator('risk_level', always=True)
    def determine_risk_level(cls, v, values):
        """Автоматическое определение уровня риска"""
        if v is None and 'total_score' in values:
            score = values['total_score']
            if score <= 6:
                return RiskLevel.LOW
            elif score <= 14:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.HIGH
        return v


class CriticEvaluation(BaseModel):
    """Оценка критика работы агентов-оценщиков"""
    
    risk_type: RiskType = Field(..., description="Проверяемый тип риска")
    quality_score: float = Field(..., ge=0.0, le=10.0, description="Качество оценки (0-10)")
    is_acceptable: bool = Field(..., description="Приемлемо ли качество")
    
    # Замечания и рекомендации
    issues_found: List[str] = Field(default_factory=list, description="Найденные проблемы")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Предложения по улучшению")
    
    # Метаданные
    critic_reasoning: str = Field(..., description="Обоснование критика")
    review_time: datetime = Field(default_factory=datetime.now)


class FinancialLossEstimate(BaseModel):
    """Количественная оценка финансовых потерь"""
    
    risk_type: RiskType = Field(..., description="Тип риска")
    expected_loss: float = Field(..., ge=0.0, description="Ожидаемые потери, руб")
    exposure_amount: Optional[float] = Field(None, description="Подверженность риску, руб")
    probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Вероятность (0-1)")
    recovery_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="Степень возмещения")
    
    calculation_method: str = Field(..., description="Метод расчета")
    assumptions: List[str] = Field(default_factory=list, description="Допущения расчета")


# ===============================
# Итоговые модели результата
# ===============================

class AgentRiskAssessment(BaseModel):
    """Итоговая оценка рисков ИИ-агента"""
    
    # Основная информация
    agent_profile: AgentProfile = Field(..., description="Профиль агента")
    assessment_id: str = Field(..., description="ID оценки")
    
    # Оценки по типам рисков
    risk_evaluations: Dict[RiskType, RiskEvaluation] = Field(..., description="Оценки рисков")
    critic_evaluations: Dict[RiskType, CriticEvaluation] = Field(default_factory=dict, description="Оценки критика")
    
    # Итоговые результаты
    overall_risk_score: int = Field(..., description="Общий балл риска")
    overall_risk_level: RiskLevel = Field(..., description="Общий уровень риска")
    highest_risk_areas: List[RiskType] = Field(default_factory=list, description="Области наивысшего риска")
    
    # Финансовые оценки
    financial_estimates: Dict[RiskType, FinancialLossEstimate] = Field(default_factory=dict)
    total_expected_loss: Optional[float] = Field(None, description="Общие ожидаемые потери")
    
    # Рекомендации
    priority_recommendations: List[str] = Field(default_factory=list, description="Приоритетные рекомендации")
    suggested_guardrails: List[str] = Field(default_factory=list, description="Предлагаемые меры защиты")
    
    # Метаданные процесса
    assessment_timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_seconds: Optional[float] = Field(None, description="Время обработки")
    quality_checks_passed: bool = Field(True, description="Прошли ли проверки качества")
    
    @validator('overall_risk_score', always=True)
    def calculate_overall_score(cls, v, values):
        """Расчет общего балла как максимум среди всех типов рисков"""
        if 'risk_evaluations' in values:
            evaluations = values['risk_evaluations']
            if evaluations:
                return max([eval.total_score for eval in evaluations.values()])
        return v
    
    @validator('overall_risk_level', always=True)
    def determine_overall_level(cls, v, values):
        """Определение общего уровня риска"""
        if 'overall_risk_score' in values:
            score = values['overall_risk_score']
            if score <= 6:
                return RiskLevel.LOW
            elif score <= 14:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.HIGH
        return v


# ===============================
# Вспомогательные модели
# ===============================

class ProcessingStatus(str, Enum):
    """Статусы обработки"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY_NEEDED = "retry_needed"


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


class WorkflowState(BaseModel):
    """Состояние workflow для LangGraph"""
    
    # Входные данные
    agent_profile: Optional[AgentProfile] = None
    source_files: List[str] = Field(default_factory=list)
    
    # Промежуточные результаты
    profiling_result: Optional[AgentTaskResult] = None
    evaluation_results: Dict[RiskType, AgentTaskResult] = Field(default_factory=dict)
    critic_results: Dict[RiskType, AgentTaskResult] = Field(default_factory=dict)
    
    # Итоговый результат
    final_assessment: Optional[AgentRiskAssessment] = None
    
    # Управление процессом
    current_step: str = Field("initialization", description="Текущий шаг")
    retry_count: Dict[str, int] = Field(default_factory=dict, description="Счетчики повторов")
    max_retries: int = Field(3, description="Максимум повторов")
    
    # Настройки качества
    quality_threshold: float = Field(7.0, description="Порог качества для критика")
    require_critic_approval: bool = Field(True, description="Требовать одобрение критика")