# src/models/database.py
"""
SQLite база данных для системы оценки рисков ИИ-агентов
Асинхронная работа с базой данных
"""

import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update, delete

from .risk_models import (
    AgentProfile, AgentRiskAssessment, RiskEvaluation, 
    CriticEvaluation, RiskType, RiskLevel, ProcessingStatus
)

Base = declarative_base()


# ===============================
# SQLAlchemy модели таблиц
# ===============================

class AgentProfileDB(Base):
    """Таблица профилей ИИ-агентов"""
    __tablename__ = "agent_profiles"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, index=True)
    version = Column(String, nullable=False, default="1.0")
    description = Column(Text, nullable=False)
    agent_type = Column(String, nullable=False)
    
    # Техническая спецификация (JSON)
    llm_model = Column(String, nullable=False)
    autonomy_level = Column(String, nullable=False)
    data_access = Column(JSON, nullable=True)  # List[DataSensitivity]
    external_apis = Column(JSON, nullable=True)  # List[str]
    
    # Бизнес-контекст
    target_audience = Column(String, nullable=False)
    operations_per_hour = Column(Integer, nullable=True)
    revenue_per_operation = Column(Float, nullable=True)
    
    # Промпты и ограничения (JSON)
    system_prompts = Column(JSON, nullable=True)  # List[str]
    guardrails = Column(JSON, nullable=True)  # List[str]
    source_files = Column(JSON, nullable=True)  # List[str]
    
    # Метаданные
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=False)


class RiskAssessmentDB(Base):
    """Таблица итоговых оценок рисков"""
    __tablename__ = "risk_assessments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_profile_id = Column(String, nullable=False, index=True)
    assessment_id = Column(String, nullable=False, index=True)
    # Итоговые результаты
    overall_risk_score = Column(Integer, nullable=False)
    overall_risk_level = Column(String, nullable=False)
    highest_risk_areas = Column(JSON, nullable=True)  # List[RiskType]
    
    # Рекомендации (JSON)
    priority_recommendations = Column(JSON, nullable=True)  # List[str]
    suggested_guardrails = Column(JSON, nullable=True)  # List[str]
    
    # Финансовые оценки
    total_expected_loss = Column(Float, nullable=True)
    
    # Метаданные процесса
    assessment_timestamp = Column(DateTime, default=datetime.now, nullable=False)
    processing_time_seconds = Column(Float, nullable=True)
    quality_checks_passed = Column(Boolean, default=True, nullable=False)


class RiskEvaluationDB(Base):
    """Таблица оценок отдельных типов рисков"""
    __tablename__ = "risk_evaluations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    assessment_id = Column(String, nullable=False, index=True)
    risk_type = Column(String, nullable=False, index=True)
    
    # Оценки
    probability_score = Column(Integer, nullable=False)
    impact_score = Column(Integer, nullable=False)
    total_score = Column(Integer, nullable=False)
    risk_level = Column(String, nullable=False)
    
    # Обоснование
    probability_reasoning = Column(Text, nullable=False)
    impact_reasoning = Column(Text, nullable=False)
    key_factors = Column(JSON, nullable=True)  # List[str]
    recommendations = Column(JSON, nullable=True)  # List[str]
    
    # Метаданные
    evaluator_agent = Column(String, nullable=False)
    evaluation_time = Column(DateTime, default=datetime.now, nullable=False)
    confidence_level = Column(Float, nullable=False)


class CriticEvaluationDB(Base):
    """Таблица оценок критика"""
    __tablename__ = "critic_evaluations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    assessment_id = Column(String, nullable=False, index=True)
    risk_type = Column(String, nullable=False, index=True)
    
    # Оценка качества
    quality_score = Column(Float, nullable=False)
    is_acceptable = Column(Boolean, nullable=False)
    
    # Замечания (JSON)
    issues_found = Column(JSON, nullable=True)  # List[str]
    improvement_suggestions = Column(JSON, nullable=True)  # List[str]
    
    # Метаданные
    critic_reasoning = Column(Text, nullable=False)
    review_time = Column(DateTime, default=datetime.now, nullable=False)


class ProcessingLogDB(Base):
    """Таблица логов обработки"""
    __tablename__ = "processing_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    assessment_id = Column(String, nullable=False, index=True)
    
    # Информация о задаче
    agent_name = Column(String, nullable=False)
    task_type = Column(String, nullable=False)
    status = Column(String, nullable=False)
    
    # Результаты
    result_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Метрики времени
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    execution_time_seconds = Column(Float, nullable=True)


# ===============================
# Менеджер базы данных
# ===============================

class DatabaseManager:
    """Асинхронный менеджер базы данных"""
    
    def __init__(self, database_url: str = "sqlite+aiosqlite:///./ai_risk_assessment.db"):
        self.database_url = database_url
        self.engine = None
        self.async_session = None
        
    async def initialize(self):
        """Инициализация базы данных"""
        # Создаем директорию для БД если нужно
        if "sqlite" in self.database_url:
            db_path = Path(self.database_url.split("///")[-1])
            db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Создаем движок и сессию
        self.engine = create_async_engine(
            self.database_url,
            echo=False,  # Установить True для отладки SQL
            future=True
        )
        
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        # Создаем таблицы
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def close(self):
        """Закрытие соединения"""
        if self.engine:
            await self.engine.dispose()
    
    # ===============================
    # Методы для работы с профилями
    # ===============================
    
    async def save_agent_profile(self, profile: AgentProfile) -> str:
        """Сохранение профиля агента"""
        async with self.async_session() as session:
            # Проверяем, есть ли уже такой профиль
            stmt = select(AgentProfileDB).where(
                AgentProfileDB.name == profile.name,
                AgentProfileDB.version == profile.version
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                # Обновляем существующий
                profile_id = existing.id
                await session.execute(
                    update(AgentProfileDB)
                    .where(AgentProfileDB.id == profile_id)
                    .values(
                        description=profile.description,
                        agent_type=profile.agent_type.value,
                        llm_model=profile.llm_model,
                        autonomy_level=profile.autonomy_level.value,
                        data_access=[ds.value for ds in profile.data_access],
                        external_apis=profile.external_apis,
                        target_audience=profile.target_audience,
                        operations_per_hour=profile.operations_per_hour,
                        revenue_per_operation=profile.revenue_per_operation,
                        system_prompts=profile.system_prompts,
                        guardrails=profile.guardrails,
                        source_files=profile.source_files,
                        updated_at=datetime.now()
                    )
                )
            else:
                # Создаем новый
                profile_id = str(uuid.uuid4())
                db_profile = AgentProfileDB(
                    id=profile_id,
                    name=profile.name,
                    version=profile.version,
                    description=profile.description,
                    agent_type=profile.agent_type.value,
                    llm_model=profile.llm_model,
                    autonomy_level=profile.autonomy_level.value,
                    data_access=[ds.value for ds in profile.data_access],
                    external_apis=profile.external_apis,
                    target_audience=profile.target_audience,
                    operations_per_hour=profile.operations_per_hour,
                    revenue_per_operation=profile.revenue_per_operation,
                    system_prompts=profile.system_prompts,
                    guardrails=profile.guardrails,
                    source_files=profile.source_files
                )
                session.add(db_profile)
            
            await session.commit()
            return profile_id
    
    async def get_agent_profile(self, profile_id: str) -> Optional[AgentProfile]:
        """Получение профиля агента по ID"""
        async with self.async_session() as session:
            stmt = select(AgentProfileDB).where(AgentProfileDB.id == profile_id)
            result = await session.execute(stmt)
            db_profile = result.scalar_one_or_none()
            
            if not db_profile:
                return None
            
            # Конвертируем в Pydantic модель
            return AgentProfile(
                name=db_profile.name,
                version=db_profile.version,
                description=db_profile.description,
                agent_type=db_profile.agent_type,
                llm_model=db_profile.llm_model,
                autonomy_level=db_profile.autonomy_level,
                data_access=db_profile.data_access or [],
                external_apis=db_profile.external_apis or [],
                target_audience=db_profile.target_audience,
                operations_per_hour=db_profile.operations_per_hour,
                revenue_per_operation=db_profile.revenue_per_operation,
                system_prompts=db_profile.system_prompts or [],
                guardrails=db_profile.guardrails or [],
                source_files=db_profile.source_files or [],
                created_at=db_profile.created_at,
                updated_at=db_profile.updated_at
            )
    
    async def list_agent_profiles(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Список профилей агентов"""
        async with self.async_session() as session:
            stmt = select(AgentProfileDB).order_by(AgentProfileDB.updated_at.desc()).limit(limit)
            result = await session.execute(stmt)
            profiles = result.scalars().all()
            
            return [
                {
                    "id": p.id,
                    "name": p.name,
                    "version": p.version,
                    "agent_type": p.agent_type,
                    "updated_at": p.updated_at
                }
                for p in profiles
            ]
    
    # ===============================
    # Методы для работы с оценками
    # ===============================
    
    # src/models/database.py - ИСПРАВЛЕНИЕ метода save_risk_assessment

    async def save_risk_assessment(self, assessment: AgentRiskAssessment, profile_id: str) -> str:
        """Сохранение итоговой оценки рисков - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
        assessment_id = str(uuid.uuid4())
        
        async with self.async_session() as session:
            # Сохраняем основную оценку
            db_assessment = RiskAssessmentDB(
                id=assessment_id,
                agent_profile_id=profile_id,
                assessment_id=assessment.assessment_id,
                overall_risk_score=assessment.overall_risk_score,
                overall_risk_level=assessment.overall_risk_level.value if hasattr(assessment.overall_risk_level, 'value') else str(assessment.overall_risk_level),
                # ИСПРАВЛЕНО: highest_risk_areas уже список строк, не нужно .value
                highest_risk_areas=assessment.highest_risk_areas,
                priority_recommendations=assessment.priority_recommendations,
                suggested_guardrails=assessment.suggested_guardrails,
                total_expected_loss=assessment.total_expected_loss,
                processing_time_seconds=assessment.processing_time_seconds,
                quality_checks_passed=assessment.quality_checks_passed
            )
            session.add(db_assessment)
            
            # Сохраняем оценки по типам рисков
            for risk_type_key, evaluation in assessment.risk_evaluations.items():
                # ИСПРАВЛЕНО: risk_type_key уже строка, не нужно .value
                risk_type_str = risk_type_key if isinstance(risk_type_key, str) else str(risk_type_key)
                
                db_evaluation = RiskEvaluationDB(
                    assessment_id=assessment_id,
                    risk_type=risk_type_str,
                    probability_score=evaluation.probability_score,
                    impact_score=evaluation.impact_score,
                    total_score=evaluation.total_score,
                    risk_level=evaluation.risk_level.value if hasattr(evaluation.risk_level, 'value') else str(evaluation.risk_level),
                    probability_reasoning=evaluation.probability_reasoning,
                    impact_reasoning=evaluation.impact_reasoning,
                    key_factors=evaluation.key_factors or [],
                    recommendations=evaluation.recommendations or [],
                    evaluator_agent=evaluation.evaluator_agent,
                    confidence_level=evaluation.confidence_level
                )
                session.add(db_evaluation)
            
            # Сохраняем оценки критика
            for risk_type_key, critic_eval in assessment.critic_evaluations.items():
                # ИСПРАВЛЕНО: risk_type_key уже строка, не нужно .value
                risk_type_str = risk_type_key if isinstance(risk_type_key, str) else str(risk_type_key)
                
                db_critic = CriticEvaluationDB(
                    assessment_id=assessment_id,
                    risk_type=risk_type_str,
                    quality_score=critic_eval.quality_score,
                    is_acceptable=critic_eval.is_acceptable,
                    issues_found=critic_eval.issues_found or [],
                    improvement_suggestions=critic_eval.improvement_suggestions or [],
                    critic_reasoning=critic_eval.critic_reasoning
                )
                session.add(db_critic)
            
            await session.commit()
            return assessment_id
    
    async def get_risk_assessment(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        """Получение оценки рисков по ID"""
        async with self.async_session() as session:
            # Основная оценка
            stmt = select(RiskAssessmentDB).where(RiskAssessmentDB.id == assessment_id)
            result = await session.execute(stmt)
            assessment = result.scalar_one_or_none()
            
            if not assessment:
                return None
            
            # Оценки по типам рисков
            stmt = select(RiskEvaluationDB).where(RiskEvaluationDB.assessment_id == assessment_id)
            result = await session.execute(stmt)
            evaluations = result.scalars().all()
            
            # Оценки критика
            stmt = select(CriticEvaluationDB).where(CriticEvaluationDB.assessment_id == assessment_id)
            result = await session.execute(stmt)
            critic_evals = result.scalars().all()
            
            return {
                "assessment": assessment,
                "evaluations": evaluations,
                "critic_evaluations": critic_evals
            }
    
    async def get_assessments_for_agent(self, profile_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Получение всех оценок для агента"""
        async with self.async_session() as session:
            stmt = (
                select(RiskAssessmentDB)
                .where(RiskAssessmentDB.agent_profile_id == profile_id)
                .order_by(RiskAssessmentDB.assessment_timestamp.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            assessments = result.scalars().all()
            
            return [
                {
                    "id": a.id,
                    "overall_risk_level": a.overall_risk_level,
                    "overall_risk_score": a.overall_risk_score,
                    "assessment_timestamp": a.assessment_timestamp,
                    "quality_checks_passed": a.quality_checks_passed
                }
                for a in assessments
            ]
    
    # ===============================
    # Методы для логирования
    # ===============================
    
    async def log_processing_step(
        self, 
        assessment_id: str,
        agent_name: str,
        task_type: str,
        status: ProcessingStatus,
        result_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        execution_time: Optional[float] = None
    ) -> str:
        """Логирование шага обработки"""
        log_id = str(uuid.uuid4())
        
        async with self.async_session() as session:
            log_entry = ProcessingLogDB(
                id=log_id,
                assessment_id=assessment_id,
                agent_name=agent_name,
                task_type=task_type,
                status=status.value,
                result_data=result_data,
                error_message=error_message,
                start_time=datetime.now(),
                execution_time_seconds=execution_time
            )
            session.add(log_entry)
            await session.commit()
            
        return log_id
    
    async def get_processing_logs(self, assessment_id: str) -> List[Dict[str, Any]]:
        """Получение логов обработки"""
        async with self.async_session() as session:
            stmt = (
                select(ProcessingLogDB)
                .where(ProcessingLogDB.assessment_id == assessment_id)
                .order_by(ProcessingLogDB.start_time)
            )
            result = await session.execute(stmt)
            logs = result.scalars().all()
            
            return [
                {
                    "agent_name": log.agent_name,
                    "task_type": log.task_type,
                    "status": log.status,
                    "start_time": log.start_time,
                    "execution_time": log.execution_time_seconds,
                    "error_message": log.error_message
                }
                for log in logs
            ]


# ===============================
# Глобальный экземпляр
# ===============================

# Создаем глобальный экземпляр менеджера БД
db_manager = DatabaseManager()

async def get_db_manager() -> DatabaseManager:
    """Получение менеджера базы данных"""
    if not db_manager.engine:
        await db_manager.initialize()
    return db_manager