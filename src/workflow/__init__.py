# src/workflow/__init__.py
"""
Workflow модуль для системы оценки рисков ИИ-агентов
Содержит LangGraph workflow и оркестрацию мультиагентной системы
"""

from .graph_builder import (
    RiskAssessmentWorkflow,
    create_risk_assessment_workflow,
    create_workflow_from_env
)

__all__ = [
    "RiskAssessmentWorkflow",
    "create_risk_assessment_workflow", 
    "create_workflow_from_env"
]