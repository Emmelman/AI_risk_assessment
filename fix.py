# final_workflow_fix.py
"""
Финальное исправление ошибки в workflow
Исправляет проблему с типами данных в WorkflowState
"""

import sys
from pathlib import Path

def apply_critical_fixes():
    """Применение критических исправлений"""
    
    print("🔧 Применение критических исправлений workflow...")
    
    # Исправление 1: Обновляем WorkflowState в risk_models.py
    risk_models_file = Path("src/models/risk_models.py")
    
    if risk_models_file.exists():
        content = risk_models_file.read_text(encoding='utf-8')
        
        # Находим и заменяем WorkflowState
        old_workflow_state = """class WorkflowState(BaseModel):
    \"\"\"Состояние workflow для LangGraph\"\"\"
    
    # Входные данные
    agent_profile: Optional[AgentProfile] = None
    source_files: List[str] = Field(default_factory=list)
    
    # Промежуточные результаты
    profiling_result: Optional[AgentTaskResult] = None
    evaluation_results: Dict[RiskType, AgentTaskResult] = Field(default_factory=dict)
    critic_results: Dict[RiskType, AgentTaskResult] = Field(default_factory=dict)"""
        
        new_workflow_state = """class WorkflowState(BaseModel):
    \"\"\"Состояние workflow для LangGraph - ИСПРАВЛЕНО для совместимости\"\"\"
    
    # Идентификаторы
    assessment_id: Optional[str] = Field(None, description="ID оценки")
    preliminary_agent_name: Optional[str] = Field(None, description="Предварительное имя агента")
    
    # Входные данные
    source_files: List[str] = Field(default_factory=list, description="Файлы для анализа")
    agent_profile: Optional[Dict[str, Any]] = Field(None, description="Профиль агента")
    
    # Промежуточные результаты (как словари для совместимости с LangGraph)
    profiling_result: Optional[Dict[str, Any]] = Field(None, description="Результат профилирования")
    evaluation_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Результаты оценки рисков")
    critic_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Результаты критического анализа")"""
        
        if old_workflow_state in content:
            content = content.replace(old_workflow_state, new_workflow_state)
            
            # Добавляем методы для словарной совместимости
            workflow_state_methods = """
    
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
    
    # Итоговый результат
    final_assessment: Optional[Dict[str, Any]] = Field(None, description="Итоговая оценка")
    
    class Config:
        \"\"\"Конфигурация модели\"\"\"
        extra = "allow"  # Разрешаем дополнительные поля
        use_enum_values = True
        arbitrary_types_allowed = True
        
    def __getitem__(self, key: str):
        \"\"\"Поддержка доступа как к словарю\"\"\"
        return getattr(self, key, None)
    
    def __setitem__(self, key: str, value):
        \"\"\"Поддержка записи как в словарь\"\"\"
        setattr(self, key, value)
    
    def get(self, key: str, default=None):
        \"\"\"Метод get() как у словаря\"\"\"
        return getattr(self, key, default)
    
    def update(self, updates: Dict[str, Any]):
        \"\"\"Обновление значений как у словаря\"\"\"
        for key, value in updates.items():
            setattr(self, key, value)
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        \"\"\"Преобразование в словарь с поддержкой всех полей\"\"\"
        result = super().dict(**kwargs)
        
        # Добавляем все дополнительные атрибуты
        for key, value in self.__dict__.items():
            if key not in result and not key.startswith('_'):
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                else:
                    result[key] = value
        
        return result"""
            
            # Находим конец класса WorkflowState и заменяем
            import re
            pattern = r"(class WorkflowState\(BaseModel\):.*?)(\n\nclass|\n\n#|\n\ndef|\Z)"
            
            def replace_workflow_state(match):
                return new_workflow_state + workflow_state_methods + match.group(2)
            
            content = re.sub(pattern, replace_workflow_state, content, flags=re.DOTALL)
            
            risk_models_file.write_text(content, encoding='utf-8')
            print("✅ WorkflowState исправлен в risk_models.py")
        else:
            print("⚠️ Не найден старый WorkflowState - возможно уже исправлен")
    
    # Исправление 2: Обновляем profiler_node_function в profiler_agent.py
    profiler_file = Path("src/agents/profiler_agent.py")
    
    if profiler_file.exists():
        content = profiler_file.read_text(encoding='utf-8')
        
        # Находим и заменяем profiler_node_function
        old_profiler_node = """        # Обновляем состояние
        updated_state = state.copy()
        updated_state["profiling_result"] = result
        
        if result.status == ProcessingStatus.COMPLETED:
            # Добавляем профиль агента в состояние для дальнейшего использования
            agent_profile_data = result.result_data["agent_profile"]
            updated_state["agent_profile"] = agent_profile_data"""
        
        new_profiler_node = """        # Обновляем состояние - преобразуем AgentTaskResult в словарь
        updated_state = state.copy()
        updated_state["profiling_result"] = result.dict()  # Преобразуем в словарь
        
        if result.status == ProcessingStatus.COMPLETED:
            # Добавляем профиль агента в состояние для дальнейшего использования
            agent_profile_data = result.result_data["agent_profile"]
            updated_state["agent_profile"] = agent_profile_data"""
        
        if old_profiler_node in content:
            content = content.replace(old_profiler_node, new_profiler_node)
            profiler_file.write_text(content, encoding='utf-8')
            print("✅ profiler_node_function исправлен")
        else:
            print("⚠️ Не найден старый profiler_node - ручное исправление")
    
    # Исправление 3: Создание быстрого патча для evaluator_agents.py
    evaluator_file = Path("src/agents/evaluator_agents.py")
    
    if evaluator_file.exists():
        content = evaluator_file.read_text(encoding='utf-8')
        
        # Добавляем в конец файла исправленные функции для LangGraph
        patch_content = '''

# ===============================
# ИСПРАВЛЕННЫЕ ФУНКЦИИ ДЛЯ LANGGRAPH
# ===============================

def create_evaluator_nodes_for_langgraph_fixed(evaluators: Dict[RiskType, Any]) -> Dict[str, callable]:
    """Создание узлов для LangGraph с исправлением типов данных"""
    
    def create_evaluator_node(risk_type: RiskType, evaluator):
        async def evaluator_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Узел оценщика в LangGraph workflow"""
            
            # Извлекаем данные из состояния
            assessment_id = state.get("assessment_id", "unknown")
            agent_profile = state.get("agent_profile", {})
            
            # Подготавливаем входные данные
            input_data = {"agent_profile": agent_profile}
            
            # Запускаем оценщика
            result = await evaluator.run(input_data, assessment_id)
            
            # Обновляем состояние - преобразуем AgentTaskResult в словарь
            updated_state = state.copy()
            
            # Сохраняем результат как словарь
            if "evaluation_results" not in updated_state:
                updated_state["evaluation_results"] = {}
            
            updated_state["evaluation_results"][risk_type.value] = result.dict()
            
            return updated_state
        
        return evaluator_node
    
    # Создаем узлы для всех оценщиков
    nodes = {}
    for risk_type, evaluator in evaluators.items():
        node_name = f"{risk_type.value}_evaluator_node"
        nodes[node_name] = create_evaluator_node(risk_type, evaluator)
    
    return nodes


def create_critic_node_function_fixed(critic_agent):
    """Создает исправленную функцию узла критика для LangGraph"""
    
    async def critic_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Узел критика в LangGraph workflow"""
        
        assessment_id = state.get("assessment_id", "unknown")
        evaluation_results = state.get("evaluation_results", {})
        agent_profile = state.get("agent_profile", {})
        
        # Обновляем состояние
        updated_state = state.copy()
        
        if "critic_results" not in updated_state:
            updated_state["critic_results"] = {}
        
        # Анализируем каждую оценку риска
        for risk_type, eval_result in evaluation_results.items():
            if isinstance(eval_result, dict) and eval_result.get("result_data"):
                risk_evaluation = eval_result["result_data"].get("risk_evaluation")
                
                if risk_evaluation:
                    # Подготавливаем данные для критика
                    critic_input = {
                        "risk_type": risk_type,
                        "risk_evaluation": risk_evaluation,
                        "agent_profile": agent_profile,
                        "evaluator_name": eval_result.get("agent_name", "Unknown")
                    }
                    
                    # Запускаем критика
                    critic_result = await critic_agent.run(critic_input, assessment_id)
                    
                    # Сохраняем результат как словарь
                    updated_state["critic_results"][risk_type] = critic_result.dict()
        
        return updated_state
    
    return critic_node
'''
        
        # Добавляем патч в конец файла если его еще нет
        if "create_evaluator_nodes_for_langgraph_fixed" not in content:
            content += patch_content
            evaluator_file.write_text(content, encoding='utf-8')
            print("✅ Исправленные функции добавлены в evaluator_agents.py")
        else:
            print("✅ Исправленные функции уже есть в evaluator_agents.py")
    
    # Исправление 4: Патч для graph_builder.py
    graph_builder_file = Path("src/workflow/graph_builder.py")
    
    if graph_builder_file.exists():
        content = graph_builder_file.read_text(encoding='utf-8')
        
        # Заменяем импорты
        old_import = """from ..agents.evaluator_agents import (
    create_all_evaluator_agents, create_evaluator_nodes_for_langgraph,
    extract_risk_evaluations_from_results, calculate_overall_risk_score,
    get_highest_risk_areas
)"""
        
        new_import = """from ..agents.evaluator_agents import (
    create_all_evaluator_agents, create_evaluator_nodes_for_langgraph_fixed,
    extract_risk_evaluations_from_results, calculate_overall_risk_score,
    get_highest_risk_areas
)"""
        
        # Заменяем использование функции
        old_usage = "evaluator_nodes = create_evaluator_nodes_for_langgraph(self.evaluators)"
        new_usage = "evaluator_nodes = create_evaluator_nodes_for_langgraph_fixed(self.evaluators)"
        
        content = content.replace(old_import, new_import)
        content = content.replace(old_usage, new_usage)
        
        # Исправляем critic_agent импорт
        old_critic_import = """from ..agents.critic_agent import (
    create_critic_agent, create_critic_node_function, create_quality_check_router
)"""
        
        new_critic_import = """from ..agents.critic_agent import (
    create_critic_agent, create_quality_check_router
)
from ..agents.evaluator_agents import create_critic_node_function_fixed"""
        
        content = content.replace(old_critic_import, new_critic_import)
        
        # Заменяем использование критика
        old_critic_usage = "critic_node = create_critic_node_function(self.critic)"
        new_critic_usage = "critic_node = create_critic_node_function_fixed(self.critic)"
        
        content = content.replace(old_critic_usage, new_critic_usage)
        
        graph_builder_file.write_text(content, encoding='utf-8')
        print("✅ graph_builder.py исправлен")
    
    print("\n🎉 Все критические исправления применены!")
    print("\n🚀 Теперь можно запустить тест:")
    print("   python test_complete_workflow.py")

if __name__ == "__main__":
    apply_critical_fixes()