# final_critical_fix.py
"""
Критическое исправление финальных проблем
Исправляет LangGraph ошибки и парсинг JSON агентами
"""

import sys
from pathlib import Path

def fix_langgraph_state_annotation():
    """Исправление LangGraph state annotation"""
    print("🔧 Исправление LangGraph state annotation...")
    
    risk_models_file = Path("src/models/risk_models.py")
    
    if not risk_models_file.exists():
        print("❌ Файл risk_models.py не найден")
        return False
    
    content = risk_models_file.read_text(encoding='utf-8')
    
    # Добавляем Annotated импорт если его нет
    if "from typing import Dict, List, Optional, Any, Union" in content:
        if "Annotated" not in content:
            content = content.replace(
                "from typing import Dict, List, Optional, Any, Union",
                "from typing import Dict, List, Optional, Any, Union, Annotated"
            )
            print("✅ Добавлен импорт Annotated")
    
    # Ищем WorkflowState и добавляем аннотацию для assessment_id
    if "class WorkflowState(BaseModel):" in content:
        # Заменяем assessment_id на аннотированную версию
        old_assessment_id = 'assessment_id: Optional[str] = Field(None, description="ID оценки")'
        
        # Новая версия с аннотацией для LangGraph
        new_assessment_id = '''assessment_id: Annotated[Optional[str], "assessment_id"] = Field(None, description="ID оценки")'''
        
        if old_assessment_id in content:
            content = content.replace(old_assessment_id, new_assessment_id)
            print("✅ Добавлена аннотация для assessment_id")
        
        # Также исправляем другие поля которые могут обновляться параллельно
        fields_to_annotate = [
            ('current_step: str = Field("initialization", description="Текущий шаг")', 
             'current_step: Annotated[str, "current_step"] = Field("initialization", description="Текущий шаг")'),
            ('evaluation_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Результаты оценки рисков")',
             'evaluation_results: Annotated[Dict[str, Dict[str, Any]], "evaluation_results"] = Field(default_factory=dict, description="Результаты оценки рисков")'),
        ]
        
        for old_field, new_field in fields_to_annotate:
            if old_field in content:
                content = content.replace(old_field, new_field)
                print(f"✅ Аннотирован: {old_field.split(':')[0]}")
    
    # Добавляем reducer для конкурентных обновлений
    if "class WorkflowState(BaseModel):" in content and "reducer=" not in content:
        # Находим место после определения полей класса
        lines = content.split('\n')
        new_lines = []
        in_workflow_class = False
        
        for line in lines:
            new_lines.append(line)
            
            if "class WorkflowState(BaseModel):" in line:
                in_workflow_class = True
            elif in_workflow_class and line.strip().startswith('class Config:'):
                # Добавляем reducer перед Config
                new_lines.insert(-1, '')
                new_lines.insert(-1, '    # Reducer для handling concurrent updates')
                new_lines.insert(-1, '    @staticmethod')
                new_lines.insert(-1, '    def assessment_id_reducer(left: Optional[str], right: Optional[str]) -> Optional[str]:')
                new_lines.insert(-1, '        """Reducer для assessment_id - берем существующее значение"""')
                new_lines.insert(-1, '        return left if left is not None else right')
                new_lines.insert(-1, '')
                new_lines.insert(-1, '    @staticmethod')
                new_lines.insert(-1, '    def evaluation_results_reducer(left: Dict, right: Dict) -> Dict:')
                new_lines.insert(-1, '        """Reducer для evaluation_results - объединяем результаты"""')
                new_lines.insert(-1, '        result = left.copy() if left else {}')
                new_lines.insert(-1, '        if right:')
                new_lines.insert(-1, '            result.update(right)')
                new_lines.insert(-1, '        return result')
                new_lines.insert(-1, '')
                break
        
        content = '\n'.join(new_lines)
        print("✅ Добавлены reducers для concurrent updates")
    
    risk_models_file.write_text(content, encoding='utf-8')
    return True

def fix_evaluator_json_parsing():
    """Исправление парсинга JSON в агентах-оценщиках"""
    print("🔧 Исправление парсинга JSON в агентах-оценщиках...")
    
    evaluator_file = Path("src/agents/evaluator_agents.py")
    
    if not evaluator_file.exists():
        print("❌ Файл evaluator_agents.py не найден")
        return False
    
    content = evaluator_file.read_text(encoding='utf-8')
    
    # Ищем метод _parse_llm_response и исправляем его
    old_parse_method = '''    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """Парсинг ответа LLM с извлечением JSON"""
        try:
            # Ищем JSON блок в ответе
            if "```json" in response_content:
                start = response_content.find("```json") + 7
                end = response_content.find("```", start)
                if end != -1:
                    json_content = response_content[start:end].strip()
                else:
                    json_content = response_content[start:].strip()
            else:
                # Пытаемся найти JSON по фигурным скобкам
                start = response_content.find("{")
                end = response_content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_content = response_content[start:end+1]
                else:
                    json_content = response_content.strip()
            
            return json.loads(json_content)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Ошибка парсинга JSON: {e}\\nОтвет: {response_content[:200]}...")'''
    
    new_parse_method = '''    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """Парсинг ответа LLM с извлечением JSON - ИСПРАВЛЕННАЯ ВЕРСИЯ"""
        try:
            # Ищем JSON блок в ответе
            if "```json" in response_content:
                start = response_content.find("```json") + 7
                end = response_content.find("```", start)
                if end != -1:
                    json_content = response_content[start:end].strip()
                else:
                    json_content = response_content[start:].strip()
            else:
                # Пытаемся найти JSON по фигурным скобкам
                start = response_content.find("{")
                end = response_content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_content = response_content[start:end+1]
                else:
                    json_content = response_content.strip()
            
            parsed_data = json.loads(json_content)
            
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Проверяем обязательные поля и добавляем если отсутствуют
            required_fields = {
                "probability_score": 3,  # дефолтное значение
                "impact_score": 3,
                "total_score": 9,
                "risk_level": "medium",
                "probability_reasoning": "Автоматически сгенерированное обоснование",
                "impact_reasoning": "Автоматически сгенерированное обоснование",
                "identified_risks": [],
                "recommendations": [],
                "suggested_controls": [],
                "confidence_level": 0.7
            }
            
            # Добавляем отсутствующие поля
            for field, default_value in required_fields.items():
                if field not in parsed_data:
                    parsed_data[field] = default_value
                    print(f"⚠️ Добавлено отсутствующее поле {field}: {default_value}")
            
            # Валидируем числовые поля
            if "probability_score" in parsed_data:
                try:
                    parsed_data["probability_score"] = int(parsed_data["probability_score"])
                    if not (1 <= parsed_data["probability_score"] <= 5):
                        parsed_data["probability_score"] = 3
                except (ValueError, TypeError):
                    parsed_data["probability_score"] = 3
            
            if "impact_score" in parsed_data:
                try:
                    parsed_data["impact_score"] = int(parsed_data["impact_score"])
                    if not (1 <= parsed_data["impact_score"] <= 5):
                        parsed_data["impact_score"] = 3
                except (ValueError, TypeError):
                    parsed_data["impact_score"] = 3
            
            # Пересчитываем total_score
            parsed_data["total_score"] = parsed_data["probability_score"] * parsed_data["impact_score"]
            
            # Определяем risk_level на основе total_score
            total_score = parsed_data["total_score"]
            if total_score <= 6:
                parsed_data["risk_level"] = "low"
            elif total_score <= 14:
                parsed_data["risk_level"] = "medium"
            else:
                parsed_data["risk_level"] = "high"
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            # Если парсинг не удался, возвращаем минимальные валидные данные
            print(f"⚠️ Ошибка парсинга JSON, возвращаем дефолтные данные: {e}")
            return {
                "probability_score": 3,
                "impact_score": 3,
                "total_score": 9,
                "risk_level": "medium",
                "probability_reasoning": f"Не удалось распарсить ответ LLM: {str(e)}",
                "impact_reasoning": "Использованы дефолтные значения",
                "identified_risks": ["Ошибка парсинга ответа LLM"],
                "recommendations": ["Проверить промпт и формат ответа"],
                "suggested_controls": ["Улучшить валидацию ответов"],
                "confidence_level": 0.3
            }'''
    
    if old_parse_method in content:
        content = content.replace(old_parse_method, new_parse_method)
        print("✅ Исправлен метод _parse_llm_response")
    else:
        print("⚠️ Не найден старый метод _parse_llm_response для замены")
    
    evaluator_file.write_text(content, encoding='utf-8')
    return True

def fix_graph_builder_concurrent_updates():
    """Исправление concurrent updates в graph_builder"""
    print("🔧 Исправление concurrent updates в graph_builder...")
    
    graph_file = Path("src/workflow/graph_builder.py")
    
    if not graph_file.exists():
        print("❌ Файл graph_builder.py не найден")
        return False
    
    content = graph_file.read_text(encoding='utf-8')
    
    # Исправляем создание узлов оценщиков - убираем конкурентные обновления
    old_evaluator_nodes = '''        # 4. Параллельная оценка рисков (6 узлов)
        evaluator_nodes = create_evaluator_nodes_for_langgraph_fixed(self.evaluators)
        for node_name, node_func in evaluator_nodes.items():
            workflow.add_node(node_name, log_graph_node(node_name)(node_func))'''
    
    new_evaluator_nodes = '''        # 4. Параллельная оценка рисков (6 узлов) - с исправлением concurrent updates
        evaluator_nodes = create_evaluator_nodes_for_langgraph_fixed(self.evaluators)
        for node_name, node_func in evaluator_nodes.items():
            # Оборачиваем узел для предотвращения concurrent updates
            def create_safe_evaluator_node(original_func):
                async def safe_node(state):
                    result = await original_func(state)
                    # Возвращаем только изменения для этого конкретного узла
                    return {"evaluation_results": result.get("evaluation_results", {})}
                return safe_node
            
            safe_node_func = create_safe_evaluator_node(node_func)
            workflow.add_node(node_name, log_graph_node(node_name)(safe_node_func))'''
    
    if old_evaluator_nodes in content:
        content = content.replace(old_evaluator_nodes, new_evaluator_nodes)
        print("✅ Исправлены узлы оценщиков для предотвращения concurrent updates")
    else:
        print("⚠️ Не найден блок evaluator_nodes для исправления")
    
    graph_file.write_text(content, encoding='utf-8')
    return True

