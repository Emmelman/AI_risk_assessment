# src/tools/code_analyzer.py
"""
Анализатор кодовой базы ИИ-агентов
Извлекает информацию о структуре, функциях, зависимостях и потенциальных рисках
"""

import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass
from datetime import datetime

from ..utils.logger import get_logger, LogContext


@dataclass
class CodeFile:
    """Информация о файле кода"""
    path: str
    language: str
    size_lines: int
    size_bytes: int
    imports: List[str]
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    variables: List[str]
    comments: List[str]
    complexity_score: float
    security_issues: List[str]


@dataclass
class CodeAnalysisResult:
    """Результат анализа кодовой базы"""
    project_path: str
    total_files: int
    total_lines: int
    languages: Dict[str, int]
    files: List[CodeFile]
    dependencies: Dict[str, List[str]]
    entry_points: List[str]
    configuration_files: List[str]
    security_summary: Dict[str, Any]
    complexity_summary: Dict[str, Any]
    analysis_time: float
    success: bool
    error_message: Optional[str] = None


class CodeAnalyzerError(Exception):
    """Исключение при анализе кода"""
    pass


class BaseCodeAnalyzer:
    """Базовый класс для анализаторов кода"""
    
    def __init__(self):
        self.logger = get_logger()
        self.supported_extensions = []
        self.security_patterns = []
        self.complexity_weights = {}
    
    def can_analyze(self, file_path: Union[str, Path]) -> bool:
        """Проверка, может ли анализатор обработать файл"""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions
    
    def analyze_file(self, file_path: Union[str, Path]) -> CodeFile:
        """Анализ отдельного файла (переопределяется в наследниках)"""
        raise NotImplementedError
    
    def _read_file_safely(self, file_path: Union[str, Path]) -> Optional[str]:
        """Безопасное чтение файла с обработкой кодировок"""
        path = Path(file_path)
        encodings = ['utf-8', 'cp1251', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.bind_context("code_analyzer", "base_analyzer").error(
                    f"Ошибка чтения файла {path}: {e}"
                )
                return None
        
        return None
    
    def _calculate_complexity_score(self, content: str, language: str) -> float:
        """Расчет базовой сложности кода"""
        if not content:
            return 0.0
        
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Базовые метрики
        line_count = len(non_empty_lines)
        avg_line_length = sum(len(line) for line in non_empty_lines) / max(line_count, 1)
        
        # Сложные конструкции
        complexity_indicators = [
            r'\bif\b', r'\bwhile\b', r'\bfor\b', r'\btry\b',
            r'\bexcept\b', r'\bwith\b', r'\basync\b', r'\bawait\b'
        ]
        
        complexity_count = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in complexity_indicators
        )
        
        # Нормализованная оценка сложности
        complexity_score = (line_count * 0.1 + avg_line_length * 0.01 + complexity_count * 2) / 10
        return min(complexity_score, 10.0)  # Ограничиваем максимум
    
    def _find_security_issues(self, content: str, language: str) -> List[str]:
        """Поиск потенциальных проблем безопасности"""
        issues = []
        
        # Общие паттерны безопасности
        security_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ],
            'unsafe_eval': [
                r'\beval\s*\(',
                r'\bexec\s*\('
            ],
            'sql_injection': [
                r'execute\s*\(\s*["\'].*%.*["\']',
                r'query\s*\(\s*["\'].*\+.*["\']'
            ],
            'path_traversal': [
                r'open\s*\(\s*.*\+.*\)',
                r'\.\.\/.*\.\.\/'
            ]
        }
        
        for issue_type, patterns in security_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    issues.append(f"{issue_type}: найден подозрительный паттерн")
        
        return issues


class PythonCodeAnalyzer(BaseCodeAnalyzer):
    """Анализатор Python кода"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.py']
    
    def analyze_file(self, file_path: Union[str, Path]) -> CodeFile:
        """Анализ Python файла"""
        path = Path(file_path)
        content = self._read_file_safely(path)
        
        if content is None:
            return CodeFile(
                path=str(path),
                language="python",
                size_lines=0,
                size_bytes=0,
                imports=[],
                functions=[],
                classes=[],
                variables=[],
                comments=[],
                complexity_score=0.0,
                security_issues=["Ошибка чтения файла"]
            )
        
        try:
            # Парсим Python AST
            tree = ast.parse(content)
            
            # Извлекаем информацию
            imports = self._extract_imports(tree)
            functions = self._extract_functions(tree, content)
            classes = self._extract_classes(tree, content)
            variables = self._extract_variables(tree)
            comments = self._extract_comments(content)
            
            # Анализируем сложность и безопасность
            complexity_score = self._calculate_python_complexity(tree, content)
            security_issues = self._find_python_security_issues(content, tree)
            
            return CodeFile(
                path=str(path),
                language="python",
                size_lines=len(content.split('\n')),
                size_bytes=len(content.encode('utf-8')),
                imports=imports,
                functions=functions,
                classes=classes,
                variables=variables,
                comments=comments,
                complexity_score=complexity_score,
                security_issues=security_issues
            )
            
        except SyntaxError as e:
            return CodeFile(
                path=str(path),
                language="python",
                size_lines=len(content.split('\n')),
                size_bytes=len(content.encode('utf-8')),
                imports=[],
                functions=[],
                classes=[],
                variables=[],
                comments=self._extract_comments(content),
                complexity_score=0.0,
                security_issues=[f"Синтаксическая ошибка: {e}"]
            )
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Извлечение импортов"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        
        return imports
    
    def _extract_functions(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Извлечение функций"""
        functions = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    "name": node.name,
                    "line_start": node.lineno,
                    "line_end": node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "decorators": [self._get_decorator_name(dec) for dec in node.decorator_list],
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                    "docstring": ast.get_docstring(node),
                    "complexity": self._calculate_function_complexity(node)
                }
                
                # Извлекаем тело функции
                if node.lineno <= len(lines):
                    start_line = node.lineno - 1
                    end_line = getattr(node, 'end_lineno', len(lines)) - 1
                    func_info["body"] = '\n'.join(lines[start_line:end_line + 1])
                
                functions.append(func_info)
        
        return functions
    
    def _extract_classes(self, tree: ast.AST, content: str) -> List[Dict[str, Any]]:
        """Извлечение классов"""
        classes = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "line_start": node.lineno,
                    "line_end": getattr(node, 'end_lineno', node.lineno),
                    "bases": [self._get_base_name(base) for base in node.bases],
                    "decorators": [self._get_decorator_name(dec) for dec in node.decorator_list],
                    "methods": [],
                    "docstring": ast.get_docstring(node)
                }
                
                # Извлекаем методы класса
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_info["methods"].append({
                            "name": item.name,
                            "line_start": item.lineno,
                            "args": [arg.arg for arg in item.args.args],
                            "is_property": any(
                                self._get_decorator_name(dec) == "property" 
                                for dec in item.decorator_list
                            )
                        })
                
                classes.append(class_info)
        
        return classes
    
    def _extract_variables(self, tree: ast.AST) -> List[str]:
        """Извлечение переменных верхнего уровня"""
        variables = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append(target.id)
        
        return variables
    
    def _extract_comments(self, content: str) -> List[str]:
        """Извлечение комментариев"""
        comments = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                comments.append(line)
        
        return comments
    
    def _calculate_python_complexity(self, tree: ast.AST, content: str) -> float:
        """Расчет сложности Python кода"""
        complexity = 0
        
        # Цикломатическая сложность
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        # Глубина вложенности
        max_depth = self._calculate_nesting_depth(tree)
        
        # Базовая сложность по строкам
        base_complexity = self._calculate_complexity_score(content, "python")
        
        # Итоговая оценка
        total_complexity = (complexity * 0.5 + max_depth * 0.3 + base_complexity * 0.2)
        return min(total_complexity, 10.0)
    
    def _calculate_nesting_depth(self, tree: ast.AST) -> int:
        """Расчет максимальной глубины вложенности"""
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            
            for child in ast.iter_child_nodes(node):
                child_depth = current_depth
                if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                    child_depth += 1
                
                max_depth = max(max_depth, get_depth(child, child_depth))
            
            return max_depth
        
        return get_depth(tree)
    
    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Расчет сложности отдельной функции"""
        complexity = 1  # Базовая сложность
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        
        return complexity
    
    def _find_python_security_issues(self, content: str, tree: ast.AST) -> List[str]:
        """Поиск проблем безопасности в Python коде"""
        issues = self._find_security_issues(content, "python")
        
        # Специфичные для Python проверки
        for node in ast.walk(tree):
            # Небезопасные функции
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        issues.append(f"Небезопасная функция: {node.func.id}")
                    elif node.func.id == 'input' and any(arg for arg in node.args):
                        issues.append("Небезопасное использование input() с промптом")
                
                # Проверка subprocess с shell=True
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['call', 'run', 'Popen']:
                        for keyword in node.keywords:
                            if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant):
                                if keyword.value.value is True:
                                    issues.append("Небезопасное использование subprocess с shell=True")
        
        # Проверка импортов потенциально небезопасных модулей
        dangerous_imports = ['pickle', 'marshal', 'shelve', 'dill']
        for imp in self._extract_imports(tree):
            for dangerous in dangerous_imports:
                if dangerous in imp:
                    issues.append(f"Потенциально небезопасный импорт: {imp}")
        
        return issues
    
    def _get_decorator_name(self, decorator) -> str:
        """Получение имени декоратора"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        else:
            return "unknown"
    
    def _get_base_name(self, base) -> str:
        """Получение имени базового класса"""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return base.attr
        else:
            return "unknown"


class JavaScriptCodeAnalyzer(BaseCodeAnalyzer):
    """Анализатор JavaScript кода"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.js', '.jsx', '.ts', '.tsx']
    
    def analyze_file(self, file_path: Union[str, Path]) -> CodeFile:
        """Анализ JavaScript файла"""
        path = Path(file_path)
        content = self._read_file_safely(path)
        
        if content is None:
            return self._create_empty_code_file(path, "javascript")
        
        # Извлекаем информацию с помощью регулярных выражений
        imports = self._extract_js_imports(content)
        functions = self._extract_js_functions(content)
        classes = self._extract_js_classes(content)
        variables = self._extract_js_variables(content)
        comments = self._extract_js_comments(content)
        
        # Анализируем сложность и безопасность
        complexity_score = self._calculate_js_complexity(content)
        security_issues = self._find_js_security_issues(content)
        
        return CodeFile(
            path=str(path),
            language="javascript",
            size_lines=len(content.split('\n')),
            size_bytes=len(content.encode('utf-8')),
            imports=imports,
            functions=functions,
            classes=classes,
            variables=variables,
            comments=comments,
            complexity_score=complexity_score,
            security_issues=security_issues
        )
    
    def _extract_js_imports(self, content: str) -> List[str]:
        """Извлечение импортов JavaScript"""
        imports = []
        
        # ES6 imports
        import_patterns = [
            r'import\s+.*?\s+from\s+["\']([^"\']+)["\']',
            r'import\s+["\']([^"\']+)["\']',
            r'require\s*\(\s*["\']([^"\']+)["\']\s*\)'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            imports.extend(matches)
        
        return imports
    
    def _extract_js_functions(self, content: str) -> List[Dict[str, Any]]:
        """Извлечение функций JavaScript"""
        functions = []
        
        # Паттерны для разных типов функций
        function_patterns = [
            r'function\s+(\w+)\s*\(([^)]*)\)\s*{',
            r'const\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>\s*{',
            r'(\w+)\s*:\s*function\s*\(([^)]*)\)\s*{',
            r'async\s+function\s+(\w+)\s*\(([^)]*)\)\s*{',
            r'const\s+(\w+)\s*=\s*async\s*\(([^)]*)\)\s*=>\s*{'
        ]
        
        for i, pattern in enumerate(function_patterns):
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                func_name = match.group(1)
                params = match.group(2).strip() if len(match.groups()) > 1 else ""
                
                functions.append({
                    "name": func_name,
                    "args": [p.strip() for p in params.split(',') if p.strip()],
                    "is_async": "async" in match.group(0),
                    "type": ["function", "arrow", "method", "async_function", "async_arrow"][i],
                    "line": content[:match.start()].count('\n') + 1
                })
        
        return functions
    
    def _extract_js_classes(self, content: str) -> List[Dict[str, Any]]:
        """Извлечение классов JavaScript"""
        classes = []
        
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{'
        matches = re.finditer(class_pattern, content)
        
        for match in matches:
            class_name = match.group(1)
            base_class = match.group(2) if match.group(2) else None
            
            # Ищем методы в классе
            class_start = match.end()
            brace_count = 1
            class_end = class_start
            
            for i, char in enumerate(content[class_start:], class_start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        class_end = i
                        break
            
            class_body = content[class_start:class_end]
            methods = self._extract_js_methods(class_body)
            
            classes.append({
                "name": class_name,
                "base": base_class,
                "methods": methods,
                "line": content[:match.start()].count('\n') + 1
            })
        
        return classes
    
    def _extract_js_methods(self, class_body: str) -> List[Dict[str, Any]]:
        """Извлечение методов из тела класса"""
        methods = []
        
        method_patterns = [
            r'(\w+)\s*\(([^)]*)\)\s*{',
            r'async\s+(\w+)\s*\(([^)]*)\)\s*{',
            r'static\s+(\w+)\s*\(([^)]*)\)\s*{',
            r'get\s+(\w+)\s*\(\s*\)\s*{',
            r'set\s+(\w+)\s*\(([^)]*)\)\s*{'
        ]
        
        for pattern in method_patterns:
            matches = re.finditer(pattern, class_body)
            for match in matches:
                method_name = match.group(1)
                params = match.group(2).strip() if len(match.groups()) > 1 else ""
                
                method_type = "method"
                if "async" in match.group(0):
                    method_type = "async_method"
                elif "static" in match.group(0):
                    method_type = "static_method"
                elif "get" in match.group(0):
                    method_type = "getter"
                elif "set" in match.group(0):
                    method_type = "setter"
                
                methods.append({
                    "name": method_name,
                    "args": [p.strip() for p in params.split(',') if p.strip()],
                    "type": method_type
                })
        
        return methods
    
    def _extract_js_variables(self, content: str) -> List[str]:
        """Извлечение переменных JavaScript"""
        variables = []
        
        var_patterns = [
            r'var\s+(\w+)',
            r'let\s+(\w+)',
            r'const\s+(\w+)'
        ]
        
        for pattern in var_patterns:
            matches = re.findall(pattern, content)
            variables.extend(matches)
        
        return variables
    
    def _extract_js_comments(self, content: str) -> List[str]:
        """Извлечение комментариев JavaScript"""
        comments = []
        
        # Однострочные комментарии
        single_line_comments = re.findall(r'//.*$', content, re.MULTILINE)
        comments.extend(single_line_comments)
        
        # Многострочные комментарии
        multi_line_comments = re.findall(r'/\*.*?\*/', content, re.DOTALL)
        comments.extend(multi_line_comments)
        
        return comments
    
    def _calculate_js_complexity(self, content: str) -> float:
        """Расчет сложности JavaScript кода"""
        # Базовая сложность
        base_complexity = self._calculate_complexity_score(content, "javascript")
        
        # Специфичные для JS паттерны
        js_complexity_patterns = [
            r'\bif\s*\(',
            r'\bwhile\s*\(',
            r'\bfor\s*\(',
            r'\btry\s*{',
            r'\bcatch\s*\(',
            r'\.then\s*\(',
            r'\.catch\s*\(',
            r'\basync\s+',
            r'\bawait\s+'
        ]
        
        complexity_count = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in js_complexity_patterns
        )
        
        return min(base_complexity + complexity_count * 0.5, 10.0)
    
    def _find_js_security_issues(self, content: str) -> List[str]:
        """Поиск проблем безопасности в JavaScript"""
        issues = self._find_security_issues(content, "javascript")
        
        # JS-специфичные проблемы
        js_security_patterns = {
            'eval_usage': r'\beval\s*\(',
            'innerhtml_usage': r'\.innerHTML\s*=',
            'document_write': r'document\.write\s*\(',
            'unsafe_json': r'JSON\.parse\s*\([^)]*\+',
            'prototype_pollution': r'\.prototype\s*\[',
            'xss_risk': r'\.append\s*\([^)]*\+',
        }
        
        for issue_type, pattern in js_security_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"{issue_type}: найден подозрительный паттерн")
        
        return issues

class JavaCodeAnalyzer(BaseCodeAnalyzer):
    """Анализатор Java кода"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.java']
    
    def analyze_file(self, file_path: Union[str, Path]) -> CodeFile:
        """Анализ Java файла"""
        path = Path(file_path)
        content = self._read_file_safely(path)
        
        if content is None:
            return self._create_empty_code_file(path, "java")
        
        # Извлекаем информацию с помощью регулярных выражений
        imports = self._extract_java_imports(content)
        functions = self._extract_java_methods(content)
        classes = self._extract_java_classes(content)
        variables = self._extract_java_fields(content)
        comments = self._extract_java_comments(content)
        
        # Анализируем сложность и безопасность
        complexity_score = self._calculate_java_complexity(content)
        security_issues = self._find_java_security_issues(content)
        
        return CodeFile(
            path=str(path),
            language="java",
            size_lines=len(content.split('\n')),
            size_bytes=len(content.encode('utf-8')),
            imports=imports,
            functions=functions,
            classes=classes,
            variables=variables,
            comments=comments,
            complexity_score=complexity_score,
            security_issues=security_issues
        )
    
    def _extract_java_imports(self, content: str) -> List[str]:
        """Извлечение импортов Java"""
        imports = []
        
        import_pattern = r'import\s+(?:static\s+)?([a-zA-Z_][a-zA-Z0-9_.]*(?:\.\*)?)\s*;'
        matches = re.findall(import_pattern, content)
        imports.extend(matches)
        
        # Package declaration
        package_pattern = r'package\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s*;'
        package_match = re.search(package_pattern, content)
        if package_match:
            imports.insert(0, f"package:{package_match.group(1)}")
        
        return imports
    
    def _extract_java_methods(self, content: str) -> List[Dict[str, Any]]:
        """Извлечение методов Java"""
        methods = []
        
        # Паттерн для методов Java
        method_pattern = r'(?:public|private|protected|static|\s)*\s+(\w+)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[\w\s,]+)?\s*{'
        
        matches = re.finditer(method_pattern, content, re.MULTILINE)
        for match in matches:
            return_type = match.group(1)
            method_name = match.group(2)
            params = match.group(3).strip()
            
            # Извлекаем модификаторы
            method_line = content[:match.start()].split('\n')[-1] + match.group(0)
            modifiers = []
            if 'public' in method_line:
                modifiers.append('public')
            if 'private' in method_line:
                modifiers.append('private')
            if 'protected' in method_line:
                modifiers.append('protected')
            if 'static' in method_line:
                modifiers.append('static')
            if 'final' in method_line:
                modifiers.append('final')
            if 'abstract' in method_line:
                modifiers.append('abstract')
            
            # Парсим параметры
            param_list = []
            if params:
                for param in params.split(','):
                    param = param.strip()
                    if param:
                        parts = param.split()
                        if len(parts) >= 2:
                            param_list.append(parts[-1])  # Имя параметра
            
            methods.append({
                "name": method_name,
                "return_type": return_type,
                "args": param_list,
                "modifiers": modifiers,
                "line": content[:match.start()].count('\n') + 1,
                "is_constructor": method_name[0].isupper() and return_type == method_name
            })
        
        return methods
    
    def _extract_java_classes(self, content: str) -> List[Dict[str, Any]]:
        """Извлечение классов Java"""
        classes = []
        
        # Паттерн для классов и интерфейсов
        class_pattern = r'(?:public|private|protected|\s)*\s*(?:abstract\s+)?(?:class|interface|enum)\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w\s,]+))?\s*{'
        
        matches = re.finditer(class_pattern, content, re.MULTILINE)
        for match in matches:
            class_name = match.group(1)
            extends_class = match.group(2) if match.group(2) else None
            implements_list = []
            
            if match.group(3):
                implements_list = [impl.strip() for impl in match.group(3).split(',')]
            
            # Определяем тип (class, interface, enum)
            class_line = content[:match.start()].split('\n')[-1] + match.group(0)
            class_type = "class"
            if "interface" in class_line:
                class_type = "interface"
            elif "enum" in class_line:
                class_type = "enum"
            
            # Извлекаем модификаторы
            modifiers = []
            if 'public' in class_line:
                modifiers.append('public')
            if 'private' in class_line:
                modifiers.append('private')
            if 'protected' in class_line:
                modifiers.append('protected')
            if 'abstract' in class_line:
                modifiers.append('abstract')
            if 'final' in class_line:
                modifiers.append('final')
            
            classes.append({
                "name": class_name,
                "type": class_type,
                "extends": extends_class,
                "implements": implements_list,
                "modifiers": modifiers,
                "line": content[:match.start()].count('\n') + 1
            })
        
        return classes
    
    def _extract_java_fields(self, content: str) -> List[str]:
        """Извлечение полей Java"""
        fields = []
        
        # Паттерн для полей класса
        field_pattern = r'(?:public|private|protected|static|final|\s)+\s+\w+\s+(\w+)\s*(?:=|;)'
        
        matches = re.findall(field_pattern, content, re.MULTILINE)
        fields.extend(matches)
        
        return fields
    
    def _extract_java_comments(self, content: str) -> List[str]:
        """Извлечение комментариев Java"""
        comments = []
        
        # Однострочные комментарии
        single_line_comments = re.findall(r'//.*$', content, re.MULTILINE)
        comments.extend(single_line_comments)
        
        # Многострочные комментарии
        multi_line_comments = re.findall(r'/\*.*?\*/', content, re.DOTALL)
        comments.extend(multi_line_comments)
        
        # JavaDoc комментарии
        javadoc_comments = re.findall(r'/\*\*.*?\*/', content, re.DOTALL)
        comments.extend(javadoc_comments)
        
        return comments
    # Добавить в конец файла src/tools/code_analyzer.py перед __all__

        
    def _calculate_java_complexity(self, content: str) -> float:
        """Расчет сложности Java кода"""
        base_complexity = self._calculate_complexity_score(content, "java")
        
        java_complexity_patterns = [
            r'\bif\s*\(',
            r'\bwhile\s*\(',
            r'\bfor\s*\(',
            r'\btry\s*{',
            r'\bcatch\s*\(',
            r'\bswitch\s*\(',
            r'\bcase\s+',
            r'\?\s*.*\s*:'  # тернарный оператор
        ]
        
        complexity_count = sum(
            len(re.findall(pattern, content, re.IGNORECASE))
            for pattern in java_complexity_patterns
        )
        
        return min(base_complexity + complexity_count * 0.5, 10.0)
    
    def _find_java_security_issues(self, content: str) -> List[str]:
        """Поиск проблем безопасности в Java"""
        issues = self._find_security_issues(content, "java")
        
        java_security_patterns = {
            'sql_injection': r'Statement\.execute\s*\([^)]*\+',
            'path_traversal': r'new\s+File\s*\([^)]*\+',
            'reflection_usage': r'Class\.forName\s*\(',
            'serialization': r'ObjectInputStream\s*\(',
            'random_weak': r'new\s+Random\s*\(',
            'hardcoded_crypto': r'AES|DES|MD5|SHA1'
        }
        
        for issue_type, pattern in java_security_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"{issue_type}: найден подозрительный паттерн")
        
        return issues
    
    def _create_empty_code_file(self, path: Path, language: str) -> CodeFile:
        """Создание пустого объекта CodeFile для ошибок"""
        return CodeFile(
            path=str(path),
            language=language,
            size_lines=0,
            size_bytes=0,
            imports=[],
            functions=[],
            classes=[],
            variables=[],
            comments=[],
            complexity_score=0.0,
            security_issues=["Ошибка чтения файла"]
        )


class ConfigurationAnalyzer:
    """Анализатор конфигурационных файлов"""
    
    def __init__(self):
        self.logger = get_logger()
        self.supported_extensions = ['.json', '.yaml', '.yml', '.toml', '.ini', '.env']
    
    def can_analyze(self, file_path: Union[str, Path]) -> bool:
        """Проверка, может ли анализатор обработать файл"""
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions
    
    def analyze_config_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Анализ конфигурационного файла"""
        path = Path(file_path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            config_info = {
                "file_path": str(path),
                "file_type": path.suffix.lower(),
                "size": len(content),
                "settings": self._extract_settings(content, path.suffix.lower()),
                "security_issues": self._find_config_security_issues(content)
            }
            
            return config_info
            
        except Exception as e:
            return {
                "file_path": str(path),
                "file_type": path.suffix.lower(),
                "error": str(e)
            }
    
    def _extract_settings(self, content: str, file_type: str) -> Dict[str, Any]:
        """Извлечение настроек из конфигурации"""
        settings = {}
        
        if file_type == '.json':
            try:
                import json
                settings = json.loads(content)
            except json.JSONDecodeError:
                pass
        elif file_type in ['.yaml', '.yml']:
            # Простой парсинг YAML без библиотеки
            lines = content.split('\n')
            for line in lines:
                if ':' in line and not line.strip().startswith('#'):
                    key, value = line.split(':', 1)
                    settings[key.strip()] = value.strip()
        elif file_type == '.env':
            lines = content.split('\n')
            for line in lines:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.split('=', 1)
                    settings[key.strip()] = value.strip()
        
        return settings
    
    def _find_config_security_issues(self, content: str) -> List[str]:
        """Поиск проблем безопасности в конфигурации"""
        issues = []
        
        # Поиск секретов в конфигурации
        secret_patterns = [
            r'password\s*[:=]\s*["\']?[^"\'\\n]+["\']?',
            r'secret\s*[:=]\s*["\']?[^"\'\\n]+["\']?',
            r'api_key\s*[:=]\s*["\']?[^"\'\\n]+["\']?',
            r'token\s*[:=]\s*["\']?[^"\'\\n]+["\']?'
        ]
        
        for pattern in secret_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append("hardcoded_secrets: найдены потенциальные секреты")
                break
        
        # Проверка небезопасных настроек
        if re.search(r'debug\s*[:=]\s*true', content, re.IGNORECASE):
            issues.append("debug_mode: включен режим отладки")
        
        if re.search(r'ssl\s*[:=]\s*false', content, re.IGNORECASE):
            issues.append("ssl_disabled: отключен SSL")
        
        return issues


class CodeAnalyzer:
    """Главный класс для анализа кодовой базы проекта"""
    
    def __init__(self):
        self.code_analyzers = [
            PythonCodeAnalyzer(),
            JavaScriptCodeAnalyzer(),
            JavaCodeAnalyzer()
        ]
        self.config_analyzer = ConfigurationAnalyzer()
        self.logger = get_logger()
    
    def analyze_project(
        self, 
        project_path: Union[str, Path], 
        max_files: int = 100,
        exclude_dirs: Optional[List[str]] = None
    ) -> CodeAnalysisResult:
        """Анализ всего проекта"""
        start_time = datetime.now()
        project_path = Path(project_path)
        
        if exclude_dirs is None:
            exclude_dirs = ['node_modules', '.git', '__pycache__', '.venv', 'venv', 'target', '.idea']
        
        try:
            with LogContext("analyze_project", "code_analyzer", "code_analyzer"):
                # Сканируем файлы
                all_files = []
                for file_path in project_path.rglob('*'):
                    if file_path.is_file() and not any(exclude_dir in str(file_path) for exclude_dir in exclude_dirs):
                        all_files.append(file_path)
                
                # Ограничиваем количество файлов
                if len(all_files) > max_files:
                    all_files = all_files[:max_files]
                
                # Анализируем файлы
                analyzed_files = []
                total_lines = 0
                languages = {}
                dependencies = {}
                entry_points = []
                config_files = []
                
                for file_path in all_files:
                    # Код файлы
                    for analyzer in self.code_analyzers:
                        if analyzer.can_analyze(file_path):
                            code_file = analyzer.analyze_file(file_path)
                            analyzed_files.append(code_file)
                            total_lines += code_file.size_lines
                            
                            # Статистика языков
                            if code_file.language not in languages:
                                languages[code_file.language] = 0
                            languages[code_file.language] += 1
                            
                            # Зависимости
                            if code_file.imports:
                                dependencies[str(file_path)] = code_file.imports
                            
                            # Точки входа
                            if self._is_entry_point(code_file):
                                entry_points.append(str(file_path))
                            
                            break
                    
                    # Конфигурационные файлы
                    if self.config_analyzer.can_analyze(file_path):
                        config_files.append(str(file_path))
                
                # Анализируем безопасность и сложность
                security_summary = self._analyze_security_summary(analyzed_files)
                complexity_summary = self._analyze_complexity_summary(analyzed_files)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return CodeAnalysisResult(
                    project_path=str(project_path),
                    total_files=len(analyzed_files),
                    total_lines=total_lines,
                    languages=languages,
                    files=analyzed_files,
                    dependencies=dependencies,
                    entry_points=entry_points,
                    configuration_files=config_files,
                    security_summary=security_summary,
                    complexity_summary=complexity_summary,
                    analysis_time=execution_time,
                    success=True
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return CodeAnalysisResult(
                project_path=str(project_path),
                total_files=0,
                total_lines=0,
                languages={},
                files=[],
                dependencies={},
                entry_points=[],
                configuration_files=[],
                security_summary={},
                complexity_summary={},
                analysis_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    def _is_entry_point(self, code_file: CodeFile) -> bool:
        """Определение, является ли файл точкой входа"""
        file_path = Path(code_file.path)
        
        # Общие точки входа
        entry_point_names = ['main.py', 'app.py', 'index.js', 'main.js', 'Main.java']
        if file_path.name in entry_point_names:
            return True
        
        # Наличие main функции
        for func in code_file.functions:
            if func.get('name') == 'main' or func.get('name') == '__main__':
                return True
        
        return False
    
    def _analyze_security_summary(self, files: List[CodeFile]) -> Dict[str, Any]:
        """Анализ общей безопасности проекта"""
        all_issues = []
        for file in files:
            all_issues.extend(file.security_issues)
        
        # Группируем по типам
        issue_types = {}
        for issue in all_issues:
            issue_type = issue.split(':')[0] if ':' in issue else 'unknown'
            if issue_type not in issue_types:
                issue_types[issue_type] = 0
            issue_types[issue_type] += 1
        
        return {
            "total_issues": len(all_issues),
            "issue_types": issue_types,
            "high_risk_files": [f.path for f in files if len(f.security_issues) > 3],
            "security_score": max(0, 10 - len(all_issues))  # Чем меньше проблем, тем выше балл
        }
    
    def _analyze_complexity_summary(self, files: List[CodeFile]) -> Dict[str, Any]:
        """Анализ общей сложности проекта"""
        if not files:
            return {"average_complexity": 0, "max_complexity": 0, "complex_files": []}
        
        complexities = [f.complexity_score for f in files]
        average_complexity = sum(complexities) / len(complexities)
        max_complexity = max(complexities)
        
        # Файлы с высокой сложностью
        complex_files = [f.path for f in files if f.complexity_score > 7]
        
        return {
            "average_complexity": average_complexity,
            "max_complexity": max_complexity,
            "complex_files": complex_files,
            "total_functions": sum(len(f.functions) for f in files),
            "total_classes": sum(len(f.classes) for f in files)
        }


def _analyze_tech_stack(dependencies: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Анализ технологического стека"""
    tech_categories = {
        "ml_frameworks": [],
        "web_frameworks": [],
        "databases": [],
        "api_clients": [],
        "security": [],
        "testing": [],
        "java_frameworks": [],
        "other": []
    }
    
    # Добавить Java библиотеки в существующие категории
    ml_libs = {
        'tensorflow', 'torch', 'pytorch', 'sklearn', 'scikit-learn', 'numpy', 'pandas', 
        'transformers', 'langchain', 'openai', 'anthropic', 'huggingface', 'keras', 
        'lightgbm', 'xgboost',
        # Java ML библиотеки
        'weka', 'deeplearning4j', 'mallet', 'mahout', 'smile'
    }
    
    web_libs = {
        'flask', 'django', 'fastapi', 'express', 'react', 'vue', 'angular', 'nodejs', 
        'koa', 'nest', 'nextjs',
        # Java web фреймворки
        'spring', 'hibernate', 'struts', 'wicket', 'vaadin', 'jsf'
    }
    
    # Новая категория Java фреймворков
    java_frameworks = {
        'spring', 'springboot', 'hibernate', 'struts', 'jsf', 'wicket', 'vaadin', 
        'grails', 'play', 'akka', 'vertx'
    }
    
    db_libs = {'sqlite3', 'psycopg2', 'pymongo', 'sqlalchemy', 'mysql', 'postgresql'}
    api_libs = {'requests', 'httpx', 'aiohttp', 'urllib3', 'axios', 'fetch'}
    security_libs = {'cryptography', 'bcrypt', 'jwt', 'passlib', 'ssl'}
    test_libs = {'pytest', 'unittest', 'jest', 'mocha', 'junit'}
    
    # Анализируем все зависимости
    all_deps = []
    for deps_list in dependencies.values():
        all_deps.extend(deps_list)
    
    for dep in set(all_deps):
        dep_lower = dep.lower()
        
        if any(ml_lib in dep_lower for ml_lib in ml_libs):
            tech_categories["ml_frameworks"].append(dep)
        elif any(web_lib in dep_lower for web_lib in web_libs):
            tech_categories["web_frameworks"].append(dep)
        elif any(java_fw in dep_lower for java_fw in java_frameworks):
            tech_categories["java_frameworks"].append(dep)
        elif any(db_lib in dep_lower for db_lib in db_libs):
            tech_categories["databases"].append(dep)
        elif any(api_lib in dep_lower for api_lib in api_libs):
            tech_categories["api_clients"].append(dep)
        elif any(sec_lib in dep_lower for sec_lib in security_libs):
            tech_categories["security"].append(dep)
        elif any(test_lib in dep_lower for test_lib in test_libs):
            tech_categories["testing"].append(dep)
        else:
            tech_categories["other"].append(dep)
    
    return tech_categories


# ===============================
# Утилитарные функции
# ===============================

def create_code_analyzer() -> CodeAnalyzer:
    """Фабрика для создания анализатора кода"""
    return CodeAnalyzer()


def analyze_agent_codebase(
    project_path: Union[str, Path], 
    max_files: int = 100
) -> CodeAnalysisResult:
    """
    Удобная функция для анализа кодовой базы агента
    
    Args:
        project_path: Путь к проекту
        max_files: Максимальное количество файлов для анализа
    
    Returns:
        Результат анализа кода
    """
    analyzer = create_code_analyzer()
    return analyzer.analyze_project(project_path, max_files)


def extract_agent_technical_info(analysis_result: CodeAnalysisResult) -> Dict[str, Any]:
    """Извлечение технической информации об агенте"""
    if not analysis_result.success:
        return {
            "success": False,
            "error": analysis_result.error_message,
            "analysis_time": analysis_result.analysis_time
        }
    
    # Анализируем технологический стек
    tech_stack = _analyze_tech_stack(analysis_result.dependencies)
    
    return {
        "success": True,
        "project_path": analysis_result.project_path,
        "total_files": analysis_result.total_files,
        "total_lines": analysis_result.total_lines,
        "languages": analysis_result.languages,
        "tech_stack": tech_stack,
        "entry_points": analysis_result.entry_points,
        "configuration_files": analysis_result.configuration_files,
        "security_summary": analysis_result.security_summary,
        "complexity_summary": analysis_result.complexity_summary,
        "analysis_time": analysis_result.analysis_time
    }


def get_code_analysis_summary(analysis_result: CodeAnalysisResult) -> Dict[str, Any]:
    """Получение краткой сводки анализа кода"""
    if not analysis_result.success:
        return {
            "success": False,
            "error": analysis_result.error_message
        }
    
    return {
        "success": True,
        "total_files": analysis_result.total_files,
        "total_lines": analysis_result.total_lines,
        "languages": list(analysis_result.languages.keys()),
        "avg_complexity": analysis_result.complexity_summary.get("average_complexity", 0),
        "security_score": analysis_result.security_summary.get("security_score", 0),
        "analysis_time": analysis_result.analysis_time
    }


# Экспорт основных классов и функций
__all__ = [
    "CodeAnalyzer",
    "CodeAnalysisResult", 
    "CodeFile",
    "CodeAnalyzerError",
    "PythonCodeAnalyzer",
    "JavaScriptCodeAnalyzer", 
    "JavaCodeAnalyzer",
    "ConfigurationAnalyzer",
    "create_code_analyzer",
    "analyze_agent_codebase",
    "extract_agent_technical_info",
    "get_code_analysis_summary"
]