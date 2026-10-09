"""AST-based static extractor for FORGE / wien2k_gen.

Does not import or execute application modules. All resolution is syntactic.
"""

from __future__ import annotations

import ast
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from .stdlib_modules import is_stdlib

INTERNAL_ROOT = "forge"


@dataclass
class ImportRecord:
    module: str
    names: list[str]
    alias: Optional[str]
    level: int
    lineno: int
    col: int
    kind: str  # absolute | relative
    context: str  # top_level | function | class | conditional | type_checking
    resolved: Optional[str] = None
    category: str = "unresolved"  # internal | stdlib | third_party | unresolved


@dataclass
class FunctionRecord:
    name: str
    qualname: str
    lineno: int
    end_lineno: Optional[int]
    args: list[str]
    decorators: list[str]
    is_method: bool
    is_nested: bool
    is_async: bool
    class_name: Optional[str] = None


@dataclass
class ClassRecord:
    name: str
    qualname: str
    lineno: int
    end_lineno: Optional[int]
    bases: list[str]
    decorators: list[str]
    methods: list[str]


@dataclass
class CallRecord:
    lineno: int
    col: int
    raw: str
    kind: str  # function | method | attribute | constructor | super | unresolved
    caller: str
    callee_name: str
    resolved: Optional[str]
    confidence: str  # verified | partial | unresolved


@dataclass
class ExternalExecRecord:
    lineno: int
    mechanism: str
    callee: str
    args_preview: str
    caller: str


@dataclass
class ModuleRecord:
    path: str
    module: str
    category: str
    package: str
    has_main_guard: bool
    syntax_error: Optional[str]
    classes: list[ClassRecord] = field(default_factory=list)
    functions: list[FunctionRecord] = field(default_factory=list)
    imports: list[ImportRecord] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    external_exec: list[ExternalExecRecord] = field(default_factory=list)
    docstring: Optional[str] = None
    loc: int = 0


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def path_to_module(rel_path: str) -> str:
    p = Path(rel_path)
    parts = list(p.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__.py":
        parts = parts[:-1]
    elif parts:
        parts[-1] = Path(parts[-1]).stem
    return ".".join(parts)


def classify_file(rel_path: str) -> str:
    if rel_path.startswith("src/"):
        return "production"
    if rel_path.startswith("tests/"):
        return "test"
    if rel_path.startswith("docs/"):
        return "documentation"
    if rel_path.startswith("examples/"):
        return "example"
    if rel_path.startswith("tools/"):
        return "tool"
    name = Path(rel_path).name
    if name in {
        "pyproject.toml",
        "environment.yml",
        "docker-compose.yml",
        "Dockerfile",
        "Singularity.def",
        "Makefile",
        ".pre-commit-config.yaml",
        ".gitignore",
        ".dockerignore",
        "CITATION.cff",
    }:
        return "configuration"
    if rel_path.startswith(".github/"):
        return "ci"
    if rel_path.startswith("offline_packages/"):
        return "data"
    if rel_path.endswith(".py"):
        return "script"
    return "other"


def _decorator_name(node: ast.AST) -> str:
    return ast.unparse(node) if hasattr(ast, "unparse") else type(node).__name__


def _base_name(node: ast.AST) -> str:
    return ast.unparse(node) if hasattr(ast, "unparse") else type(node).__name__


def _call_raw(node: ast.Call) -> str:
    try:
        return ast.unparse(node.func)
    except Exception:
        return type(node.func).__name__


def _is_under_type_checking(stack: list[ast.AST]) -> bool:
    for n in stack:
        if isinstance(n, ast.If):
            test = n.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                return True
            if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
                return True
    return False


def _is_conditional(stack: list[ast.AST]) -> bool:
    return any(isinstance(n, (ast.If, ast.Try, ast.ExceptHandler)) for n in stack)


def _enclosing_function(stack: list[ast.AST]) -> Optional[str]:
    names: list[str] = []
    for n in stack:
        if isinstance(n, ast.ClassDef):
            names.append(n.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(n.name)
    return ".".join(names) if names else "<module>"


def _enclosing_class(stack: list[ast.AST]) -> Optional[str]:
    for n in reversed(stack):
        if isinstance(n, ast.ClassDef):
            return n.name
    return None


def resolve_relative(
    current_module: str,
    level: int,
    imported: Optional[str],
    is_package: bool = False,
) -> Optional[str]:
    if level <= 0:
        return imported
    parts = current_module.split(".")
    # Regular modules live inside a package; __init__.py *is* the package.
    package_parts = parts if is_package else parts[:-1]
    if level - 1 > len(package_parts):
        return None
    base = package_parts[: len(package_parts) - (level - 1)]
    if imported:
        base.extend(imported.split("."))
    return ".".join(base) if base else imported


def classify_import(resolved: Optional[str], known_internal: set[str]) -> str:
    if not resolved:
        return "unresolved"
    top = resolved.split(".", 1)[0]
    if top == INTERNAL_ROOT or resolved in known_internal or any(
        resolved == m or resolved.startswith(m + ".") for m in known_internal
    ):
        return "internal"
    if is_stdlib(resolved):
        return "stdlib"
    return "third_party"


SUBPROCESS_ATTRS = {
    "run",
    "Popen",
    "call",
    "check_call",
    "check_output",
    "getoutput",
    "getstatusoutput",
}
OS_EXEC_ATTRS = {"system", "popen", "execl", "execle", "execlp", "execv", "execve", "spawnv", "spawnve"}
SHUTIL_EXEC_ATTRS = {"which"}


class ModuleVisitor(ast.NodeVisitor):
    def __init__(self, module: str, known_internal: set[str], is_package: bool = False) -> None:
        self.module = module
        self.known_internal = known_internal
        self.is_package = is_package
        self.stack: list[ast.AST] = []
        self.classes: list[ClassRecord] = []
        self.functions: list[FunctionRecord] = []
        self.imports: list[ImportRecord] = []
        self.calls: list[CallRecord] = []
        self.external_exec: list[ExternalExecRecord] = []
        self.has_main_guard = False
        self._aliases: dict[str, str] = {}
        self._local_defs: set[str] = set()
        self._class_methods: dict[str, set[str]] = defaultdict(set)

    def generic_visit(self, node: ast.AST) -> None:
        self.stack.append(node)
        super().generic_visit(node)
        self.stack.pop()

    def visit_If(self, node: ast.If) -> None:
        test = node.test
        if isinstance(test, ast.Compare):
            left = test.left
            if isinstance(left, ast.Name) and left.id == "__name__":
                self.has_main_guard = True
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        context = self._import_context()
        for alias in node.names:
            name = alias.name
            asname = alias.asname
            rec = ImportRecord(
                module=name,
                names=["*"] if asname is None else [asname],
                alias=asname,
                level=0,
                lineno=node.lineno,
                col=node.col_offset,
                kind="absolute",
                context=context,
                resolved=name,
            )
            rec.category = classify_import(name, self.known_internal)
            self.imports.append(rec)
            self._aliases[asname or name.split(".", 1)[0]] = name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        context = self._import_context()
        imported_mod = node.module
        resolved = resolve_relative(self.module, node.level, imported_mod, is_package=self.is_package)
        names = [a.name for a in node.names]
        rec = ImportRecord(
            module=imported_mod or "",
            names=names,
            alias=None,
            level=node.level,
            lineno=node.lineno,
            col=node.col_offset,
            kind="relative" if node.level else "absolute",
            context=context,
            resolved=resolved,
        )
        rec.category = classify_import(resolved, self.known_internal)
        self.imports.append(rec)
        if resolved:
            for a in node.names:
                local = a.asname or a.name
                if a.name == "*":
                    continue
                self._aliases[local] = f"{resolved}.{a.name}"
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        cls_name = self._qual(node.name)
        methods = [
            n.name
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        rec = ClassRecord(
            name=node.name,
            qualname=cls_name,
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", None),
            bases=[_base_name(b) for b in node.bases],
            decorators=[_decorator_name(d) for d in node.decorator_list],
            methods=methods,
        )
        self.classes.append(rec)
        self._class_methods[node.name] = set(methods)
        self._local_defs.add(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node, is_async=False)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_function(node, is_async=True)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        caller = f"{self.module}:{_enclosing_function(self.stack)}"
        raw = _call_raw(node)
        kind, callee, resolved, confidence = self._resolve_call(node)
        self.calls.append(
            CallRecord(
                lineno=node.lineno,
                col=node.col_offset,
                raw=raw,
                kind=kind,
                caller=caller,
                callee_name=callee,
                resolved=resolved,
                confidence=confidence,
            )
        )
        self._maybe_external(node, caller, raw)
        self.generic_visit(node)

    def _record_function(self, node: Any, is_async: bool) -> None:
        class_name = _enclosing_class(self.stack)
        nested = any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in self.stack
        )
        args = [a.arg for a in node.args.args]
        rec = FunctionRecord(
            name=node.name,
            qualname=self._qual(node.name if not class_name else f"{class_name}.{node.name}"),
            lineno=node.lineno,
            end_lineno=getattr(node, "end_lineno", None),
            args=args,
            decorators=[_decorator_name(d) for d in node.decorator_list],
            is_method=class_name is not None,
            is_nested=nested,
            is_async=is_async,
            class_name=class_name,
        )
        self.functions.append(rec)
        self._local_defs.add(node.name)

    def _qual(self, name: str) -> str:
        return f"{self.module}.{name}"

    def _import_context(self) -> str:
        if _is_under_type_checking(self.stack):
            return "type_checking"
        if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in self.stack):
            base = "function"
        elif any(isinstance(n, ast.ClassDef) for n in self.stack):
            base = "class"
        else:
            base = "top_level"
        if _is_conditional(self.stack) and base == "top_level":
            return "conditional"
        if _is_conditional(self.stack):
            return f"conditional_{base}"
        return base

    def _resolve_call(
        self, node: ast.Call
    ) -> tuple[str, str, Optional[str], str]:
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
            if name in self._aliases:
                target = self._aliases[name]
                kind = "constructor" if name[:1].isupper() else "function"
                return kind, name, target, "verified"
            if name in self._local_defs:
                kind = "constructor" if name[:1].isupper() else "function"
                return kind, name, f"{self.module}.{name}", "verified"
            kind = "constructor" if name[:1].isupper() else "function"
            return kind, name, None, "unresolved"
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "super":
                return "super", func.attr, None, "partial"
            if isinstance(func.value, ast.Call) and isinstance(func.value.func, ast.Name) and func.value.func.id == "super":
                return "super", func.attr, None, "partial"
            if isinstance(func.value, ast.Name) and func.value.id == "self":
                cls = _enclosing_class(self.stack)
                if cls and func.attr in self._class_methods.get(cls, set()):
                    return "method", func.attr, f"{self.module}.{cls}.{func.attr}", "verified"
                return "method", func.attr, None, "partial"
            if isinstance(func.value, ast.Name) and func.value.id in self._aliases:
                prefix = self._aliases[func.value.id]
                return "attribute", func.attr, f"{prefix}.{func.attr}", "verified"
            raw = _call_raw(node)
            return "attribute", func.attr, None, "unresolved"
        return "unresolved", _call_raw(node), None, "unresolved"

    def _maybe_external(self, node: ast.Call, caller: str, raw: str) -> None:
        func = node.func
        mechanism = None
        callee = raw
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            owner = func.value.id
            attr = func.attr
            if owner in {"subprocess"} and attr in SUBPROCESS_ATTRS:
                mechanism = f"subprocess.{attr}"
            elif owner in {"os"} and attr in OS_EXEC_ATTRS:
                mechanism = f"os.{attr}"
            elif owner in {"shutil"} and attr in SHUTIL_EXEC_ATTRS:
                mechanism = f"shutil.{attr}"
        if mechanism is None:
            return
        try:
            preview = ast.unparse(node)[:200]
        except Exception:
            preview = mechanism
        self.external_exec.append(
            ExternalExecRecord(
                lineno=node.lineno,
                mechanism=mechanism,
                callee=callee,
                args_preview=preview,
                caller=caller,
            )
        )


def parse_python_file(
    path: Path, root: Path, known_internal: set[str]
) -> ModuleRecord:
    rel = _rel(path, root)
    module = path_to_module(rel)
    category = classify_file(rel)
    text = ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return ModuleRecord(
            path=rel,
            module=module,
            category=category,
            package=module.rsplit(".", 1)[0] if "." in module else module,
            has_main_guard=False,
            syntax_error=f"unreadable: {e}",
        )
    loc = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError as e:
        return ModuleRecord(
            path=rel,
            module=module,
            category=category,
            package=module.rsplit(".", 1)[0] if "." in module else module,
            has_main_guard=False,
            syntax_error=f"{e.msg} (line {e.lineno})",
            loc=loc,
        )
    visitor = ModuleVisitor(module, known_internal, is_package=path.name == "__init__.py")
    visitor.visit(tree)
    package = module if path.name == "__init__.py" else (
        module.rsplit(".", 1)[0] if "." in module else module
    )
    return ModuleRecord(
        path=rel,
        module=module,
        category=category,
        package=package,
        has_main_guard=visitor.has_main_guard,
        syntax_error=None,
        classes=visitor.classes,
        functions=visitor.functions,
        imports=visitor.imports,
        calls=visitor.calls,
        external_exec=visitor.external_exec,
        docstring=ast.get_docstring(tree),
        loc=loc,
    )


def collect_python_files(root: Path, exclude_dir_names: set[str]) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dir_names]
        for fn in filenames:
            if fn.endswith(".py"):
                files.append(Path(dirpath) / fn)
    files.sort()
    return files


def analyze_repository(root: Path, exclude_dir_names: set[str]) -> dict[str, Any]:
    py_files = collect_python_files(root, exclude_dir_names)
    known_internal: set[str] = set()
    for p in py_files:
        rel = _rel(p, root)
        if rel.startswith("src/"):
            known_internal.add(path_to_module(rel))

    modules: list[ModuleRecord] = []
    for p in py_files:
        modules.append(parse_python_file(p, root, known_internal))

    import_edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for m in modules:
        for imp in m.imports:
            if imp.category != "internal" or not imp.resolved:
                continue
            src = m.module
            dst = _normalize_internal(imp.resolved, known_internal)
            key = (src, dst, imp.context)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            import_edges.append(
                {
                    "from": src,
                    "to": dst,
                    "context": imp.context,
                    "kind": imp.kind,
                    "lineno": imp.lineno,
                    "path": m.path,
                    "confidence": "verified",
                }
            )

    return {
        "modules": [module_to_dict(m) for m in modules],
        "import_edges": import_edges,
        "known_internal": sorted(known_internal),
    }


def _normalize_internal(resolved: str, known: set[str]) -> str:
    if resolved in known:
        return resolved
    # forge.backends.wien2k may resolve to package
    parts = resolved.split(".")
    while len(parts) > 1:
        cand = ".".join(parts)
        if cand in known:
            return cand
        parts = parts[:-1]
    return resolved


def module_to_dict(m: ModuleRecord) -> dict[str, Any]:
    return {
        "path": m.path,
        "module": m.module,
        "category": m.category,
        "package": m.package,
        "has_main_guard": m.has_main_guard,
        "syntax_error": m.syntax_error,
        "docstring": (m.docstring or "").split("\n", 1)[0][:240] if m.docstring else "",
        "loc": m.loc,
        "classes": [asdict(c) for c in m.classes],
        "functions": [asdict(f) for f in m.functions],
        "imports": [asdict(i) for i in m.imports],
        "calls": [asdict(c) for c in m.calls],
        "external_exec": [asdict(e) for e in m.external_exec],
    }


def dump_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
