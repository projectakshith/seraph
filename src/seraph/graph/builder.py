"""Code graph for one indexed commit: definitions, calls, imports, inheritance and references.

References are extracted per file at index time (`extract_refs`) and stored with the index, so
unchanged files are reused across commits. Names are resolved across files when the graph is
built (`build_graph`). Symbol ids are `path::symbol`; a file is its own node, keyed by its path.
"""

from __future__ import annotations

import ast
import posixpath
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from seraph.memory import InMemoryCodeGraph
from seraph.types import EdgeKind

Span = tuple[str | None, str, int, int, str]  # (symbol, kind, start, end, text) from `index._chunks`

_TS_SUFFIXES = (".ts", ".tsx", ".js", ".jsx")
_LANGUAGE_SUFFIXES = {
    ".py": "python", ".go": "go", ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".mts": "typescript", ".cts": "typescript", ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hh": "cpp", ".hpp": "cpp", ".hxx": "cpp", ".ipp": "cpp", ".inl": "cpp",
    ".java": "java", ".cs": "c_sharp", ".rs": "rust", ".php": "php",
    ".rb": "ruby", ".rake": "ruby", ".gemspec": "ruby", ".swift": "swift",
    ".kt": "kotlin", ".kts": "kotlin", ".sh": "bash", ".bash": "bash",
    ".dart": "dart", ".lua": "lua", ".sql": "sql", ".f": "fortran", ".for": "fortran",
    ".f90": "fortran", ".f95": "fortran", ".f03": "fortran", ".f08": "fortran",
    ".scala": "scala", ".sc": "scala", ".sbt": "scala",
}


@dataclass(frozen=True)
class Ref:
    """One reference as written in the source, before cross-file resolution.

    `symbol` is the enclosing chunk's symbol (None at module level). Calls, inheritance and
    references keep the name as written: `parse_config`, `json.load`, `self.save`, or `?.save`
    when the receiver is an expression. Imports carry a module (`pkg.cfg`, `src/util`,
    `example.com/x/util`), optionally `module:name`, and `local` is the name they bind.
    """

    symbol: str | None
    kind: str
    target: str
    local: str | None = None


def symbol_id(path: str, symbol: str | None) -> str:
    return f"{path}::{symbol}" if symbol else path


def _language(path: str) -> str | None:
    suffix = posixpath.splitext(path)[1].lower()
    return _LANGUAGE_SUFFIXES.get(suffix)


def _enclosing(spans: Sequence[Span], n_lines: int):
    """Line -> innermost non-file chunk symbol covering it."""
    owner: list[str | None] = [None] * (n_lines + 2)
    for symbol, kind, start, end, _ in sorted(spans, key=lambda s: s[3] - s[2]):
        if kind == "file" or symbol is None:
            continue
        for line in range(max(start, 1), min(end, n_lines + 1) + 1):
            if owner[line] is None:
                owner[line] = symbol
    return lambda line: owner[line] if 0 <= line < len(owner) else None


def extract_refs(path: str, text: str, spans: Sequence[Span]) -> list[Ref]:
    language = _language(path)
    if language is None:
        return []
    at = _enclosing(spans, text.count("\n") + 1)
    try:
        refs = _python_refs(path, text, at, spans) if language == "python" else _tree_refs(path, text, at, language)
    except Exception:
        return []
    return list(dict.fromkeys(refs))


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id, *reversed(parts)])
    return f"?.{parts[0]}" if parts else None


def _python_refs(path: str, text: str, at, spans: Sequence[Span]) -> list[Ref]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    package = path.split("/")[:-1]
    refs: list[Ref] = []
    names: list[tuple[int, str]] = []
    call_funcs: set[int] = set()
    for node in ast.walk(tree):  # breadth-first: a Call is seen before its `func`
        if isinstance(node, ast.Import):
            refs += [Ref(None, EdgeKind.IMPORTS, a.name, a.asname or a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: max(len(package) - node.level + 1, 0)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            refs += [Ref(None, EdgeKind.IMPORTS, f"{module}:{a.name}", a.asname or a.name) for a in node.names]
        elif isinstance(node, ast.Call):
            call_funcs.add(id(node.func))
            target = _dotted(node.func)
            if target:
                refs.append(Ref(at(node.lineno), EdgeKind.CALLS, target))
        elif isinstance(node, ast.ClassDef):
            for b in node.bases:
                target = _dotted(b)
                if target and not target.startswith("?"):
                    refs.append(Ref(at(node.lineno), EdgeKind.INHERITS, target))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and id(node) not in call_funcs:
            names.append((node.lineno, node.id))
    # Bare-name references only resolve through this file's definitions or imports.
    bound = {r.local for r in refs if r.local} | {s[0] for s in spans if s[0]}
    refs += [Ref(at(line), EdgeKind.REFERENCES, name) for line, name in names if name in bound]
    return refs


def _text(node) -> str:
    return node.text.decode("utf8", "replace")


def _unquote(node) -> str:
    return _text(node).strip("\"'`")


def _ts_module(directory: str, source: str) -> str:
    if not source.startswith("."):
        return source
    path = posixpath.normpath(posixpath.join(directory, source))
    root, ext = posixpath.splitext(path)
    return root if ext in _TS_SUFFIXES else path


def _member_target(fn, obj_field: str, prop_field: str) -> str | None:
    if fn is None:
        return None
    if fn.type == "identifier":
        return _text(fn)
    obj, prop = fn.child_by_field_name(obj_field), fn.child_by_field_name(prop_field)
    if prop is None:
        return None
    if obj is not None and obj.type in ("identifier", "this"):
        return f"{_text(obj)}.{_text(prop)}"
    return f"?.{_text(prop)}"


def _tree_refs(path: str, text: str, at, language: str) -> list[Ref]:
    from seraph.index import _parser, _walk

    root = _parser(language).parse(text.encode("utf8")).root_node
    directory = posixpath.dirname(path)
    refs: list[Ref] = []
    generic_imports = {
        "preproc_include", "import_declaration", "import_header", "using_directive",
        "use_declaration", "namespace_import", "include_directive",
    }
    generic_calls = {
        "call_expression", "function_call_expression", "method_invocation", "invocation_expression",
        "function_expression_invocation", "call", "function_call", "command", "subroutine_call",
    }
    for node in _walk(root):
        line = node.start_point[0] + 1
        if language == "go":
            if node.type == "import_spec":
                spec, name = node.child_by_field_name("path"), node.child_by_field_name("name")
                module = _unquote(spec)
                local = _text(name) if name is not None else module.rsplit("/", 1)[-1]
                if local != "_":
                    refs.append(Ref(None, EdgeKind.IMPORTS, module, local))
            elif node.type == "call_expression":
                target = _member_target(node.child_by_field_name("function"), "operand", "field")
                if target:
                    refs.append(Ref(at(line), EdgeKind.CALLS, target))
            continue
        if language not in {"javascript", "typescript", "tsx"}:
            if node.type in generic_imports:
                raw = _text(node).strip()
                if node.type == "preproc_include":
                    match = re.search(r"[<\"]([^>\"]+)[>\"]", raw)
                    module = match.group(1) if match else ""
                else:
                    raw = re.sub(r"^(?:import|use|using)\s+", "", raw)
                    raw = raw.rstrip(";{}").strip().strip("\"'`")
                    alias = None
                    if " as " in raw:
                        raw, alias = raw.rsplit(" as ", 1)
                    module = raw.rstrip(".*")
                    if module.startswith("package:"):
                        module = module[len("package:"):]
                    if module.startswith("crate::"):
                        module = module[len("crate::"):]
                    if module.startswith("."):
                        module = posixpath.normpath(posixpath.join(directory, module))
                    imported_name = None
                    if language == "rust" and "::" in module:
                        module, imported_name = module.rsplit("::", 1)
                    leaf = (imported_name or module.replace("\\", "/").replace("::", "/")
                            .rsplit("/", 1)[-1].rsplit(".", 1)[-1])
                    local = alias or leaf
                    if local and module:
                        target = f"{module}:{imported_name}" if imported_name else module
                        refs.append(Ref(None, EdgeKind.IMPORTS, target, local))
                if node.type == "preproc_include" and module:
                    refs.append(Ref(None, EdgeKind.IMPORTS, module, posixpath.basename(module)))
                continue
            if node.type in generic_calls:
                obj = node.child_by_field_name("object")
                member = node.child_by_field_name("name") or node.child_by_field_name("method")
                fn = (None if obj is not None and member is not None else
                      next((node.child_by_field_name(field) for field in
                            ("function", "name", "method", "target", "called_expression")
                            if node.child_by_field_name(field) is not None), None))
                if fn is None:
                    fn = next((child for child in node.named_children
                               if child.type not in {"arguments", "argument_list", "type_arguments"}), None)
                if fn is not None or (obj is not None and member is not None):
                    target = f"{_text(obj)}.{_text(member)}" if obj is not None and member is not None else _text(fn).strip()
                    target = target.split("(", 1)[0].strip()
                    if target and target not in {"this", "self"}:
                        refs.append(Ref(at(line), EdgeKind.CALLS, target))
                continue
            if language == "sql" and node.type == "relation":
                reference = next((child for child in node.named_children if child.type == "object_reference"), None)
                name = reference.child_by_field_name("name") if reference is not None else None
                if name is not None:
                    refs.append(Ref(at(line), EdgeKind.REFERENCES, _text(name)))
                continue
            if language == "dart" and node.type == "selector" and node.parent is not None:
                is_call = any(child.type == "argument_part" for child in node.named_children)
                siblings = node.parent.named_children
                position = next((i for i, child in enumerate(siblings) if child.id == node.id), -1)
                if is_call and position > 0:
                    refs.append(Ref(at(line), EdgeKind.CALLS, _text(siblings[position - 1])))
                continue
        if node.type == "import_statement":
            source = node.child_by_field_name("source")
            if source is None:
                continue
            module = _ts_module(directory, _unquote(source))
            clause = next((c for c in node.named_children if c.type == "import_clause"), None)
            if clause is None:
                refs.append(Ref(None, EdgeKind.IMPORTS, module))
                continue
            for c in clause.named_children:
                if c.type == "identifier":
                    refs.append(Ref(None, EdgeKind.IMPORTS, f"{module}:default", _text(c)))
                elif c.type == "namespace_import":
                    ident = next((i for i in c.named_children if i.type == "identifier"), None)
                    if ident is not None:
                        refs.append(Ref(None, EdgeKind.IMPORTS, module, _text(ident)))
                elif c.type == "named_imports":
                    for spec in c.named_children:
                        name = spec.child_by_field_name("name")
                        if spec.type != "import_specifier" or name is None:
                            continue
                        alias = spec.child_by_field_name("alias")
                        refs.append(Ref(None, EdgeKind.IMPORTS, f"{module}:{_text(name)}", _text(alias or name)))
        elif node.type in ("call_expression", "new_expression"):
            field = "function" if node.type == "call_expression" else "constructor"
            target = _member_target(node.child_by_field_name(field), "object", "property")
            if target:
                refs.append(Ref(at(line), EdgeKind.CALLS, target))
        elif node.type == "class_heritage":
            for n in _walk(node):
                parent = n.parent.type if n.parent is not None else ""
                if n.type in ("identifier", "type_identifier") and parent in (
                    "class_heritage", "extends_clause", "implements_clause"
                ):
                    refs.append(Ref(at(line), EdgeKind.INHERITS, _text(n)))
    return refs


class RepoGraph(InMemoryCodeGraph):
    """`CodeGraph` over one commit, with each symbol's defining chunk kept for lookups."""

    def __init__(self) -> None:
        super().__init__()
        self.defs: dict[str, Any] = {}
        self._edges: set[tuple[str, str, EdgeKind]] = set()

    def add_edge(self, src: str, dst: str, kind: EdgeKind) -> None:
        if src != dst and (src, dst, kind) not in self._edges:
            self._edges.add((src, dst, kind))
            super().add_edge(src, dst, kind)

    def edges(self) -> set[tuple[str, str, EdgeKind]]:
        return set(self._edges)

    def resolve(self, name: str) -> list[str]:
        """Symbol ids for `path::symbol`, `symbol`, `Class.method` or a bare method name."""
        if name in self.defs:
            return [name]
        return sorted(
            s for s, c in self.defs.items() if c.symbol == name or c.symbol.rsplit(".", 1)[-1] == name
        )


def _python_modules(paths: Iterable[str]) -> dict[str, list[str]]:
    """Every dotted suffix of a module path, so `seraph.index` finds `src/seraph/index.py`."""
    out: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        if not p.endswith(".py"):
            continue
        parts = p[:-3].split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        for i in range(len(parts)):
            out[".".join(parts[i:])].append(p)
    return out


class _Resolver:
    def __init__(self, chunks: Sequence[Any]) -> None:
        self.paths = {c.path for c in chunks}
        self.local: dict[str, dict[str, str]] = defaultdict(dict)
        self.methods: dict[str, list[str]] = defaultdict(list)
        for c in chunks:
            if c.symbol:
                sid = symbol_id(c.path, c.symbol)
                self.local[c.path][c.symbol] = sid
                if "." in c.symbol:
                    self.methods[c.symbol.rsplit(".", 1)[-1]].append(sid)
        self.py_modules = _python_modules(self.paths)
        self.go_dirs: dict[str, list[str]] = defaultdict(list)
        for p in self.paths:
            if p.endswith(".go"):
                self.go_dirs[posixpath.dirname(p)].append(p)
        self.bindings: dict[str, dict[str, tuple[list[str], str | None]]] = defaultdict(dict)
        self.stars: dict[str, list[list[str]]] = defaultdict(list)

    def module_files(self, importer: str, module: str) -> list[str]:
        language = _language(importer)
        if language == "python":
            files = self.py_modules.get(module, [])
            return files if len(files) <= 3 else []
        if language == "go":
            segments = module.split("/")
            for i in range(len(segments)):
                files = self.go_dirs.get("/".join(segments[i:]))
                if files:
                    return sorted(files)
            return []
        if language in {"java", "c_sharp", "rust", "php", "c", "cpp", "kotlin", "scala", "ruby", "dart", "lua"}:
            normalized = module.replace("\\", "/").replace("::", "/")
            if language in {"java", "c_sharp", "kotlin", "scala"}:
                normalized = normalized.replace(".", "/")
            extensions = {
                "java": (".java",), "c_sharp": (".cs",), "rust": (".rs",),
                "php": (".php",), "c": (".h", ".c"), "cpp": (".hpp", ".hh", ".hxx", ".h", ".cpp", ".cc", ".cxx"),
                "kotlin": (".kt", ".kts"), "scala": (".scala", ".sc"),
                "ruby": (".rb", ".rake"), "dart": (".dart",), "lua": (".lua",),
            }[language]
            candidates = ([normalized] if any(normalized.endswith(ext) for ext in extensions)
                          else [normalized + suffix for suffix in extensions])
            matches = [path for path in self.paths for candidate in candidates
                       if path == candidate or path.endswith("/" + candidate)]
            if matches:
                return sorted(set(matches))
        candidates = [module + s for s in _TS_SUFFIXES] + [f"{module}/index{s}" for s in _TS_SUFFIXES]
        return [c for c in candidates if c in self.paths]

    def bind(self, path: str, ref: Ref) -> list[str]:
        """Record what an import binds in `path`; returns the nodes it imports."""
        module, _, name = ref.target.partition(":")
        submodule = []
        if name and name != "*" and _language(path) == "python":
            submodule = self.module_files(path, f"{module}.{name}" if module else name)
        files = submodule or self.module_files(path, module)
        if submodule:
            name = ""
        if not files:
            return []
        if name == "*":
            self.stars[path].append(files)
            return files
        if ref.local:
            self.bindings[path][ref.local] = (files, name or None)
        if name and name != "default":
            return self.lookup(files, name) or files
        return files

    def lookup(self, files: Iterable[str], name: str) -> list[str]:
        return [self.local[f][name] for f in files if name in self.local.get(f, {})]

    def lookup_member(self, files: Iterable[str], name: str) -> list[str]:
        matches = [symbol_id(path, symbol) for path in files for symbol in self.local.get(path, {})
                   if symbol.rsplit(".", 1)[-1] == name]
        return sorted(set(matches))

    def resolve(self, path: str, ref: Ref) -> list[str]:
        target, local = ref.target, self.local.get(path, {})
        if target.startswith(("self.", "this.")):
            cls = ref.symbol.split(".")[0] if ref.symbol else None
            hit = local.get(f"{cls}.{target.split('.')[1]}") if cls else None
            return [hit] if hit else []
        if target.startswith("?."):
            return self._unique_method(target[2:], ref.kind)
        parts = target.split(".")
        bound = self.bindings.get(path, {})
        for i in range(len(parts) - 1, 0, -1):
            binding = bound.get(".".join(parts[:i]))
            if binding:
                files, name = binding
                if name is None:
                    suffix = ".".join(parts[i:])
                    return (self.lookup(files, suffix) or self.lookup(files, parts[i])
                            or self.lookup_member(files, parts[i]))
                # `Class.method()` on an imported class; Python methods live in the class chunk.
                return self.lookup(files, f"{name}.{parts[i]}") or self.lookup(files, name)
        head = parts[0]
        if len(parts) == 1:
            if head in local:
                return [local[head]]
            if _language(path) == "go":
                hits = self.lookup(self.go_dirs.get(posixpath.dirname(path), []), head)
                if hits:
                    return hits
            if head in bound:
                files, name = bound[head]
                return self.lookup(files, head if name in (None, "default") else name)
            for files in self.stars.get(path, []):
                hits = self.lookup(files, head)
                if hits:
                    return hits
            if _language(path) in {"c", "cpp"} and ref.kind == EdgeKind.CALLS:
                included = sorted({file for files, _ in bound.values() for file in files})
                hits = self.lookup(included, target) or self.lookup(included, head)
                if len(hits) == 1:
                    return hits
            if ref.kind == EdgeKind.CALLS:
                hits = self._unique_method(head, ref.kind)
                if len(hits) == 1:
                    return hits
            return []
        qualified = local.get(f"{head}.{parts[1]}") or local.get(head)
        if qualified:
            return [qualified]
        return self._unique_method(parts[-1], ref.kind)

    def _unique_method(self, name: str, kind: str) -> list[str]:
        """A call on an unknown receiver resolves only if exactly one method has that name."""
        ids = self.methods.get(name, [])
        return ids if kind == EdgeKind.CALLS and len(ids) == 1 else []


def build_graph(chunks: Iterable[Any], refs: Iterable[tuple[str, Ref]]) -> RepoGraph:
    """`chunks` are `seraph.index.Chunk`s of one commit; `refs` are `(path, Ref)` for the same commit."""
    chunks = list(chunks)
    graph = RepoGraph()
    for c in chunks:
        if c.symbol is None:
            graph.add_symbol(c.path, c.occurrence_id)
            continue
        sid = symbol_id(c.path, c.symbol)
        graph.add_symbol(sid, c.occurrence_id)
        graph.defs[sid] = c
        graph.add_edge(c.path, sid, EdgeKind.DEFINES)
    resolver = _Resolver(chunks)
    refs = list(refs)
    for path, ref in refs:
        if ref.kind == EdgeKind.IMPORTS:
            for dst in resolver.bind(path, ref):
                graph.add_edge(path, dst, EdgeKind.IMPORTS)
    for path, ref in refs:
        if ref.kind == EdgeKind.IMPORTS:
            continue
        src = symbol_id(path, ref.symbol)
        for dst in resolver.resolve(path, ref):
            graph.add_edge(src, dst, EdgeKind(ref.kind))
    return graph
