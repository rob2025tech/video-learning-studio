"""Architecture boundaries, enforced on the import and attribute forms in ``src/``.

The layering this project relies on is a set of *negative* facts — the CLI does not import
the API, the core does not import an HTTP framework — and negative facts rot silently: a
single convenient import in a service module would break the optional-extra promise
without failing any behavioral test. This module turns each one into an assertion.

Rules enforced (all verified against the current tree, none aspirational):

* ``cli`` must not import ``api``, and must not import the HTTP stack.
* ``core`` must not import ``api`` or any HTTP-stack module, and stays pure: stdlib plus
  ``video_learning.core`` only.
* ``services`` must not import ``api``, ``cli`` or any HTTP-stack module.
* ``adapters`` and ``providers`` must not import ``api``, ``cli`` or any HTTP-stack module.
* ``api`` may import shared ``services`` and ``core`` — asserted positively, so a refactor
  that quietly re-implements search inside the API also fails.
* ``api`` must not import or reach the CLI, and must not import *or address* the report
  pipeline (``ReportService`` and the analyze/transcribe/segment/inspect services). Reaching
  a pipeline module through a bound name or a dotted attribute chain counts, because an
  import-name check alone does not: see :func:`_pipeline_reach`.
* ``api`` may bind only the two ``services`` seams, ``report_service`` and ``search_service``.
* ``subprocess`` appears only in ``adapters``.
* Every scanned file classifies into a layer, so no file escapes the rules by sitting
  somewhere the scan did not anticipate.

What static analysis here can and cannot prove. It reads each file's own text, so it sees an
import in any scope (including a lazy one inside a function), in aliased, relative,
submodule and ``import_module`` form, and it resolves a dotted attribute chain through the
names those imports bound. It does **not** see a call graph: a module object passed to a
helper in another file, or a name rebound to an instance of a service class, is beyond it.
So what this file guarantees is "no API module names or reaches pipeline code", which is the
claim a reviewer checks by eye. The companion runtime guarantee — that no route *executes*
media work, asserted with every ``subprocess`` entry point patched to raise — is in
``test_api_no_media_processing.py``, and "the CLI runs with FastAPI genuinely absent" is
proven by ``tests/integration/test_cli_without_api_extra.py``, not inferred from here.

Deliberate exceptions, so a future reader does not "fix" them:

* ``services`` may import ``adapters`` (``report_service`` needs the ``WhisperCpp`` type).
  That direction is allowed; only ``cli``/``api``/HTTP are banned below the API.
* ``api/app.py`` imports ``uvicorn`` inside ``main()`` and ``starlette.concurrency`` in a
  route body. The scan records imports at any scope, because a lazy import is still a
  dependency of that layer.
* ``anyio`` is not in :data:`HTTP_MODULES`: it is a general async runtime, not an HTTP
  framework. ``httpx`` is, and is banned outside ``api``.
* Only ``src/`` is scanned for the layering rules, and ``tests/conftest.py`` is scanned for
  one rule of its own. Test modules legitimately import FastAPI and the CLI together; the
  conftest is the exception because the CLI-only integration run imports it without the
  ``api`` extra installed.

The extractor resolves relative imports and records ``from X import Y`` as both ``X`` and
``X.Y``, so aliased, submodule, from-import and dynamic forms are all caught. Attribute
chains resolve through the bindings an import introduced, so
``import video_learning.services as s`` followed by ``s.analyze_service.build(...)`` becomes
``video_learning.services.analyze_service.build``, not an unrelated name. Neither form occurs
in the tree today; :func:`test_import_forms_are_detected` and
:func:`test_pipeline_reach_forms_are_detected` pin both halves against synthetic snippets, so
the guarantee is tested rather than assumed.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src" / "video_learning"
PACKAGE = "video_learning"
API_PACKAGE = f"{PACKAGE}.api"
CLI_PACKAGE = f"{PACKAGE}.cli"
SERVICES_PACKAGE = f"{PACKAGE}.services"

#: The six packages under ``src/video_learning``. A file that classifies as
#: ``(unclassified)`` fails :func:`test_every_scanned_file_is_classified`, so a new
#: directory cannot quietly escape the rules, and the two modules at the package root are
#: named here rather than treated as an unknown layer.
LAYERS = ("api", "adapters", "cli", "core", "providers", "services")
ROOT_FILES = frozenset({"__init__.py", "__main__.py"})

#: Importable module names of the HTTP/API stack. ``python_multipart`` is what the
#: ``python-multipart`` distribution installs; ``pydantic_core`` is pydantic's other half.
HTTP_MODULES = frozenset(
    {
        "fastapi",
        "starlette",
        "pydantic",
        "pydantic_core",
        "uvicorn",
        "multipart",
        "python_multipart",
        "httpx",
    }
)

#: The pipeline orchestrators: what would put ffmpeg/tesseract/whisper behind an HTTP
#: request. ``ReportStore`` (same module as ``ReportService``) is the reuse seam and allowed,
#: so the class is listed apart from its module rather than by naming the module banned.
PIPELINE_MODULES = frozenset(
    {
        f"{SERVICES_PACKAGE}.analyze_service",
        f"{SERVICES_PACKAGE}.transcribe_service",
        f"{SERVICES_PACKAGE}.segment_service",
        f"{SERVICES_PACKAGE}.inspect_service",
    }
)
PIPELINE_SYMBOLS = frozenset({f"{SERVICES_PACKAGE}.report_service.ReportService"})
PIPELINE_IMPORTS = PIPELINE_MODULES | PIPELINE_SYMBOLS

#: The only two things ``api`` may bind out of ``services``: the artifact reader and the
#: retrieval entry point. Reaching any other name in that package is running the pipeline.
API_SERVICE_SEAMS = frozenset(
    {f"{SERVICES_PACKAGE}.report_service", f"{SERVICES_PACKAGE}.search_service"}
)


def _absolute_from_import(node: ast.ImportFrom, package: str) -> str:
    """Resolve ``from ... import`` to an absolute dotted module name."""
    if not node.level:
        return node.module or ""
    parts = package.split(".")
    if node.level > len(parts):
        return node.module or ""
    # level 1 is the containing package, level 2 its parent, and so on.
    base = ".".join(parts[: len(parts) - node.level + 1])
    return f"{base}.{node.module}" if node.module else base


def _dynamic_import_name(node: ast.Call) -> str | None:
    """The literal module name in ``import_module("x")`` / ``__import__("x")`` / ``find_spec``."""
    func = node.func
    if isinstance(func, ast.Attribute):
        name = func.attr
    elif isinstance(func, ast.Name):
        name = func.id
    else:
        return None
    if name not in {"import_module", "__import__", "find_spec"}:
        return None
    if node.args and isinstance(node.args[0], ast.Constant):
        value = node.args[0].value
        if isinstance(value, str):
            return value
    return None


def imported_modules(source: str, package: str) -> set[str]:
    """Every module ``source`` imports, as absolute dotted names.

    ``package`` is the package the scanned file lives in (``video_learning.cli`` for
    ``cli/main.py``), which is what relative imports resolve against. Aliases do not
    matter: the *module* is recorded, never the name it is bound to.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_from_import(node, package)
            if base:
                found.add(base)
                found.update(
                    f"{base}.{alias.name}" for alias in node.names if alias.name != "*"
                )
        elif isinstance(node, ast.Call):
            dynamic = _dynamic_import_name(node)
            if dynamic:
                found.add(dynamic)
    return found


def import_bindings(source: str, package: str) -> dict[str, str]:
    """Map every name an import introduces to what it stands for, as a dotted path.

    ``import a.b.c`` binds only the head ``a``, which therefore means ``a``;
    ``import a.b.c as m`` binds ``m`` to ``a.b.c``, and ``from a import b`` binds ``b`` to
    ``a.b`` whatever ``b`` is — submodule or symbol. That ambiguity is why the rules below
    compare the *bound path* against the names that are allowed: a symbol inside an allowed
    module (``ReportStore``) stays clean while the banned one (``ReportService``) does not.
    """
    bound: dict[str, str] = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_from_import(node, package)
            for alias in node.names:
                if alias.name != "*":
                    target = f"{base}.{alias.name}"
                    bound[alias.asname or alias.name] = target if base else alias.name
        elif isinstance(node, ast.Call):
            dynamic = _dynamic_import_name(node)
            if dynamic:
                bound[dynamic] = dynamic
    return bound


def _dotted_chain(node: ast.Attribute) -> str | None:
    """``a.b.c`` as a dotted string when the root is a plain name, else ``None``."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return ".".join(reversed(parts))


def accessed_paths(source: str, package: str) -> set[str]:
    """Every dotted path ``source`` addresses, resolved through the names it imported.

    ``s.analyze_service.build`` where ``s`` came from ``import video_learning.services as s``
    becomes ``video_learning.services.analyze_service.build``. A chain rooted at a name no
    import bound — a local, a parameter, a built-in — is skipped rather than guessed at, and
    a bare name an import did bind is recorded as that binding, so passing ``SearchService``
    around as a value still names what it is.
    """
    bound = import_bindings(source, package)
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            chain = _dotted_chain(node)
            if chain is None:
                continue
            head, _, rest = chain.partition(".")
            target = bound.get(head)
            if target is None:
                continue
            found.add(f"{target}.{rest}" if rest else target)
        elif isinstance(node, ast.Name) and node.id in bound:
            found.add(bound[node.id])
    return found


def _pipeline_reach(bindings: Mapping[str, str], paths: Iterable[str]) -> set[str]:
    """Describe every bound name and addressed chain that can get hold of pipeline code.

    Empty is the pass condition, and the return value is the offending descriptions rather
    than a flag, because a rule that only says "no" hides which form tripped it. A binding
    to the bare ``video_learning.services`` package counts even though it names nothing
    banned: bound whole, that namespace reaches every module in it without a later form the
    file has to write down.
    """
    banned = PIPELINE_MODULES | PIPELINE_SYMBOLS
    hits: set[str] = set()
    for name, target in bindings.items():
        if target in banned:
            hits.add(f"binding {name} -> {target}")
        elif target == SERVICES_PACKAGE:
            hits.add(f"namespace binding {name} -> {target}")
    for path in paths:
        if any(path == module or path.startswith(f"{module}.") for module in banned):
            hits.add(f"attribute chain {path}")
    return hits


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC_ROOT.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_of(path: Path) -> str:
    module = _module_name(path)
    return module if path.name == "__init__.py" else module.rsplit(".", 1)[0]


def _layer_of(relative: Path) -> str:
    """Which layer a scanned file belongs to, so none can be silently unruled."""
    if len(relative.parts) == 1:
        return "(root)" if relative.name in ROOT_FILES else "(unclassified)"
    return relative.parts[0] if relative.parts[0] in LAYERS else "(unclassified)"


def _scan() -> dict[str, tuple[str, set[str], dict[str, str], set[str]]]:
    scanned: dict[str, tuple[str, set[str], dict[str, str], set[str]]] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        relative = path.relative_to(SRC_ROOT)
        source = path.read_text(encoding="utf-8")
        package = _package_of(path)
        scanned[str(relative)] = (
            _layer_of(relative),
            imported_modules(source, package),
            import_bindings(source, package),
            accessed_paths(source, package),
        )
    return scanned


MODULES = _scan()


def _under(imports: Iterable[str], root: str) -> set[str]:
    return {name for name in imports if name == root or name.startswith(f"{root}.")}


def _in_layer(layer: str) -> list[str]:
    return [name for name, found in MODULES.items() if found[0] == layer]


def _http(imports: Iterable[str]) -> set[str]:
    return {name for name in imports if name.split(".")[0] in HTTP_MODULES}


def test_the_scan_actually_sees_imports() -> None:
    """A broken extractor would make every negative rule below pass vacuously."""
    assert MODULES, "no source files were scanned"
    assert len(MODULES) == len(list(SRC_ROOT.rglob("*.py")))
    assert "fastapi" in MODULES["api/app.py"][1]
    assert "typer" in MODULES["cli/main.py"][1]
    assert f"{PACKAGE}.core.study" in MODULES["services/report_service.py"][1]
    assert f"{PACKAGE}.services.search_service" in MODULES["api/reports.py"][1]
    # The binding and chain halves have to see real code too, or the next two rules are
    # vacuous in exactly the way the import-name rules used to be.
    assert MODULES["api/reports.py"][2]["SearchService"] == (
        f"{SERVICES_PACKAGE}.search_service.SearchService"
    )
    assert f"{SERVICES_PACKAGE}.search_service.SearchService" in MODULES["api/reports.py"][3]


@pytest.mark.parametrize("name", _in_layer("cli"))
def test_cli_does_not_import_api(name: str) -> None:
    assert _under(MODULES[name][1], API_PACKAGE) == set()


def test_cli_does_not_import_the_http_stack() -> None:
    """``typer`` and ``rich`` are the CLI's own two; the HTTP stack would end the extra promise."""
    offenders = [name for name in _in_layer("cli") if _http(MODULES[name][1])]

    assert offenders == []


@pytest.mark.parametrize("name", _in_layer("core"))
def test_core_imports_neither_api_nor_http(name: str) -> None:
    imports = MODULES[name][1]
    assert _under(imports, API_PACKAGE) == set()
    assert _http(imports) == set()


@pytest.mark.parametrize("name", _in_layer("core"))
def test_core_stays_pure(name: str) -> None:
    """stdlib plus ``video_learning.core`` only — no services, adapters, providers or cli."""
    for module in MODULES[name][1]:
        root = module.split(".")[0]
        if root == PACKAGE:
            assert module == f"{PACKAGE}.core" or module.startswith(f"{PACKAGE}.core."), module
        else:
            assert root in sys.stdlib_module_names, module


@pytest.mark.parametrize("layer", ["services", "adapters", "providers"])
def test_lower_layers_import_neither_api_nor_cli_nor_http(layer: str) -> None:
    for name in _in_layer(layer):
        imports = MODULES[name][1]
        assert _under(imports, API_PACKAGE) == set(), name
        assert _under(imports, CLI_PACKAGE) == set(), name
        assert _http(imports) == set(), name


def test_api_reuses_the_shared_services_and_core() -> None:
    """The positive half: the API must delegate, not re-implement search or validation."""
    combined: set[str] = set()
    for name in _in_layer("api"):
        combined |= MODULES[name][1]

    assert f"{SERVICES_PACKAGE}.search_service" in combined
    assert f"{SERVICES_PACKAGE}.report_service" in combined
    assert f"{PACKAGE}.core.search" in combined


def test_api_reaches_services_only_through_the_two_seams() -> None:
    """Whatever ``api`` binds out of ``services`` must be the reader or the search service.

    ``import video_learning.services`` whole binds a namespace that can reach the pipeline
    without ever naming a banned module, so it is refused here instead of being trusted.
    """
    offenders: list[str] = []
    for name in _in_layer("api"):
        for bound, target in MODULES[name][2].items():
            if not _under([target], SERVICES_PACKAGE):
                continue
            seam = target in API_SERVICE_SEAMS or target.startswith(
                tuple(f"{seam}." for seam in API_SERVICE_SEAMS)
            )
            if not seam:
                offenders.append(f"{name}: {bound} -> {target}")

    assert offenders == []


@pytest.mark.parametrize("name", _in_layer("api"))
def test_api_does_not_import_or_invoke_the_cli(name: str) -> None:
    module = MODULES[name]
    imports, paths = module[1], module[3]
    assert _under(imports, CLI_PACKAGE) == set()
    assert "typer" not in imports
    assert "typer.testing" not in imports
    assert "subprocess" not in imports
    assert not any(path.startswith(CLI_PACKAGE) for path in paths), (name, sorted(paths))


@pytest.mark.parametrize("name", _in_layer("api"))
def test_api_does_not_import_the_report_pipeline(name: str) -> None:
    assert MODULES[name][1] & PIPELINE_IMPORTS == set()


@pytest.mark.parametrize("name", _in_layer("api"))
def test_api_does_not_reach_the_pipeline_by_name_or_attribute(name: str) -> None:
    """Module-qualified access is the form an import-name comparison cannot see.

    ``import video_learning.services as s`` then ``s.analyze_service.build(...)``, or
    ``from ... import analyze_service`` then ``analyze_service.detect(...)``, never mentions
    a banned import name in the form the rule above compares; both resolve to a dotted path
    here. A name rebound to an *instance* of a service class stays invisible, which is why a
    runtime test, not this one, owns the "no route executes media work" claim.
    """
    module = MODULES[name]

    assert _pipeline_reach(module[2], module[3]) == set(), (name, sorted(module[3]))


def test_every_scanned_file_is_classified() -> None:
    """A file nobody named a layer for would be scanned and then ruled by nothing."""
    unclassified = sorted(name for name, found in MODULES.items() if found[0] == "(unclassified)")

    assert unclassified == []
    assert _in_layer("(root)") == sorted(ROOT_FILES)


@pytest.mark.parametrize("layer", LAYERS)
def test_every_layer_has_at_least_one_file(layer: str) -> None:
    """A layer in the list that matches nothing is a directory the scan no longer reaches."""
    assert _in_layer(layer), layer


def test_package_roots_import_nothing_banned() -> None:
    """``__init__.py`` and ``__main__.py`` are entry points, not a place to hide a dependency."""
    for name in _in_layer("(root)"):
        module = MODULES[name]
        assert _under(module[1], API_PACKAGE) == set(), name
        assert _http(module[1]) == set(), name
        assert module[1] & PIPELINE_IMPORTS == set(), name
        assert _pipeline_reach(module[2], module[3]) == set(), name


def test_the_shared_conftest_stays_importable_without_the_api_extra() -> None:
    """``tests/conftest.py`` is imported by the CLI-only run, so it may not need FastAPI.

    That run is what makes the optional-extra promise real; this rule keeps the shared
    fixture honest about it rather than leaving the claim to the integration test alone.
    """
    source = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    imports = imported_modules(source, "tests")

    assert _http(imports) == set()
    assert _under(imports, API_PACKAGE) == set()


def test_subprocess_appears_only_in_adapters() -> None:
    offenders = [
        name
        for name, module in MODULES.items()
        if "subprocess" in module[1] and module[0] != "adapters"
    ]
    assert offenders == []
    assert any("subprocess" in MODULES[name][1] for name in _in_layer("adapters"))


@pytest.mark.parametrize(
    ("source", "package", "expected"),
    [
        ("import fastapi", f"{PACKAGE}.core", "fastapi"),
        ("import fastapi as f", f"{PACKAGE}.core", "fastapi"),
        ("from fastapi import FastAPI as F", f"{PACKAGE}.core", "fastapi"),
        ("import starlette.responses", f"{PACKAGE}.services", "starlette.responses"),
        ("import video_learning.api", f"{PACKAGE}.cli", API_PACKAGE),
        ("import video_learning.api.app as m", f"{PACKAGE}.cli", f"{API_PACKAGE}.app"),
        ("from video_learning import api", f"{PACKAGE}.cli", API_PACKAGE),
        ("from video_learning import api as a", f"{PACKAGE}.cli", API_PACKAGE),
        ("from video_learning.api import create_app", f"{PACKAGE}.cli", API_PACKAGE),
        ("from .. import api", f"{PACKAGE}.cli", API_PACKAGE),
        ("from ..api.app import create_app", f"{PACKAGE}.cli", f"{API_PACKAGE}.app"),
        ("from .sessions import ReportSession", f"{PACKAGE}.api", f"{PACKAGE}.api.sessions"),
        ("importlib.import_module('fastapi')", f"{PACKAGE}.core", "fastapi"),
        ('__import__("starlette")', f"{PACKAGE}.core", "starlette"),
        ('importlib.util.find_spec("uvicorn")', f"{PACKAGE}.services", "uvicorn"),
        (
            "from video_learning.services.report_service import ReportService",
            f"{PACKAGE}.api",
            f"{SERVICES_PACKAGE}.report_service.ReportService",
        ),
    ],
)
def test_import_forms_are_detected(source: str, package: str, expected: str) -> None:
    """Direct, aliased, submodule, from-import, relative and dynamic forms all register.

    Also checks the enforcement helpers see them: a dotted ``video_learning.api.app`` is
    recorded in full, and it is ``_under`` that turns it into a banned-prefix match.
    """
    found = imported_modules(source, package)

    assert expected in found
    if expected.startswith(API_PACKAGE):
        assert _under(found, API_PACKAGE)
    if expected.split(".")[0] in HTTP_MODULES:
        assert _http(found)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # Module-qualified pipeline access: the import names nothing banned, the use does.
        (
            "from video_learning.services import analyze_service\nanalyze_service.detect(x)",
            f"{SERVICES_PACKAGE}.analyze_service",
        ),
        (
            "import video_learning.services as svc\nsvc.analyze_service.build(x)",
            f"{SERVICES_PACKAGE}.analyze_service",
        ),
        (
            "import video_learning.services\n"
            "video_learning.services.transcribe_service.transcribe(p)",
            f"{SERVICES_PACKAGE}.transcribe_service",
        ),
        # A symbol reached through an allowed module's alias is the other half of the loophole.
        (
            "import video_learning.services.report_service as rs\n"
            "service = rs.ReportService()\nservice.build_video_report(p)",
            f"{SERVICES_PACKAGE}.report_service.ReportService",
        ),
        # Bound whole, the services namespace reaches the pipeline without naming it.
        (f"from {PACKAGE} import services\nservices(x)", SERVICES_PACKAGE),
        # The two seam forms and the version line must stay clean, or the rule is over-broad.
        (
            "from video_learning.services.report_service import ReportStore\n"
            "ReportStore().load_evidence(p)",
            None,
        ),
        (
            "from video_learning.services.search_service import SearchService\n"
            "SearchService(e).query('q', limit=5)",
            None,
        ),
        (f"from {PACKAGE} import __version__\nprint(__version__)", None),
    ],
)
def test_pipeline_reach_forms_are_detected(source: str, expected: str | None) -> None:
    """Banned forms trip the rule; the artifact and retrieval seams do not.

    ``expected`` is the dotted path that must appear in an offender description, or ``None``
    for a form that must be clean. In the alias case the binding is what trips it: the file
    addresses ``rs.ReportService`` through a module the seam rule allows, and a later rebinding
    of ``service`` is beyond what any of this can see.
    """
    package = f"{PACKAGE}.api"
    hits = _pipeline_reach(import_bindings(source, package), accessed_paths(source, package))

    if expected is None:
        assert hits == set(), (source, hits)
    else:
        assert any(expected in hit for hit in hits), (source, hits)


def test_a_relative_sibling_import_is_not_mistaken_for_the_api() -> None:
    """``from . import search`` inside core is core.search, not video_learning.api."""
    found = imported_modules("from . import search", f"{PACKAGE}.core")

    assert f"{PACKAGE}.core.search" in found
    assert _under(found, API_PACKAGE) == set()


def test_the_pipeline_guard_distinguishes_store_from_service() -> None:
    """``ReportStore`` is the allowed seam; only ``ReportService`` trips the rule."""
    form = "from video_learning.services.report_service import ReportStore"
    allowed = imported_modules(form, package=API_PACKAGE)
    banned = imported_modules(
        "from video_learning.services.report_service import ReportService", API_PACKAGE
    )

    assert allowed & PIPELINE_IMPORTS == set()
    assert banned & PIPELINE_IMPORTS
    assert _pipeline_reach(
        import_bindings(form, API_PACKAGE), accessed_paths(form, API_PACKAGE)
    ) == set()
    service_form = "from video_learning.services.report_service import ReportService"
    assert _pipeline_reach(
        import_bindings(service_form, API_PACKAGE), accessed_paths(service_form, API_PACKAGE)
    ) == {f"binding ReportService -> {SERVICES_PACKAGE}.report_service.ReportService"}
