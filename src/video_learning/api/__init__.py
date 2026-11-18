"""Optional local HTTP API over the existing evidence-search services.

Importing this package needs the API stack — ``fastapi``, ``uvicorn``,
``python-multipart`` and ``pydantic``. ``uv pip install '.[api]'`` selects them; so does a
project ``uv sync``, because the dev group repeats these four for the tests — while
``uv sync --no-dev`` installs the CLI alone. Nothing in ``cli/``, ``core/``, ``services/``,
``adapters/`` or ``providers/`` imports this package, and ``video_learning/__init__.py`` stays
a single version line, so the CLI runs unchanged in an environment where FastAPI is not
installed at all.
"""
