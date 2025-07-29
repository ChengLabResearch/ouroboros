FROM thehale/python-poetry:2.1.3-py3.11-slim AS builder

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

RUN apt-get update -y
RUN apt-get install gcc -y

WORKDIR /app

COPY poetry.lock pyproject.toml ./
RUN touch README.md

RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY ./ouroboros ./ouroboros

ENTRYPOINT ["python", "-m", "ouroboros.docker_server"]