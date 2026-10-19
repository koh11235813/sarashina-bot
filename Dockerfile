FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project --no-editable

COPY pyproject.toml uv.lock .python-version ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM python:3.12-slim

# kubelet の runAsNonRoot は数値 UID でないと検証できないので、名前ではなく UID で指定する
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app

# venv 内の shebang と pyvenv.cfg は絶対パスなので、builder と同じ /app/.venv に置く
COPY --from=builder /app/.venv /app/.venv

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER 10001
CMD ["python", "-m", "sarashina_bot"]
