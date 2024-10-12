FROM python:3.11-buster AS build
WORKDIR /app
RUN pip3 install poetry==1.8.3
COPY pyproject.toml poetry.lock README.md ./
RUN POETRY_VIRTUALENVS_IN_PROJECT=1 poetry install --without dev --no-root --no-cache

FROM python:3.11-slim-buster
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
COPY --from=build /app/.venv /app/.venv
COPY autofeat ./autofeat
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "autofeat/ui/__main__.py"]
