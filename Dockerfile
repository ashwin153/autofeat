FROM python:3.13-bookworm AS build
WORKDIR /app
RUN apt-get update && apt-get install -y cargo=0.66.0+ds1-1
RUN pip3 install poetry==1.8.3
COPY pyproject.toml poetry.lock README.md ./
RUN POETRY_VIRTUALENVS_IN_PROJECT=1 poetry install --without dev --no-root --no-cache

FROM python:3.13-slim-bookworm
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
COPY --from=build /app/.venv /app/.venv
COPY autofeat ./autofeat
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "autofeat/ui/__main__.py"]
