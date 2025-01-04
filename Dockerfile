FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    apt-get install -y --no-install-recommends build-essential gcc supervisor nginx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY server_config/supervisord.conf /supervisord.conf
COPY server_config/nginx /etc/nginx/sites-available/default
COPY server_config/docker-entrypoint.sh /entrypoint.sh

ADD . /app

WORKDIR /app

RUN uv sync --frozen

EXPOSE 9000 9001
ENTRYPOINT ["sh", "/entrypoint.sh"]