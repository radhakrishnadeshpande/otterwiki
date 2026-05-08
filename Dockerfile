FROM redimp/otterwiki:2

RUN mkdir -p /app-data

COPY app-data/app-data/*.md /app-data/