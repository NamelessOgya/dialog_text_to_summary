# syntax=docker/dockerfile:1.5
FROM pytorch/pytorch:latest

ARG  POETRY_VERSION=2.0.1
ENV  POETRY_HOME="/opt/poetry" \
     POETRY_VIRTUALENVS_CREATE="false" \
     POETRY_NO_INTERACTION="1" \
     PATH="$POETRY_HOME/bin:${PATH}"

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 # ---- Poetry インストール ----
 && curl -sSL https://install.python-poetry.org | POETRY_VERSION=${POETRY_VERSION} python3 - \
 && ln -s $POETRY_HOME/bin/poetry /usr/local/bin/poetry \
 && poetry --version        # ← ここでバージョンが表示されれば OK

WORKDIR /app
CMD ["bash"]                # 元イメージと同じ振る舞いを継承
