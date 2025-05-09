# 1. ベースイメージ
FROM pytorch/pytorch:latest

# 2. apt の更新＆必要パッケージのインストール
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
 && rm -rf /var/lib/apt/lists/*

# 3. Poetry インストール
RUN curl -sSL https://install.python-poetry.org | python3 - \
 && mv ~/.local/bin/poetry /usr/local/bin/poetry

# 4. 作業ディレクトリ設定
WORKDIR /app

# （必要ならソースコードをコピーして、poetry install もここで実行できます）
# COPY pyproject.toml poetry.lock ./
# RUN poetry install --no-dev

# デフォルトのコマンド／エントリーポイントを指定
ENTRYPOINT ["bash"]
