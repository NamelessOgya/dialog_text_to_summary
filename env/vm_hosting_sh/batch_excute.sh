#!/usr/bin/env bash
set -e

# もし同名コンテナが残っていれば消す
docker rm -f dialog_text_to_summary 2>/dev/null || true

# バックグラウンドで立ち上げ。tail で常駐させる
docker run -d \
  --gpus all \
  --name dialog_text_to_summary \
  -v "$(pwd):/app" \
  -w "/app" \
  --restart unless-stopped \
  --entrypoint /bin/bash \
  my-pytorch-poetry \
  bash -l -c "\
  poetry install --no-root && \
  poetry run python ./run/finetune.py \
"
