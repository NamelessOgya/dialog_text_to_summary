#!/bin/bash
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
  pytorch/pytorch \
  bash -c "./env/vm_hosting_sh/additional_installs.sh && source ~/.bashrc && poetry run python run -m ./run/finetune.py" #実行コマンドに応じて変更

