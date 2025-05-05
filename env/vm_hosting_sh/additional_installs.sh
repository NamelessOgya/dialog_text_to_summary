apt update
apt install -y curl

curl -sSL https://install.python-poetry.org | python3 -


# パスを通す
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
# 追記後、ファイルを読み込み直す
source ~/.bashrc


poetry install --no-root