apt update
apt install -y curl

curl -sSL https://install.python-poetry.org | python3 -


# パスを通す
ENV PATH="/root/.local/bin:${PATH}"