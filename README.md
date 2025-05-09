# regular-expression-using-kagayaki  
　　
## 概要  
youtubeのトランスクリプトを要約に変換するAIを学習するためのrepository

## 実行準備  
### 1. .envファイルの準備  
以下のようにyoutube api名を記載した,`.env`ファイルを作成  
```
API_KEY=xxxx
```
  
  
### 2.環境の取得  
```
./env/vm_hosting_sh/install_nvidia_toolkit.sh
./env/vm_hosting_sh/make_new_sif_env.sh
```  
  
## コンテナでのinteractive実行  
### 1.コンテナの起動  
```
./env/vm_hosting_sh/start_container.sh
```  
  
### 2.コンテナ無いで必要ファイルのinstall  
```
./env/vm_hosting_sh/additional_installs.sh 
source ~/.bashrc
```  
  
### 3.実行  
youtubeの動画を取得する。  
```
poetry run python -m run.fetch_and_preprocess
```  
  
モデルの学習を実施  
```
poetry run python -m run.finetune.py
```

## バッチ実行  
```
./env/vm_hosting_sh/batch_excute.sh
```  
  
ログの確認は以下  
```
docker logs dialog_text_to_summary
```  
  
停止  
```
docker stop dialog_text_to_summary
```