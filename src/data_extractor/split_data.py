
import pandas as pd
import hashlib

# サンプル DataFrame
# res = pd.DataFrame({"video_id": [...], ...})

def assign_split(video_id: str) -> str:
    # MD5 ハッシュを取得し、16 進数文字列→整数に変換
    h = int(hashlib.md5(video_id.encode("utf-8")).hexdigest(), 16)
    # 0.0～1.0 未満の浮動小数点に正規化
    p = (h % 10_000) / 10_000
    # 確率に応じてラベルを返す
    if p < 0.8:
        return "train"
    elif p < 0.9:
        return "test"
    else:
        return "valid"
