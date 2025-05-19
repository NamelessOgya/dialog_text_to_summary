"""
    youtubeからデータをダウンロードして、raw_extracted.csvとして保存
    poetry run python -m run.fetch_and_preprocess
"""

import pandas as pd
import tqdm

from src.data_extractor.gaiyou_formatter import generate_formatted_text
from src.data_extractor.youtube_extractor import extract_info
from src.data_extractor.analyze_youtube_ch import analyze_youtube_ch
from src.data_extractor.split_data import assign_split


CHANNEL_ID = 'UCGkctuF55veBi7xDGCgcYkw'
MINIMUM_VIDEO_TIME = 1 * 60 * 45  # 1時間45分以上の動画を対象とする

SKIP_CHANNEL_ANALYSIS = False
SKIP_DISCRIPTION_TRANSCRIPT_FETCH = False
SKIP_DATA_PREPROCESS = False

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO

if __name__ == "__main__":

    if not SKIP_CHANNEL_ANALYSIS:
        analyze_youtube_ch(CHANNEL_ID, MINIMUM_VIDEO_TIME)

    if not SKIP_DISCRIPTION_TRANSCRIPT_FETCH:
        res = pd.read_csv("./data/source_videos.csv")

        res = res[res["time"] > MINIMUM_VIDEO_TIME]
        dics = res.to_dict(orient='records')
        
        li = []
        for i in tqdm.tqdm(dics):
            try:
                data = extract_info(i["video_url"]) # video_url, transcript, title, description
                i["transcript"] = data["transcript"]
                i["title"] = data["title"]
                i["description"] = data["description"]
                li.append(i)

            except Exception as e:
                print(f"Error: {e}")
                continue
            
        
        df = pd.DataFrame(li)

        df.to_csv("./data/extracted_data.csv", index=False)

    if not SKIP_DATA_PREPROCESS:
        res = pd.read_csv("./data/extracted_data.csv")

        res["target"] = res["description"].apply(lambda x: generate_formatted_text(x))
        res["split"] = res["video_id"].apply(assign_split)

        print(f"target is null...{res['target'].isna().sum()}")

        for i in range(3):
            print(f"data index={i}")
            print(res["target"][i])

        res.to_csv("./data/processed_data.csv", index=False)

    print("done!")