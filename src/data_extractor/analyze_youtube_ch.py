"""
YouTube動画の概要欄とトランスクリプトを取得するスクリプト
poetry run python -m src.data_extractor.analyze_youtube_ch
"""


from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import isodate
from typing import List, Dict
import pandas as pd
import tqdm

from src.data_extractor.gaiyou_formatter import extract_timelines


MINIMUM_VIDEO_TIME = 1 * 60 * 45  # 1時間45分以上の動画を対象とする

# .env がプロジェクトルートにある想定
load_dotenv()

API_KEY = os.getenv("API_KEY")



def get_format_matched_video_urls(api_key: str, channel_id: str) -> list[Dict[str, str]]:
    """
    指定したチャンネルの全動画 URL を取得してリストで返します。
    """
    youtube = build('youtube', 'v3', developerKey=api_key)

    # 1) チャンネル情報取得 → アップロード動画用プレイリスト ID を得る
    ch_response = youtube.channels().list(
        part='contentDetails',
        id=channel_id
    ).execute()
    uploads_pl_id = ch_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    # 2) プレイリスト項目をページごとに取得
    video_urls = []
    next_page_token = None

    while True:
        
        pl_response = youtube.playlistItems().list(
            part='snippet',
            playlistId=uploads_pl_id,
            maxResults=50,            # 1リクエストあたり最大50件
            pageToken=next_page_token
        ).execute()

        for item in pl_response['items']:
            if len(extract_timelines(item["snippet"]["description"])) > 0:
                video_id = item['snippet']['resourceId']['videoId']
                video_urls.append(
                    {
                        "video_id": video_id,
                        "video_url": f'https://www.youtube.com/watch?v={video_id}'
                    }
                )

        next_page_token = pl_response.get('nextPageToken')
        print(next_page_token)
        if not next_page_token:
            break

    return video_urls

def add_time_to_video_urls_df(video_urls_df: pd.DataFrame) -> pd.DataFrame:
    """
    動画 URL dfに動画の長さを追加したdfを返します。
    """
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    print(len(df))
    candidate_ids = df["video_id"].values

    long_videos = []
    for i in tqdm.tqdm(range(0, len(candidate_ids), 50)):
        batch = candidate_ids[i:i+50]
        vd = youtube.videos().list(
            part='contentDetails',
            id=','.join(batch)
        ).execute()

        for item in vd['items']:
            vid = item['id']
            dur = isodate.parse_duration(item['contentDetails']['duration']).total_seconds()
            
            long_videos.append(
                {
                    "video_id": vid,
                    "video_url": f'https://www.youtube.com/watch?v={vid}',
                    "time": int(dur)
                }
            )
    
    long_videos_df = pd.DataFrame(long_videos)

    return long_videos_df

def analyze_youtube_ch(channel_id):
    res = get_format_matched_video_urls(API_KEY, channel_id)

    df = pd.DataFrame(res)
    print(f"sample num extracted...{len(df)}")
    df.to_csv("./data/tmp_video_urls.csv", index=False)

    # 動画の長さを取得
    df = add_time_to_video_urls_df(df)
    df.to_csv("./data/tmp_videos_urls_with_time.csv", index=False)

    df = df[df["time"] > MINIMUM_VIDEO_TIME]
    print(f"sample num under time condition...{len(df)}")
    df.to_csv("./data/source_videos.csv", index=False)

if __name__ == '__main__':
    CHANNEL_ID = 'UCGkctuF55veBi7xDGCgcYkw'
    analyze_youtube_ch(CHANNEL_ID)
    
    