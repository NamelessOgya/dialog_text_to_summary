"""
YouTube動画の概要欄とトランスクリプトを取得するスクリプト
poetry run python -m src.data_extractor.youtube_extractor
"""


from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import re
from apiclient.discovery import build
from dotenv import load_dotenv
import os

# .env がプロジェクトルートにある想定
load_dotenv()

API_KEY = os.getenv("API_KEY")

def extract_video_id(url: str) -> str:
    # URLから動画IDを抽出
    # channel名は不要のため除外。 
    url = url.split("&ab_channel=")[0]
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not match:
        raise ValueError("無効なYouTube URLです")
    
    return match.group(1)

def get_youtube_title_and_description(video_id: str) -> str:
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    videos_response = youtube.videos().list(
        part='snippet,statistics',
        id='{},'.format(video_id)
    ).execute()
    # snippet
    snippetInfo = videos_response["items"][0]["snippet"]
    # 動画タイトル
    title = snippetInfo['title']
    # チャンネル名
    channeltitle = snippetInfo['channelTitle']

    # description
    description = snippetInfo['description']

    return {
        "title": title,
        "description": description
    }


def get_youtube_transcript(video_id: str, lang: str = "ja") -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
        return "\n".join([entry['text'] for entry in transcript])
    except (TranscriptsDisabled, NoTranscriptFound):
        return "※ トランスクリプトは利用できません。"

def extract_info(url: str):
    video_id = extract_video_id(url)
    # description = get_youtube_description(url)
    transcript = get_youtube_transcript(video_id)
    info = get_youtube_title_and_description(video_id)

    info["transcript"] = transcript
    info["video_url"] = url

    return info # "url", transcript, title, description


# ==== 使用例 ====
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=HVEV0tC9sB8"  # ← 適宜書き換えてください
    info = extract_info(youtube_url)

    print(f"========== title ==========")
    print(info['title'])
    print(f"========== gaiyou ==========")
    print(info['description'])
    print(f"========== transcript ==========")
    print(info['transcript'])
