"""
YouTube動画の概要欄とトランスクリプトを取得するスクリプト
poetry run python -m src.data_extractor.youtube_extractor
"""


from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import re

def extract_video_id(url: str) -> str:
    # URLから動画IDを抽出
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    if not match:
        raise ValueError("無効なYouTube URLです")
    print(match.group(1))
    return match.group(1)

def get_youtube_description(url: str) -> str:
    print(f"pytube url {url}")
    yt = YouTube(url)
    return yt.description

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

    print("▼ 概要欄 (Description)")
    # print(description)
    print("\n▼ トランスクリプト (Transcript)")
    print(transcript)

# ==== 使用例 ====
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=HVEV0tC9sB8"  # ← 適宜書き換えてください
    extract_info(youtube_url)
