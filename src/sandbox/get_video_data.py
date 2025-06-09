"""
    poetry run python -m src.sandbox.get_video_data
"""

from apiclient.discovery import build
from dotenv import load_dotenv
import os

# .env がプロジェクトルートにある想定
load_dotenv()

API_KEY = os.getenv("API_KEY")


videoId = 'hQtmJY84dNY'
YOUTUBE_API_KEY = 'ここにYouTube APIを記載'

youtube = build('youtube', 'v3', developerKey=API_KEY)
videos_response = youtube.videos().list(
    part='snippet,statistics',
    id='{},'.format(videoId)
).execute()
# snippet
snippetInfo = videos_response["items"][0]["snippet"]
# 動画タイトル
title = snippetInfo['title']
# チャンネル名
channeltitle = snippetInfo['channelTitle']

# description
description = snippetInfo['description']


print(channeltitle)
print(title)
print(description)