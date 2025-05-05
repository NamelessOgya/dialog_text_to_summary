import re
from typing import List, Tuple

def extract_timelines(description: str) -> List[Tuple[str, str]]:
    """
    YouTube の概要欄テキストから
    時間形式 (MM:SS または H:MM:SS) とタイトルを抜き出して返す。

    Args:
        description: 概要欄全文の文字列

    Returns:
        List of (timestamp, title) タプルのリスト
    """
    pattern = re.compile(
        r'^\s*'                          # 行頭の空白を無視
        r'(?P<time>\d{1,2}(?::\d{2}){1,2})'  # MM:SS または H:MM:SS
        r'\s+'                           # 区切りの空白
        r'(?P<label>.+?)'                # タイトル（行末まで）
        r'\s*$',                         # 行末の空白を無視
        re.MULTILINE
    )
    return [(m.group('time'), m.group('label')) for m in pattern.finditer(description)]


def generate_formatted_text(description) -> str:
    """
    概要欄のテキストを整形して、時間とタイトルの文字列を生成する。
    時間の情報は会話の文字起こしからは知りえないので、時間は除外する。
    """
    timelines = extract_timelines(description)

    res = ""
    for time, title in timelines:
        res += "- "
        res += title
        res += "\n"
    return res

if __name__ == "__main__":
    sample = """
    【海と科学と三陸と――地方にある高等教育・研究機関の役割――】峰岸 有紀_2022年度夏学期：高校生と大学生のための金曜特別講座,"岩手県大槌町．2011年の東日本大震災で甚大な被害を受けた三陸沿岸の町のひとつです．この町で50年，純粋な知的好奇心に基づいて海洋科学研究をしてきた私たちは，震災後，地域と生きることを選びました．私たちが何を経験し，地域や私たち自身の役割をどう考え，地域の一員として今，何をしているのか，三陸にある高等教育・研究機関と地域の関わりについてお話しします．

    峰岸 有紀 （大気海洋研究所/准教授）
    ※所属・役職は講演当時のものです。

    03:55 イントロダクション
    18:00 震災直後の大槌、隣人と友人
    24:06 海あっての三陸
    28:08 社会科学研究所の希望学
    31:00 三陸をみてみよう
    37:42 ローカルアイデンティティ再構築のために
    1:06:55 おわりに
    1:09:08 質疑応答


    ★★東大TVウェブサイトのご案内★★
    東大TVのウェブサイト（ https://tv.he.u-tokyo.ac.jp/ ）には全部で2,000本近くの動画を公開！！
    著名な東大教授陣だけでなく、トマ・ピケティ氏、マイケル・サンデル氏ら、海外の著名人の講演も見られます！

    ↓↓↓

    東大TV ウェブサイト：https://tv.he.u-tokyo.ac.jp/ 
    東大TV X(Twitter)：https://x.com/UTokyoTV
    東大TV Facebook：https://www.facebook.com/todai.tv/
    UTokyo Online Education：https://oe.he.u-tokyo.ac.jp

    運営・著作権処理・映像編集：東京大学 大学総合教育研究センター"
    """
    result = generate_formatted_text(sample)
    
    print(result)
