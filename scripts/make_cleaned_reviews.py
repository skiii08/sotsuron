import csv
import re
import html
import unicodedata

# 入力ファイルと出力ファイルのパス
INPUT_FILE = "/Users/watanabesaki/PycharmProjects/sotsuron/Toy Story 1995.csv"
OUTPUT_FILE = "/Users/watanabesaki/PycharmProjects/sotsuron/Cleanded_Toy Story 1995.csv"


def clean_text(text):
    """テキストをクレンジングする"""
    if not isinstance(text, str):
        return ""

    # HTML エンティティをデコード
    text = html.unescape(text)

    # HTML タグを削除
    text = re.sub(r'<[^>]+>', ' ', text)

    # 引用符の標準化（すべての引用符を二重引用符に統一）
    # 単一引用符類を二重引用符に変換
    text = re.sub(r'[\u2018\u2019\u201A\u201B\u2032\u2035\']', '"', text)
    # 二重引用符類を標準の二重引用符に変換
    text = re.sub(r'[\u201C\u201D\u201E\u201F\u2033\u2036]', '"', text)

    # 特殊文字の削除（制御文字など）
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)

    # Unicode 正規化
    text = unicodedata.normalize('NFC', text)

    # 複数の連続スペースを単一スペースに置換
    text = re.sub(r'\s+', ' ', text)

    # 句読点の標準化
    # ピリオド、感嘆符、疑問符の後にスペースがない場合に追加
    text = re.sub(r'([.!?])([A-Za-z0-9])', r'\1 \2', text)

    # コンマ、セミコロン、コロンの後にスペースがない場合に追加
    text = re.sub(r'([,;:])([A-Za-z0-9])', r'\1 \2', text)

    # 一貫した大文字/小文字処理
    # 文の先頭が小文字の場合、大文字に変換
    text = re.sub(r'(?<=[\.\?!]\s)([a-z])', lambda m: m.group(1).upper(), text)

    # 文章の先頭が小文字の場合、大文字に変換
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # 文章の前後の空白を削除
    text = text.strip()

    return text


def cleanse_reviews(input_file, output_file):
    """CSV ファイル内のレビュー列をクレンジングして新しいファイルに保存"""

    with open(input_file, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        if 'review' not in fieldnames:
            raise ValueError("入力ファイルに 'review' 列が存在しません。")

        cleaned_rows = []
        for row in reader:
            row['review'] = clean_text(row['review'])  # review 列をクレンジング
            cleaned_rows.append(row)

        with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cleaned_rows)

        print(f"クレンジングされたレビューを {output_file} に保存しました。")


if __name__ == "__main__":
    cleanse_reviews(INPUT_FILE, OUTPUT_FILE)
