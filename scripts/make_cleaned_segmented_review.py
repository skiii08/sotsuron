import csv
import re
import html
import unicodedata
import spacy
import os
import glob
from tqdm import tqdm

# spaCyモデルを最初に一度だけロード
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCyモデル 'en_core_web_sm' を正常にロードしました")
except OSError:
    print("エラー: spaCyモデル 'en_core_web_sm' が見つかりません")
    print("以下のコマンドでインストールしてください:")
    print("python -m spacy download en_core_web_sm")
    exit(1)

# ディレクトリパス（適宜変更してください）
INPUT_DIR = "/Users/watanabesaki/PycharmProjects/sotsuron/review_per_users"
OUTPUT_DIR = "/Users/watanabesaki/PycharmProjects/sotsuron/review_per_users/segmented"


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


def has_meaningful_content(segment_tokens):
    """セグメントが意味のある内容を持つかチェック"""
    # 名詞、動詞、形容詞のいずれかを含む場合は意味があると判定
    meaningful_pos = {'NOUN', 'VERB', 'ADJ', 'PROPN'}  # 固有名詞も追加
    return any(token.pos_ in meaningful_pos for token in segment_tokens)


def is_significant_conjunction(token, segment_tokens):
    """重要な接続詞かどうかを判定"""
    # 前のセグメントが短すぎる場合は分割しない
    if len(segment_tokens) < 3:
        return False

    # 前のセグメントに意味のある内容がない場合は分割しない
    if not has_meaningful_content(segment_tokens[:-1]):  # 現在のトークンを除く
        return False

    # 重要な接続詞のリスト（対比や転換を表すもの）
    significant_conjunctions = {
        'but', 'however', 'although', 'though', 'whereas',
        'nevertheless', 'nonetheless', 'yet', 'still'
    }

    # 単純な "and" は長いセグメントでのみ分割
    if token.text.lower() == 'and':
        return len(segment_tokens) >= 8  # 8単語以上の場合のみ

    return token.text.lower() in significant_conjunctions


def segment_text(text):
    """
    改良されたセグメンテーション関数
    """
    doc = nlp(text)
    segments = []
    current_segment = []

    for token in doc:
        current_segment.append(token)

        # 句点での分割（従来通り）
        if token.is_punct and token.text == ".":
            if current_segment:
                segment_text = " ".join([t.text for t in current_segment]).strip()
                if segment_text:
                    segments.append(segment_text)
                current_segment = []

        # 改良された接続詞での分割
        elif (token.dep_ == "cc" and
              token.head.dep_ in {"ROOT", "ccomp", "advcl"} and
              is_significant_conjunction(token, current_segment)):

            # 現在のトークンを除いたセグメントを保存
            if len(current_segment) > 1:
                segment_text = " ".join([t.text for t in current_segment[:-1]]).strip()
                if segment_text:
                    segments.append(segment_text)

            # 新しいセグメントを接続詞から開始
            current_segment = [token]

    # 最後のセグメント
    if current_segment:
        segment_text = " ".join([t.text for t in current_segment]).strip()
        if segment_text:
            segments.append(segment_text)

    # 後処理：短すぎるセグメントを結合
    return merge_short_segments(segments)


def merge_short_segments(segments, min_words=4, min_chars=20):
    """
    短すぎるセグメントを前のセグメントと結合
    """
    if not segments:
        return segments

    merged_segments = []

    for segment in segments:
        word_count = len(segment.split())
        char_count = len(segment.strip())

        # 最小要件を満たさない場合
        if (word_count < min_words or char_count < min_chars) and merged_segments:
            # 前のセグメントと結合
            merged_segments[-1] = merged_segments[-1] + " " + segment
        else:
            # 新しいセグメントとして追加
            merged_segments.append(segment)

    # 再度チェック：最初のセグメントが短い場合
    if len(merged_segments) > 1:
        first_segment = merged_segments[0]
        if len(first_segment.split()) < min_words or len(first_segment.strip()) < min_chars:
            # 2番目のセグメントと結合
            merged_segments[1] = first_segment + " " + merged_segments[1]
            merged_segments = merged_segments[1:]

    # 空のセグメントを除去
    return [seg.strip() for seg in merged_segments if seg.strip()]


def get_csv_files(directory):
    """指定ディレクトリ内のCSVファイル一覧を取得"""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    return sorted(csv_files)


def create_output_filename(input_file, output_dir):
    """出力ファイル名を生成"""
    filename = os.path.basename(input_file)
    name, ext = os.path.splitext(filename)
    output_filename = f"segmented_{name}{ext}"
    return os.path.join(output_dir, output_filename)


def process_file(input_file, output_file):
    """
    CSVファイルを処理してクレンジングとセグメント化を一気に実行
    """
    filename = os.path.basename(input_file)
    print(f"\n=== 処理開始: {filename} ===")

    # 入力ファイルの行数をカウント
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            total_rows = sum(1 for line in infile) - 1
    except Exception as e:
        print(f"エラー: ファイル読み込み失敗 - {e}")
        return False

    print(f"処理対象: {total_rows}行のレビュー")

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
                open(output_file, 'w', encoding='utf-8', newline='') as outfile:

            reader = csv.DictReader(infile)
            # 元のファイルの全カラムを保持
            fieldnames = reader.fieldnames

            if not fieldnames or 'review' not in fieldnames:
                print(f"警告: {filename} に 'review' カラムが見つかりません。スキップします。")
                return False

            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            print(f"保持するカラム: {', '.join(fieldnames)}")

            total_segments = 0
            processed_reviews = 0
            segment_stats = []
            skipped_reviews = 0

            for row in tqdm(reader, desc=f"Processing {filename}", total=total_rows):
                try:
                    # movie_titleカラムの取得（ログ用、存在しない場合は"unknown"）
                    movie_title = row.get('movie_title', 'unknown')

                    # reviewカラムの取得
                    review = row.get('review', '')

                    if not review or not review.strip():
                        skipped_reviews += 1
                        continue

                    # ステップ1: レビューをクレンジング
                    cleaned_review = clean_text(review)

                    if not cleaned_review or not cleaned_review.strip():
                        skipped_reviews += 1
                        continue

                    # ステップ2: クレンジングされたレビューをセグメント化
                    segmented_reviews = segment_text(cleaned_review)

                    # 統計記録
                    segment_stats.append(len(segmented_reviews))

                    # セグメントごとに行を追加
                    for segment in segmented_reviews:
                        if segment.strip():
                            # 元の行のすべてのデータをコピー
                            new_row = row.copy()
                            # reviewカラムのみセグメントに置き換え
                            new_row['review'] = segment
                            writer.writerow(new_row)
                            total_segments += 1

                    processed_reviews += 1

                except Exception as e:
                    print(f"\nエラー（行 {processed_reviews + 1}、映画: {movie_title}）: {e}")
                    continue

        # 統計情報の表示
        avg_segments = total_segments / processed_reviews if processed_reviews > 0 else 0

        print(f"\n=== {filename} 処理完了 ===")
        print(f"処理したレビュー数: {processed_reviews}")
        print(f"スキップしたレビュー数: {skipped_reviews}")
        print(f"生成されたセグメント数: {total_segments}")
        print(f"平均セグメント数/レビュー: {avg_segments:.2f}")

        # セグメント数の分布
        if segment_stats:
            segment_stats.sort()
            print(f"セグメント数の分布:")
            print(f"  最小: {min(segment_stats)}")
            print(f"  最大: {max(segment_stats)}")
            print(f"  中央値: {segment_stats[len(segment_stats) // 2]}")

            # 1-3セグメント、4-6セグメント、7+セグメントの割合
            counts_1_3 = sum(1 for x in segment_stats if 1 <= x <= 3)
            counts_4_6 = sum(1 for x in segment_stats if 4 <= x <= 6)
            counts_7_plus = sum(1 for x in segment_stats if x >= 7)

            print(f"  1-3セグメント: {counts_1_3} ({counts_1_3 / len(segment_stats) * 100:.1f}%)")
            print(f"  4-6セグメント: {counts_4_6} ({counts_4_6 / len(segment_stats) * 100:.1f}%)")
            print(f"  7+セグメント: {counts_7_plus} ({counts_7_plus / len(segment_stats) * 100:.1f}%)")

        print(f"出力ファイル: {output_file}")
        return True

    except Exception as e:
        print(f"エラー: {filename} の処理中にエラーが発生しました - {e}")
        return False


def main():
    """メイン処理関数"""
    print("=== 統合テキスト処理システム（バッチ処理） ===")
    print("処理内容:")
    print("1. ディレクトリ内全CSVファイルの一括処理")
    print("2. テキストクレンジング + セグメンテーション")
    print("=" * 50)

    try:
        # 出力ディレクトリの作成
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"出力ディレクトリ: {OUTPUT_DIR}")

        # CSVファイル一覧を取得
        csv_files = get_csv_files(INPUT_DIR)

        if not csv_files:
            print(f"エラー: {INPUT_DIR} 内にCSVファイルが見つかりません。")
            return 1

        print(f"\n発見されたCSVファイル数: {len(csv_files)}")
        for i, file in enumerate(csv_files, 1):
            print(f"  {i}. {os.path.basename(file)}")

        # 各ファイルを処理
        successful_files = 0
        failed_files = 0

        for input_file in csv_files:
            output_file = create_output_filename(input_file, OUTPUT_DIR)

            # ファイル処理
            if process_file(input_file, output_file):
                successful_files += 1
            else:
                failed_files += 1

        # 最終結果
        print(f"\n{'=' * 50}")
        print(f"=== 全体処理完了 ===")
        print(f"総ファイル数: {len(csv_files)}")
        print(f"成功: {successful_files}")
        print(f"失敗: {failed_files}")
        print(f"入力ディレクトリ: {INPUT_DIR}")
        print(f"出力ディレクトリ: {OUTPUT_DIR}")

        if failed_files > 0:
            print(f"\n注意: {failed_files}個のファイルの処理に失敗しました。")
            return 1

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())