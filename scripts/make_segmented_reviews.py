import csv
import spacy
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


def process_file(input_file, output_file):
    """
    CSVファイルを処理してセグメント化
    """
    # 入力ファイルの行数をカウント
    with open(input_file, 'r', encoding='utf-8') as infile:
        total_rows = sum(1 for line in infile) - 1

    print(f"処理対象: {total_rows}行のレビュー")

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)
        # 元のファイルの全カラムを保持
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        print(f"保持するカラム: {', '.join(fieldnames)}")

        total_segments = 0
        processed_reviews = 0
        segment_stats = []

        for row in tqdm(reader, desc="Processing reviews", total=total_rows):
            try:
                movie = row['movie']
                username = row['username']
                review = row['review']

                if not review or not review.strip():
                    continue

                # レビューをセグメント化
                segmented_reviews = segment_text(review)

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
                print(f"\nエラー（行 {processed_reviews + 1}）: {e}")
                continue

    # 統計情報の表示
    avg_segments = total_segments / processed_reviews if processed_reviews > 0 else 0

    print(f"\n=== 処理完了 ===")
    print(f"処理したレビュー数: {processed_reviews}")
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


# ファイルパス
input_file = '/Users/watanabesaki/PycharmProjects/sotsuron/cleaned_reviews.csv'
output_file = '/Users/watanabesaki/PycharmProjects/sotsuron/segmented_reviews.csv'

if __name__ == "__main__":
    process_file(input_file, output_file)