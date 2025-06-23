import os
import csv
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import datetime
import matplotlib

# 日本語フォント設定（文字化け対策）
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic',
                                      'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# 設定パラメータ
MOVIES_DIR = "/data/reviews_per_movie"
OUTPUT_DIR = "/data/analyze"

# 出力ディレクトリが存在しない場合は作成
os.makedirs(OUTPUT_DIR, exist_ok=True)


def count_sentences(text):
    """レビュー内の文の数を概算する"""
    # HTMLタグを削除
    text = re.sub(r'<br/>', ' ', text)
    # 文の区切りを検出 (., !, ? の後にスペースか文字列終端)
    sentences = re.split(r'[.!?](?:\s|$)', text)
    # 空の要素を除去
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)


def analyze_all_review_lengths():
    """全レビュー文章の長さ（センテンス数）を分析する"""
    movie_files = [f for f in os.listdir(MOVIES_DIR) if f.endswith(".csv")]

    # 分析結果を格納するリスト（ファイル出力用）
    output_lines = []
    output_lines.append(f"レビュー長さ分析結果 - {datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    output_lines.append("=" * 60)

    print(f"全映画数: {len(movie_files)}")
    output_lines.append(f"全映画数: {len(movie_files)}")

    # 分析結果を格納するリスト
    sentence_counts = []
    word_counts = []
    detailed_data = []  # CSVファイル用の詳細データ

    # 処理進捗表示用の変数
    total_movies = len(movie_files)
    processed_movies = 0
    start_time = time.time()
    total_reviews = 0

    print("全レビューの分析を開始...")
    output_lines.append("全レビューの分析を開始...")

    # 全ての映画ファイルを処理
    for movie_file in movie_files:
        movie_path = os.path.join(MOVIES_DIR, movie_file)
        movie_name = os.path.splitext(movie_file)[0]

        try:
            with open(movie_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                # この映画の全レビューを処理
                for review_idx, review in enumerate(reader):
                    text = review['review']
                    sentences = count_sentences(text)
                    words = len(re.findall(r'\b\w+\b', text))

                    sentence_counts.append(sentences)
                    word_counts.append(words)

                    # 詳細データに追加（CSVファイル用）
                    detailed_data.append([movie_name, review_idx + 1, sentences, words, len(text)])

                    total_reviews += 1

            # 進捗表示
            processed_movies += 1
            if processed_movies % 10 == 0 or processed_movies == total_movies:
                elapsed_time = time.time() - start_time
                reviews_per_sec = total_reviews / elapsed_time if elapsed_time > 0 else 0
                progress_msg = (f"進捗: {processed_movies}/{total_movies} 映画処理済み "
                                f"({processed_movies / total_movies * 100:.1f}%) - "
                                f"合計 {total_reviews} レビュー - {reviews_per_sec:.1f} レビュー/秒")
                print(progress_msg)
                output_lines.append(progress_msg)

        except Exception as e:
            error_msg = f"エラー ({movie_file}): {e}"
            print(error_msg)
            output_lines.append(error_msg)

    total_msg = f"\n全レビュー数: {total_reviews}"
    print(total_msg)
    output_lines.append(total_msg)

    # 統計情報の計算
    sentence_stats = {
        "最小": min(sentence_counts),
        "最大": max(sentence_counts),
        "平均": np.mean(sentence_counts),
        "中央値": np.median(sentence_counts),
        "標準偏差": np.std(sentence_counts)
    }

    word_stats = {
        "最小": min(word_counts),
        "最大": max(word_counts),
        "平均": np.mean(word_counts),
        "中央値": np.median(word_counts),
        "標準偏差": np.std(word_counts)
    }

    # パーセンタイルの計算
    percentiles = [10, 25, 33, 50, 67, 75, 90, 95, 99]
    sentence_percentiles = np.percentile(sentence_counts, percentiles)
    word_percentiles = np.percentile(word_counts, percentiles)

    # 結果の表示とファイル出力用データの準備
    print("\n=== センテンス数の統計 ===")
    output_lines.append("\n=== センテンス数の統計 ===")
    for key, value in sentence_stats.items():
        line = f"{key}: {value:.1f}"
        print(line)
        output_lines.append(line)

    print("\n=== 単語数の統計 ===")
    output_lines.append("\n=== 単語数の統計 ===")
    for key, value in word_stats.items():
        line = f"{key}: {value:.1f}"
        print(line)
        output_lines.append(line)

    print("\n=== センテンス数のパーセンタイル ===")
    output_lines.append("\n=== センテンス数のパーセンタイル ===")
    for i, p in enumerate(percentiles):
        line = f"{p}パーセンタイル: {sentence_percentiles[i]:.1f}"
        print(line)
        output_lines.append(line)

    print("\n=== 単語数のパーセンタイル ===")
    output_lines.append("\n=== 単語数のパーセンタイル ===")
    for i, p in enumerate(percentiles):
        line = f"{p}パーセンタイル: {word_percentiles[i]:.1f}"
        print(line)
        output_lines.append(line)

    # センテンス数の分布をプロット（上限を設定して見やすくする）
    plt.figure(figsize=(15, 8))

    # センテンス数のヒストグラム（上限100センテンスまで）
    plt.subplot(1, 2, 1)
    max_sentences_to_plot = 100
    plt.hist([min(c, max_sentences_to_plot) for c in sentence_counts],
             bins=50, range=(0, max_sentences_to_plot), alpha=0.7, edgecolor='black')
    plt.title('レビューあたりのセンテンス数の分布 (～100文)', fontsize=14, pad=15)
    plt.xlabel('センテンス数', fontsize=12)
    plt.ylabel('レビュー数', fontsize=12)
    plt.grid(True, alpha=0.3)

    # 単語数のヒストグラム（上限1000単語まで）
    plt.subplot(1, 2, 2)
    max_words_to_plot = 1000
    plt.hist([min(c, max_words_to_plot) for c in word_counts],
             bins=50, range=(0, max_words_to_plot), alpha=0.7, edgecolor='black')
    plt.title('レビューあたりの単語数の分布 (～1000語)', fontsize=14, pad=15)
    plt.xlabel('単語数', fontsize=12)
    plt.ylabel('レビュー数', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # グラフの保存先を指定のディレクトリに変更
    graph_path = os.path.join(OUTPUT_DIR,
                              f'review_length_distribution_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()  # メモリ節約のためクローズ

    graph_msg = f"グラフを '{graph_path}' に保存しました"
    print(f"\n{graph_msg}")
    output_lines.append(f"\n{graph_msg}")

    # 長さカテゴリの閾値を計算（33パーセンタイルと67パーセンタイル）
    short_threshold = sentence_percentiles[2]  # 33パーセンタイル
    medium_threshold = sentence_percentiles[4]  # 67パーセンタイル

    print("\n=== 長さカテゴリの推奨閾値 ===")
    output_lines.append("\n=== 長さカテゴリの推奨閾値 ===")
    threshold_lines = [
        f"短いレビュー: 1～{short_threshold:.0f}文",
        f"中程度のレビュー: {short_threshold + 1:.0f}～{medium_threshold:.0f}文",
        f"長いレビュー: {medium_threshold + 1:.0f}文以上"
    ]
    for line in threshold_lines:
        print(line)
        output_lines.append(line)

    # カテゴリごとのレビュー数と割合
    short_count = sum(1 for c in sentence_counts if c <= short_threshold)
    medium_count = sum(1 for c in sentence_counts if short_threshold < c <= medium_threshold)
    long_count = sum(1 for c in sentence_counts if c > medium_threshold)

    print("\n=== カテゴリ別レビュー数 ===")
    output_lines.append("\n=== カテゴリ別レビュー数 ===")
    category_lines = [
        f"短いレビュー: {short_count} ({short_count / total_reviews * 100:.1f}%)",
        f"中程度のレビュー: {medium_count} ({medium_count / total_reviews * 100:.1f}%)",
        f"長いレビュー: {long_count} ({long_count / total_reviews * 100:.1f}%)"
    ]
    for line in category_lines:
        print(line)
        output_lines.append(line)

    # テキストファイルとして結果を保存
    txt_path = os.path.join(OUTPUT_DIR,
                            f"review_length_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"\n分析結果を '{txt_path}' に保存しました")

    # 詳細データをCSVファイルとして保存
    csv_path = os.path.join(OUTPUT_DIR,
                            f"review_length_details_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['映画名', 'レビュー番号', 'センテンス数', '単語数', '文字数'])
        writer.writerows(detailed_data)

    print(f"詳細データを '{csv_path}' に保存しました")

    # 統計情報のCSVも作成
    stats_csv_path = os.path.join(OUTPUT_DIR,
                                  f"review_length_statistics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(stats_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # センテンス数統計
        writer.writerow(['センテンス数統計', '値'])
        for key, value in sentence_stats.items():
            writer.writerow([key, f"{value:.1f}"])

        writer.writerow(['', ''])  # 空行

        # 単語数統計
        writer.writerow(['単語数統計', '値'])
        for key, value in word_stats.items():
            writer.writerow([key, f"{value:.1f}"])

        writer.writerow(['', ''])  # 空行

        # パーセンタイル情報
        writer.writerow(['パーセンタイル', 'センテンス数', '単語数'])
        for i, p in enumerate(percentiles):
            writer.writerow([f"{p}%", f"{sentence_percentiles[i]:.1f}", f"{word_percentiles[i]:.1f}"])

        writer.writerow(['', '', ''])  # 空行

        # カテゴリ情報
        writer.writerow(['カテゴリ', 'レビュー数', '割合(%)'])
        writer.writerow(['短いレビュー', short_count, f"{short_count / total_reviews * 100:.1f}"])
        writer.writerow(['中程度のレビュー', medium_count, f"{medium_count / total_reviews * 100:.1f}"])
        writer.writerow(['長いレビュー', long_count, f"{long_count / total_reviews * 100:.1f}"])

    print(f"統計情報を '{stats_csv_path}' に保存しました")


if __name__ == "__main__":
    analyze_all_review_lengths()