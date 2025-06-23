import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib
import datetime

# 日本語フォント設定（文字化け対策）
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic',
                                      'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

# 設定パラメータ
MOVIES_DIR = "/Users/watanabesaki/PycharmProjects/PythonProject/data/2_reviews_per_movie_raw"
OUTPUT_DIR = "/data/analyze"

# 出力ディレクトリが存在しない場合は作成
os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_review_counts():
    """映画ごとのレビュー数を分析する"""
    movie_files = [f for f in os.listdir(MOVIES_DIR) if f.endswith(".csv")]
    review_counts = {}

    # 分析結果を格納するリスト（後でファイル出力用）
    output_lines = []
    output_lines.append(f"映画レビュー分析結果 - {datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    output_lines.append("=" * 60)

    print(f"全映画数: {len(movie_files)}")
    print("レビュー数の集計中...")
    output_lines.append(f"全映画数: {len(movie_files)}")

    # 各映画ファイルのレビュー数をカウント
    for movie_file in movie_files:
        movie_path = os.path.join(MOVIES_DIR, movie_file)
        movie_name = os.path.splitext(movie_file)[0]

        try:
            # CSVファイルを読み込み、行数をカウント
            with open(movie_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # ヘッダーをスキップ
                count = sum(1 for _ in reader)

            review_counts[movie_name] = count

        except Exception as e:
            error_msg = f"エラー ({movie_file}): {e}"
            print(error_msg)
            output_lines.append(error_msg)
            review_counts[movie_name] = 0

    # 統計情報の計算
    counts = list(review_counts.values())

    if counts:
        stats = {
            "総映画数": len(counts),
            "レビュー総数": sum(counts),
            "最小レビュー数": min(counts),
            "最大レビュー数": max(counts),
            "平均レビュー数": sum(counts) / len(counts),
            "中央値レビュー数": sorted(counts)[len(counts) // 2]
        }

        # レビュー数の分布を計算
        distribution = defaultdict(int)
        for count in counts:
            if count <= 10:
                distribution["1-10"] += 1
            elif count <= 50:
                distribution["11-50"] += 1
            elif count <= 100:
                distribution["51-100"] += 1
            elif count <= 200:
                distribution["101-200"] += 1
            elif count <= 500:
                distribution["201-500"] += 1
            elif count <= 1000:
                distribution["501-1000"] += 1
            else:
                distribution["1000+"] += 1

        # 結果の表示とファイル出力用データの準備
        print("\n=== 統計情報 ===")
        output_lines.append("\n=== 統計情報 ===")
        for key, value in stats.items():
            line = f"{key}: {value:.1f}" if isinstance(value, float) else f"{key}: {value}"
            print(line)
            output_lines.append(line)

        print("\n=== レビュー数の分布 ===")
        output_lines.append("\n=== レビュー数の分布 ===")
        for category, count in sorted(distribution.items(), key=lambda x: x[0]):
            percentage = (count / len(movie_files)) * 100
            line = f"{category}: {count} 映画 ({percentage:.1f}%)"
            print(line)
            output_lines.append(line)

        # レビュー数の分布をプロットする（文字化け対策済み）
        plt.figure(figsize=(12, 8))
        plt.hist(counts, bins=[0, 10, 50, 100, 200, 500, 1000, max(counts)], alpha=0.7, edgecolor='black')
        plt.title('映画あたりのレビュー数の分布', fontsize=16, pad=20)
        plt.xlabel('レビュー数', fontsize=14)
        plt.ylabel('映画数', fontsize=14)
        plt.grid(True, alpha=0.3)

        # グラフの保存先を指定のディレクトリに変更
        graph_path = os.path.join(OUTPUT_DIR, 'review_count_distribution.png')
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()  # メモリ節約のためクローズ

        graph_msg = f"グラフを '{graph_path}' に保存しました"
        print(f"\n{graph_msg}")
        output_lines.append(f"\n{graph_msg}")

        # 上位と下位の映画
        print("\n=== レビュー数上位10映画 ===")
        output_lines.append("\n=== レビュー数上位10映画 ===")
        top_movies = sorted(review_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for movie, count in top_movies:
            line = f"{movie}: {count} レビュー"
            print(line)
            output_lines.append(line)

        print("\n=== レビュー数下位10映画 ===")
        output_lines.append("\n=== レビュー数下位10映画 ===")
        bottom_movies = sorted(review_counts.items(), key=lambda x: x[1])[:10]
        for movie, count in bottom_movies:
            line = f"{movie}: {count} レビュー"
            print(line)
            output_lines.append(line)

        # pandasを使用して詳細な分位数を計算
        df = pd.DataFrame({'review_count': counts})
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        quantiles = df['review_count'].quantile(q=[p / 100 for p in percentiles])

        print("\n=== パーセンタイル ===")
        output_lines.append("\n=== パーセンタイル ===")
        for i, p in enumerate(percentiles):
            line = f"{p}パーセンタイル: {quantiles.iloc[i]:.1f}"
            print(line)
            output_lines.append(line)

        # テキストファイルとして結果を保存
        txt_path = os.path.join(OUTPUT_DIR,
                                f"review_analysis_result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))

        print(f"\n分析結果を '{txt_path}' に保存しました")

        # CSVファイルとして詳細データを保存
        csv_data = []
        # 映画別レビュー数
        for movie, count in sorted(review_counts.items(), key=lambda x: x[1], reverse=True):
            csv_data.append([movie, count])

        csv_path = os.path.join(OUTPUT_DIR,
                                f"movie_review_counts_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['映画名', 'レビュー数'])
            writer.writerows(csv_data)

        print(f"詳細データを '{csv_path}' に保存しました")

        # 統計情報のCSVも作成
        stats_csv_path = os.path.join(OUTPUT_DIR,
                                      f"review_statistics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(stats_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['統計項目', '値'])
            for key, value in stats.items():
                writer.writerow([key, value])

            # 分布情報も追加
            writer.writerow(['', ''])  # 空行
            writer.writerow(['レビュー数範囲', '映画数', '割合(%)'])
            for category, count in sorted(distribution.items(), key=lambda x: x[0]):
                percentage = (count / len(movie_files)) * 100
                writer.writerow([category, count, f"{percentage:.1f}"])

        print(f"統計情報を '{stats_csv_path}' に保存しました")

    else:
        error_msg = "レビューデータを取得できませんでした。"
        print(error_msg)
        output_lines.append(error_msg)

        # エラーの場合もファイルに保存
        txt_path = os.path.join(OUTPUT_DIR,
                                f"review_analysis_error_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))


if __name__ == "__main__":
    analyze_review_counts()