import os
import csv
import random
import re
from collections import defaultdict
import datetime

# 設定パラメータ
MOVIES_DIR = "/Users/watanabesaki/PycharmProjects/sotsuron/data/reviews_per_movie"
OUTPUT_DIR = "/Users/watanabesaki/PycharmProjects/sotsuron"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"sampled_reviews_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
SAMPLE_RATIO = 0.7  # 映画全体の70%を選択（300-400映画を目標）

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


def categorize_review_length(sentences):
    """レビューを長さに基づいて分類する (分析結果に基づく)"""
    if sentences <= 7:  # 33パーセンタイル
        return "short"
    elif sentences <= 13:  # 67パーセンタイル
        return "medium"
    else:
        return "long"


def sample_reviews():
    """映画レビューをサンプリングする"""
    # 全映画ファイルのリストを取得
    movie_files = [f for f in os.listdir(MOVIES_DIR) if f.endswith(".csv")]
    print(f"全映画数: {len(movie_files)}")

    # 映画をランダムに選択（より多くの映画をカバー）
    num_movies_to_sample = max(1, int(len(movie_files) * SAMPLE_RATIO))
    selected_movie_files = random.sample(movie_files, num_movies_to_sample)
    print(f"選択された映画数: {len(selected_movie_files)}")

    sampled_reviews = []
    movie_review_counts = {}
    movie_stats = []  # 統計情報記録用

    # 各映画からレビューをサンプリング
    for movie_file in selected_movie_files:
        movie_path = os.path.join(MOVIES_DIR, movie_file)
        movie_name = os.path.splitext(movie_file)[0]

        reviews = []
        try:
            with open(movie_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                reviews = list(reader)
        except Exception as e:
            print(f"エラー ({movie_file}): {e}")
            continue

        # レビュー数に応じたサンプルサイズを調整
        if len(reviews) > 1000:
            sample_size = 80
        elif len(reviews) > 500:
            sample_size = 65
        elif len(reviews) > 200:
            sample_size = 50
        elif len(reviews) > 100:
            sample_size = 35
        elif len(reviews) > 50:
            sample_size = 25
        else:
            sample_size = max(15, int(len(reviews) * 0.75))

        # レビューの長さによるカテゴリ分け
        reviews_by_length = defaultdict(list)

        for review in reviews:
            sentences = count_sentences(review['review'])
            length_category = categorize_review_length(sentences)
            reviews_by_length[length_category].append(review)

        # 各長さカテゴリからバランス良くサンプリング (均等に)
        movie_samples = []
        target_per_category = {
            "short": int(sample_size * 0.34),  # 約34%
            "medium": int(sample_size * 0.33),  # 約33%
            "long": int(sample_size * 0.33)  # 約33%
        }

        # 各カテゴリが少なくとも1つ以上になるように調整
        for category in target_per_category:
            if target_per_category[category] < 1 and len(reviews_by_length[category]) > 0:
                target_per_category[category] = 1

        # 合計が目標サンプルサイズになるように調整
        remaining = sample_size - sum(target_per_category.values())
        if remaining != 0:
            categories = list(target_per_category.keys())
            for _ in range(abs(remaining)):
                category = random.choice(categories)
                if remaining > 0:
                    target_per_category[category] += 1
                else:
                    target_per_category[category] = max(0, target_per_category[category] - 1)

        # 各カテゴリからサンプリング
        for category, target_count in target_per_category.items():
            category_reviews = reviews_by_length[category]
            if category_reviews:
                category_count = min(target_count, len(category_reviews))
                sampled_category = random.sample(category_reviews, category_count)
                movie_samples.extend(sampled_category)

        # カテゴリ分けで目標サンプル数に達しなかった場合、残りをランダムに追加
        remaining = sample_size - len(movie_samples)
        if remaining > 0 and len(reviews) > len(movie_samples):
            # すでにサンプリングされたレビューを除外
            remaining_reviews = [r for r in reviews if r not in movie_samples]
            if remaining_reviews:
                additional_samples = random.sample(remaining_reviews, min(remaining, len(remaining_reviews)))
                movie_samples.extend(additional_samples)

        # メタデータを追加（映画名）
        for review in movie_samples:
            review['movie'] = movie_name
            # センテンス数を追加
            review['sentence_count'] = count_sentences(review['review'])

        sampled_reviews.extend(movie_samples)
        movie_review_counts[movie_name] = len(movie_samples)

        # 統計情報を記録
        movie_stats.append({
            'movie': movie_name,
            'original_reviews': len(reviews),
            'sampled_reviews': len(movie_samples),
            'sampling_ratio': len(movie_samples) / len(reviews) if len(reviews) > 0 else 0
        })

    print(f"サンプリングされた総レビュー数: {len(sampled_reviews)}")

    # カテゴリ別の統計
    categories = {"short": 0, "medium": 0, "long": 0}
    for review in sampled_reviews:
        length_category = categorize_review_length(int(review['sentence_count']))
        categories[length_category] += 1

    print("\nサンプリングしたレビューの長さ分布:")
    for category, count in categories.items():
        percent = (count / len(sampled_reviews)) * 100 if sampled_reviews else 0
        print(f"{category}: {count} ({percent:.1f}%)")

    # 映画別サンプリング統計
    print(f"\n映画別サンプリング統計:")
    print(f"1映画あたりの平均サンプル数: {len(sampled_reviews) / len(selected_movie_files):.1f}")

    # サンプリング比率の統計
    sampling_ratios = [stat['sampling_ratio'] for stat in movie_stats]
    print(f"平均サンプリング比率: {sum(sampling_ratios) / len(sampling_ratios):.3f}")
    print(f"最小サンプリング比率: {min(sampling_ratios):.3f}")
    print(f"最大サンプリング比率: {max(sampling_ratios):.3f}")

    # 結果を保存
    if sampled_reviews:
        # 列の順序を決定（'movie'と'sentence_count'を優先）
        base_fields = ['movie', 'sentence_count']
        other_fields = [f for f in sampled_reviews[0].keys() if f not in base_fields]
        fieldnames = base_fields + other_fields

        with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sampled_reviews)

        print(f"\nサンプリング結果を {OUTPUT_FILE} に保存しました")

        # 統計情報もCSVで保存
        stats_file = os.path.join(OUTPUT_DIR,
                                  f"sampling_statistics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(stats_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['movie', 'original_reviews', 'sampled_reviews', 'sampling_ratio']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(movie_stats)

        print(f"統計情報を {stats_file} に保存しました")

        # サマリー情報をテキストファイルで保存
        summary_file = os.path.join(OUTPUT_DIR,
                                    f"sampling_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"映画レビューサンプリング結果 - {datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"全映画数: {len(movie_files)}\n")
            f.write(f"選択された映画数: {len(selected_movie_files)}\n")
            f.write(f"サンプリング比率: {SAMPLE_RATIO:.1%}\n")
            f.write(f"サンプリングされた総レビュー数: {len(sampled_reviews)}\n\n")

            f.write("レビュー長さ分布:\n")
            for category, count in categories.items():
                percent = (count / len(sampled_reviews)) * 100 if sampled_reviews else 0
                f.write(f"  {category}: {count} ({percent:.1f}%)\n")

            f.write(f"\n映画別統計:\n")
            f.write(f"  1映画あたりの平均サンプル数: {len(sampled_reviews) / len(selected_movie_files):.1f}\n")
            f.write(f"  平均サンプリング比率: {sum(sampling_ratios) / len(sampling_ratios):.3f}\n")
            f.write(f"  最小サンプリング比率: {min(sampling_ratios):.3f}\n")
            f.write(f"  最大サンプリング比率: {max(sampling_ratios):.3f}\n")

        print(f"サマリー情報を {summary_file} に保存しました")

    else:
        print("サンプリングするレビューがありませんでした。")


if __name__ == "__main__":
    sample_reviews()