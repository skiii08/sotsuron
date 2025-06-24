import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

import re

warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao',
                               'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


class TopicClusteringAnalyzer:
    def __init__(self, file1_path, file2_path,
                 output_dir="/Users/watanabesaki/PycharmProjects/sotsuron/data/analyze/topic"):
        """
        トピッククラスタリング専用分析クラス
        """
        self.file1_path = file1_path
        self.file2_path = file2_path
        self.output_dir = os.path.join(output_dir, 'clustering')

        # データ格納
        self.topic_counts = None
        self.topics = None
        self.counts = None

        # クラスタリング結果
        self.large_clusters = {}  # 12個の大クラスタ
        self.small_clusters = {}  # 50個の小クラスタ
        self.feature_vectors = None

        self.setup_directories()
        self.load_data()

        # クラスタ名前付けのための辞書
        self.cluster_themes = {
            # 映画制作関連
            'production': ['production', 'budget', 'filming', 'technology', 'technical', 'industry', 'filmmaking'],
            'visual': ['cinematography', 'visual', 'visuals', 'effects', 'cgi', 'animation', 'lighting', 'color',
                       'art direction', 'makeup', 'costume'],
            'audio': ['music', 'soundtrack', 'sound', 'voice acting', 'singing', 'song'],
            'editing': ['editing', 'pacing', 'structure', 'montage', 'continuity'],

            # ストーリー・脚本関連
            'story': ['plot', 'story', 'storyline', 'narrative', 'premise', 'ending', 'climax'],
            'script': ['screenplay', 'script', 'writing', 'dialogue', 'exposition'],
            'theme': ['theme', 'message', 'meaning', 'symbolism', 'metaphor'],

            # 演技・キャスト関連
            'acting': ['acting', 'performance', 'talent', 'believability'],
            'cast': ['cast', 'casting', 'actor', 'chemistry', 'cameo'],
            'character': ['character', 'characterization', 'protagonist', 'villain'],

            # 評価・感想関連
            'evaluation': ['opinion', 'criticism', 'evaluation', 'quality', 'rating'],
            'emotion': ['emotion', 'enjoyment', 'entertainment', 'suspense', 'humor'],
            'recommendation': ['recommendation', 'advice', 'expectation', 'preference'],

            # ジャンル・作品タイプ
            'genre': ['genre', 'comedy', 'horror', 'drama', 'romance', 'action'],
            'format': ['sequel', 'remake', 'adaptation', 'series', 'franchise'],

            # 商業・マーケティング
            'commercial': ['box office', 'commercial', 'marketing', 'popularity', 'success'],

            # 文脈・背景
            'context': ['accuracy', 'historical', 'cultural', 'authenticity', 'tradition'],
            'comparison': ['comparison', 'reference', 'influence', 'inspiration'],

            # 技術・メディア
            'technical': ['format', 'runtime', 'screening', 'distribution', 'release']
        }

    def setup_directories(self):
        """ディレクトリ設定"""
        os.makedirs(self.output_dir, exist_ok=True)
        subdirs = ['plots', 'interactive', 'data', 'reports']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        print(f"クラスタリング出力ディレクトリ: {self.output_dir}")

    def load_data(self):
        """データ読み込み"""
        print("データを読み込み中...")
        df1 = pd.read_csv(self.file1_path)
        df2 = pd.read_csv(self.file2_path)
        combined_data = pd.concat([df1, df2], ignore_index=True)

        self.topic_counts = Counter(combined_data['topic'].dropna())
        self.topics = list(self.topic_counts.keys())
        self.counts = list(self.topic_counts.values())

        print(f"総トピック数: {len(self.topics)}")
        print(f"総データ件数: {sum(self.counts)}")

    def create_feature_vectors(self):
        """特徴ベクトル作成"""
        print("特徴ベクトルを作成中...")

        # 複数の特徴を組み合わせ
        vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            max_features=1000,
            lowercase=True
        )

        self.feature_vectors = vectorizer.fit_transform(self.topics)
        print(f"特徴ベクトル形状: {self.feature_vectors.shape}")

        return self.feature_vectors

    def perform_hierarchical_clustering(self):
        """階層クラスタリング実行"""
        print("階層クラスタリングを実行中...")

        if self.feature_vectors is None:
            self.create_feature_vectors()

        # 大クラスタ（12個）
        print("大クラスタ（12個）を作成中...")
        kmeans_large = KMeans(n_clusters=12, random_state=42, n_init=10)
        large_labels = kmeans_large.fit_predict(self.feature_vectors)

        # 小クラスタ（50個）
        print("小クラスタ（50個）を作成中...")
        kmeans_small = KMeans(n_clusters=50, random_state=42, n_init=10)
        small_labels = kmeans_small.fit_predict(self.feature_vectors)

        # 結果を整理
        self.organize_clusters(large_labels, small_labels)

        # クラスタに名前をつける
        self.name_clusters()

        return self.large_clusters, self.small_clusters

    def organize_clusters(self, large_labels, small_labels):
        """クラスタ結果を整理"""
        # 大クラスタの整理
        for i, (topic, large_label, small_label) in enumerate(zip(self.topics, large_labels, small_labels)):
            if large_label not in self.large_clusters:
                self.large_clusters[large_label] = {
                    'topics': [],
                    'total_count': 0,
                    'name': f'大クラスタ_{large_label}'
                }

            self.large_clusters[large_label]['topics'].append({
                'topic': topic,
                'count': self.counts[i],
                'small_cluster': small_label
            })
            self.large_clusters[large_label]['total_count'] += self.counts[i]

        # 小クラスタの整理
        for i, (topic, small_label) in enumerate(zip(self.topics, small_labels)):
            if small_label not in self.small_clusters:
                self.small_clusters[small_label] = {
                    'topics': [],
                    'total_count': 0,
                    'name': f'小クラスタ_{small_label}'
                }

            self.small_clusters[small_label]['topics'].append({
                'topic': topic,
                'count': self.counts[i]
            })
            self.small_clusters[small_label]['total_count'] += self.counts[i]

        # 代表トピックを設定（最も出現回数が多いもの）
        for cluster in self.large_clusters.values():
            cluster['topics'].sort(key=lambda x: x['count'], reverse=True)
            cluster['representative'] = cluster['topics'][0]['topic']

        for cluster in self.small_clusters.values():
            cluster['topics'].sort(key=lambda x: x['count'], reverse=True)
            cluster['representative'] = cluster['topics'][0]['topic']

    def name_clusters(self):
        """クラスタに意味のある名前をつける"""
        print("クラスタに名前をつけています...")

        def get_cluster_theme(topics_list):
            """トピックリストからテーマを推定"""
            topic_words = [topic.lower() for topic in topics_list]

            # 各テーマとの一致度をチェック
            theme_scores = {}
            for theme, keywords in self.cluster_themes.items():
                score = 0
                for keyword in keywords:
                    for topic in topic_words:
                        if keyword in topic or topic in keyword:
                            score += 1
                theme_scores[theme] = score

            # 最も一致度の高いテーマを返す
            if theme_scores and max(theme_scores.values()) > 0:
                best_theme = max(theme_scores, key=theme_scores.get)
                return self.get_japanese_theme_name(best_theme)

            # マッチしない場合は代表トピックベースの名前
            return None

        def create_descriptive_name(representative_topic, topics_list):
            """説明的な名前を作成"""
            if len(topics_list) == 1:
                return f"{representative_topic}"
            elif len(topics_list) <= 3:
                return f"{representative_topic}関連"
            else:
                # 共通要素を探す
                common_parts = self.find_common_elements(topics_list)
                if common_parts:
                    return f"{common_parts}系"
                else:
                    return f"{representative_topic}グループ"

        # 大クラスタの命名
        for cluster_id, cluster in self.large_clusters.items():
            topics_list = [t['topic'] for t in cluster['topics']]
            theme_name = get_cluster_theme(topics_list)

            if theme_name:
                cluster['name'] = theme_name
            else:
                cluster['name'] = create_descriptive_name(cluster['representative'], topics_list)

        # 小クラスタの命名
        for cluster_id, cluster in self.small_clusters.items():
            topics_list = [t['topic'] for t in cluster['topics']]
            theme_name = get_cluster_theme(topics_list)

            if theme_name:
                cluster['name'] = theme_name
            else:
                cluster['name'] = create_descriptive_name(cluster['representative'], topics_list)

    def get_japanese_theme_name(self, theme):
        """英語テーマ名を日本語に変換"""
        theme_translations = {
            'production': '制作・プロダクション',
            'visual': '映像・ビジュアル',
            'audio': '音響・音楽',
            'editing': '編集・構成',
            'story': 'ストーリー・物語',
            'script': '脚本・台本',
            'theme': 'テーマ・メッセージ',
            'acting': '演技・パフォーマンス',
            'cast': 'キャスト・配役',
            'character': 'キャラクター',
            'evaluation': '評価・批評',
            'emotion': '感情・体験',
            'recommendation': '推薦・意見',
            'genre': 'ジャンル',
            'format': '作品形態',
            'commercial': '商業・興行',
            'context': '文脈・背景',
            'comparison': '比較・参照',
            'technical': '技術・メディア'
        }
        return theme_translations.get(theme, theme)

    def find_common_elements(self, topics_list):
        """トピックリストから共通要素を見つける"""
        if len(topics_list) < 2:
            return None

        # 共通する単語を探す
        words_sets = [set(re.findall(r'\w+', topic.lower())) for topic in topics_list]
        common_words = set.intersection(*words_sets)

        if common_words:
            # 最も意味のありそうな単語を選択
            meaningful_words = [w for w in common_words if len(w) > 2]
            if meaningful_words:
                return list(meaningful_words)[0]

        # 共通接頭辞を探す
        prefix = os.path.commonprefix(topics_list)
        if len(prefix) > 3:
            return prefix.rstrip('_- ')

        return None

    def visualize_hierarchical_clusters(self):
        """階層クラスタを可視化"""
        print("階層クラスタ可視化を作成中...")

        # 次元削減
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(self.feature_vectors.toarray())

        # 大クラスタと小クラスタのラベルを準備
        large_labels = []
        small_labels = []

        for topic in self.topics:
            # 大クラスタのラベルを見つける
            large_label = None
            small_label = None

            for cluster_id, cluster in self.large_clusters.items():
                if any(t['topic'] == topic for t in cluster['topics']):
                    large_label = cluster['name']
                    break

            for cluster_id, cluster in self.small_clusters.items():
                if any(t['topic'] == topic for t in cluster['topics']):
                    small_label = cluster['name']
                    break

            large_labels.append(large_label)
            small_labels.append(small_label)

        # 2つのサブプロットを作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

        # 大クラスタの可視化
        unique_large = list(set(large_labels))
        colors_large = plt.cm.Set3(np.linspace(0, 1, len(unique_large)))
        color_map_large = {label: colors_large[i] for i, label in enumerate(unique_large)}

        for i, (x, y) in enumerate(coords_2d):
            color = color_map_large.get(large_labels[i], 'gray')
            size = max(20, min(200, self.counts[i] / 5))
            ax1.scatter(x, y, c=[color], s=size, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax1.set_title('大クラスタ（12個）', fontsize=14, fontweight='bold')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

        # 凡例（大クラスタ）
        legend_elements_large = [plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=color_map_large[label],
                                            markersize=8, label=label[:15] + '...' if len(label) > 15 else label)
                                 for label in unique_large if label]
        ax1.legend(handles=legend_elements_large, bbox_to_anchor=(1.05, 1), loc='upper left')

        # 小クラスタの可視化（代表的なもののみ表示）
        # 出現回数の多い小クラスタのみ色分けして表示
        small_cluster_counts = {name: sum(t['count'] for t in cluster['topics'])
                                for name, cluster in self.small_clusters.items()}
        top_small_clusters = sorted(small_cluster_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        top_small_names = [name for name, _ in top_small_clusters]

        colors_small = plt.cm.tab20(np.linspace(0, 1, len(top_small_names)))
        color_map_small = {label: colors_small[i] for i, label in enumerate(top_small_names)}

        for i, (x, y) in enumerate(coords_2d):
            if small_labels[i] in color_map_small:
                color = color_map_small[small_labels[i]]
            else:
                color = 'lightgray'
            size = max(20, min(200, self.counts[i] / 5))
            ax2.scatter(x, y, c=[color], s=size, alpha=0.7, edgecolors='black', linewidth=0.5)

        ax2.set_title('小クラスタ（上位15個を色分け）', fontsize=14, fontweight='bold')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')

        # 凡例（小クラスタ）
        legend_elements_small = [plt.Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=color_map_small[label],
                                            markersize=8, label=label[:12] + '...' if len(label) > 12 else label)
                                 for label in top_small_names]
        ax2.legend(handles=legend_elements_small, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'plots', 'hierarchical_clusters_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"階層クラスタ可視化を保存: {output_path}")

    def create_cluster_summary_plots(self):
        """クラスタサマリープロットを作成"""
        print("クラスタサマリープロットを作成中...")

        # 大クラスタのサマリー
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

        # 大クラスタのサイズ分布
        large_names = [cluster['name'] for cluster in self.large_clusters.values()]
        large_sizes = [len(cluster['topics']) for cluster in self.large_clusters.values()]
        large_counts = [cluster['total_count'] for cluster in self.large_clusters.values()]

        bars1 = ax1.bar(range(len(large_names)), large_sizes, color='skyblue', alpha=0.7)
        ax1.set_xlabel('大クラスタ')
        ax1.set_ylabel('トピック数')
        ax1.set_title('大クラスタのトピック数分布', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(large_names)))
        ax1.set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in large_names],
                            rotation=45, ha='right')

        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

        # 大クラスタの出現回数分布
        bars2 = ax2.bar(range(len(large_names)), large_counts, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('大クラスタ')
        ax2.set_ylabel('総出現回数')
        ax2.set_title('大クラスタの総出現回数分布', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(large_names)))
        ax2.set_xticklabels([name[:10] + '...' if len(name) > 10 else name for name in large_names],
                            rotation=45, ha='right')

        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

        # 小クラスタのサイズ分布（ヒストグラム）
        small_sizes = [len(cluster['topics']) for cluster in self.small_clusters.values()]
        small_counts = [cluster['total_count'] for cluster in self.small_clusters.values()]

        ax3.hist(small_sizes, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('クラスタ内トピック数')
        ax3.set_ylabel('クラスタ数')
        ax3.set_title('小クラスタのサイズ分布', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 小クラスタの出現回数分布（ヒストグラム）
        ax4.hist(small_counts, bins=20, color='orange', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('クラスタ総出現回数')
        ax4.set_ylabel('クラスタ数')
        ax4.set_title('小クラスタの出現回数分布', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'plots', 'cluster_summary_plots.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"クラスタサマリープロットを保存: {output_path}")

    def create_interactive_cluster_dashboard(self):
        """インタラクティブダッシュボード作成"""
        print("インタラクティブダッシュボードを作成中...")

        # 次元削減
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(self.feature_vectors.toarray())

        # プロット用データ準備
        plot_data = []
        for i, topic in enumerate(self.topics):
            # 大クラスタを見つける
            large_cluster_name = None
            small_cluster_name = None

            for cluster_id, cluster in self.large_clusters.items():
                if any(t['topic'] == topic for t in cluster['topics']):
                    large_cluster_name = cluster['name']
                    break

            for cluster_id, cluster in self.small_clusters.items():
                if any(t['topic'] == topic for t in cluster['topics']):
                    small_cluster_name = cluster['name']
                    break

            plot_data.append({
                'x': coords_2d[i, 0],
                'y': coords_2d[i, 1],
                'topic': topic,
                'count': self.counts[i],
                'large_cluster': large_cluster_name or 'Unknown',
                'small_cluster': small_cluster_name or 'Unknown',
                'size': max(5, min(50, self.counts[i] / 20))
            })

        df_plot = pd.DataFrame(plot_data)

        # インタラクティブプロット作成
        fig = px.scatter(df_plot, x='x', y='y',
                         color='large_cluster',
                         size='size',
                         hover_data=['topic', 'count', 'small_cluster'],
                         title='階層トピッククラスタ分析ダッシュボード',
                         width=1400, height=800)

        fig.update_layout(
            title_x=0.5,
            xaxis_title=f'PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)',
            yaxis_title=f'PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)',
            showlegend=True
        )

        output_path = os.path.join(self.output_dir, 'interactive', 'hierarchical_cluster_dashboard.html')
        fig.write_html(output_path)
        print(f"インタラクティブダッシュボードを保存: {output_path}")

    def export_clustering_results(self):
        """クラスタリング結果をエクスポート"""
        print("クラスタリング結果を保存中...")

        # 大クラスタ結果をCSV出力
        large_data = []
        for cluster_id, cluster in self.large_clusters.items():
            for topic_info in cluster['topics']:
                large_data.append({
                    'cluster_id': cluster_id,
                    'cluster_name': cluster['name'],
                    'cluster_size': len(cluster['topics']),
                    'cluster_total_count': cluster['total_count'],
                    'representative_topic': cluster['representative'],
                    'topic': topic_info['topic'],
                    'topic_count': topic_info['count'],
                    'small_cluster_id': topic_info['small_cluster']
                })

        df_large = pd.DataFrame(large_data)
        large_path = os.path.join(self.output_dir, 'data', 'large_clusters_results.csv')
        df_large.to_csv(large_path, index=False, encoding='utf-8')

        # 小クラスタ結果をCSV出力
        small_data = []
        for cluster_id, cluster in self.small_clusters.items():
            for topic_info in cluster['topics']:
                small_data.append({
                    'cluster_id': cluster_id,
                    'cluster_name': cluster['name'],
                    'cluster_size': len(cluster['topics']),
                    'cluster_total_count': cluster['total_count'],
                    'representative_topic': cluster['representative'],
                    'topic': topic_info['topic'],
                    'topic_count': topic_info['count']
                })

        df_small = pd.DataFrame(small_data)
        small_path = os.path.join(self.output_dir, 'data', 'small_clusters_results.csv')
        df_small.to_csv(small_path, index=False, encoding='utf-8')

        # サマリーCSV
        large_summary = []
        for cluster_id, cluster in self.large_clusters.items():
            large_summary.append({
                'cluster_id': cluster_id,
                'cluster_name': cluster['name'],
                'representative_topic': cluster['representative'],
                'topic_count': len(cluster['topics']),
                'total_occurrences': cluster['total_count'],
                'top_topics': ', '.join([t['topic'] for t in cluster['topics'][:5]])
            })

        df_large_summary = pd.DataFrame(large_summary)
        large_summary_path = os.path.join(self.output_dir, 'data', 'large_clusters_summary.csv')
        df_large_summary.to_csv(large_summary_path, index=False, encoding='utf-8')

        small_summary = []
        for cluster_id, cluster in self.small_clusters.items():
            small_summary.append({
                'cluster_id': cluster_id,
                'cluster_name': cluster['name'],
                'representative_topic': cluster['representative'],
                'topic_count': len(cluster['topics']),
                'total_occurrences': cluster['total_count'],
                'all_topics': ', '.join([t['topic'] for t in cluster['topics']])
            })

        df_small_summary = pd.DataFrame(small_summary)
        small_summary_path = os.path.join(self.output_dir, 'data', 'small_clusters_summary.csv')
        df_small_summary.to_csv(small_summary_path, index=False, encoding='utf-8')

        # JSON出力
        results_json = {
            'analysis_date': datetime.now().isoformat(),
            'total_topics': len(self.topics),
            'large_clusters_count': len(self.large_clusters),
            'small_clusters_count': len(self.small_clusters),
            'large_clusters': {str(k): v for k, v in self.large_clusters.items()},
            'small_clusters': {str(k): v for k, v in self.small_clusters.items()}}
