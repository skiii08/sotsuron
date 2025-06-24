import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao',
                               'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


class TopicAnalyzer:
    def __init__(self, file1_path, file2_path,
                 output_dir="/Users/watanabesaki/PycharmProjects/sotsuron/data/analyze/topic"):
        self.file1_path = file1_path
        self.file2_path = file2_path
        self.output_dir = output_dir
        self.combined_data = None
        self.topic_counts = None

        self.setup_output_directory()
        self.load_and_combine_data()

    def setup_output_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)
        subdirs = ['plots', 'interactive', 'data', 'reports']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        print(f"出力ディレクトリを準備しました: {self.output_dir}")

    def load_and_combine_data(self):
        print("データを読み込み中...")
        df1 = pd.read_csv(self.file1_path)
        df2 = pd.read_csv(self.file2_path)
        self.combined_data = pd.concat([df1, df2], ignore_index=True)
        self.topic_counts = Counter(self.combined_data['topic'].dropna())

        combined_output_path = os.path.join(self.output_dir, 'data', 'combined_topic_data.csv')
        self.combined_data.to_csv(combined_output_path, index=False, encoding='utf-8')

        topic_stats = {
            'total_records': len(self.combined_data),
            'unique_topics': len(self.topic_counts),
            'topic_counts': dict(self.topic_counts.most_common()),
            'analysis_date': datetime.now().isoformat()
        }
        stats_output_path = os.path.join(self.output_dir, 'data', 'topic_statistics.json')
        with open(stats_output_path, 'w', encoding='utf-8') as f:
            json.dump(topic_stats, f, ensure_ascii=False, indent=2)

        print(f"総データ件数: {len(self.combined_data)}")
        print(f"ユニークなトピック数: {len(self.topic_counts)}")

    def get_top_topics(self, n=20):
        return dict(self.topic_counts.most_common(n))

    def plot_top_topics_bar(self, n=20):
        top_topics = self.get_top_topics(n)
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(top_topics)), list(top_topics.values()),
                       color=plt.cm.viridis(np.linspace(0, 1, len(top_topics))))

        plt.xlabel('トピック', fontsize=12)
        plt.ylabel('出現回数', fontsize=12)
        plt.title(f'上位{n}トピックの出現頻度', fontsize=14, fontweight='bold')
        plt.xticks(range(len(top_topics)), list(top_topics.keys()), rotation=45, ha='right')

        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{int(height)}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.grid(True, alpha=0.3)

        output_path = os.path.join(self.output_dir, 'plots', f'top_{n}_topics_bar.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"棒グラフを保存: {output_path}")

    def plot_topic_distribution_pie(self, n=10):
        top_topics = self.get_top_topics(n)
        others_count = sum(self.topic_counts.values()) - sum(top_topics.values())

        labels = list(top_topics.keys()) + ['その他']
        sizes = list(top_topics.values()) + [others_count]
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

        plt.figure(figsize=(10, 8))
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        plt.title(f'トピック分布（上位{n}個 + その他）', fontsize=14, fontweight='bold')
        plt.axis('equal')

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'plots', f'topic_distribution_pie_top_{n}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"円グラフを保存: {output_path}")

    def plot_cumulative_coverage(self, n=50):
        sorted_counts = sorted(self.topic_counts.values(), reverse=True)
        total_count = sum(sorted_counts)
        cumulative_percentages = [sum(sorted_counts[:i + 1]) / total_count * 100 for i in range(len(sorted_counts))]

        plt.figure(figsize=(12, 6))
        plt.plot(range(1, min(n + 1, len(cumulative_percentages) + 1)),
                 cumulative_percentages[:n], 'b-', linewidth=2, marker='o', markersize=4)

        plt.xlabel('トピック数（頻度順）', fontsize=12)
        plt.ylabel('累積カバレッジ（%）', fontsize=12)
        plt.title('トピックの累積カバレッジ', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80%')
        plt.axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='90%')
        plt.legend()

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'plots', f'cumulative_coverage_top_{n}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"累積カバレッジグラフを保存: {output_path}")

    def plot_interactive_treemap(self, n=30):
        top_topics = self.get_top_topics(n)

        fig = go.Figure(go.Treemap(
            labels=list(top_topics.keys()),
            values=list(top_topics.values()),
            parents=[""] * len(top_topics),
            textinfo="label+value+percent parent",
            hovertemplate='<b>%{label}</b><br>出現回数: %{value}<br>割合: %{percentParent}<extra></extra>',
        ))

        fig.update_layout(
            title=f"上位{n}トピックのツリーマップ",
            title_x=0.5,
            font_size=12,
            width=1000,
            height=600
        )

        output_path = os.path.join(self.output_dir, 'interactive', f'treemap_top_{n}.html')
        fig.write_html(output_path)
        print(f"インタラクティブツリーマップを保存: {output_path}")

    def plot_topic_frequency_distribution(self):
        frequencies = list(self.topic_counts.values())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.hist(frequencies, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('出現回数', fontsize=12)
        ax1.set_ylabel('トピック数', fontsize=12)
        ax1.set_title('トピック出現頻度の分布', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        ax2.hist(frequencies, bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('出現回数', fontsize=12)
        ax2.set_ylabel('トピック数', fontsize=12)
        ax2.set_title('トピック出現頻度の分布（対数スケール）', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'plots', 'frequency_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"頻度分布を保存: {output_path}")

    def plot_pareto_chart(self, n=20):
        top_topics = self.get_top_topics(n)
        total_count = sum(self.topic_counts.values())

        values = list(top_topics.values())
        cumulative_percentages = []
        cumsum = 0
        for value in values:
            cumsum += value
            cumulative_percentages.append(cumsum / total_count * 100)

        fig, ax1 = plt.subplots(figsize=(14, 8))

        bars = ax1.bar(range(len(top_topics)), values, color='steelblue', alpha=0.7)
        ax1.set_xlabel('トピック', fontsize=12)
        ax1.set_ylabel('出現回数', fontsize=12, color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')

        ax2 = ax1.twinx()
        line = ax2.plot(range(len(top_topics)), cumulative_percentages,
                        color='red', marker='o', linewidth=2, markersize=6)
        ax2.set_ylabel('累積パーセンテージ (%)', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 100)

        ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.7)
        ax2.text(len(top_topics) * 0.7, 82, '80%ライン', color='orange', fontweight='bold')

        plt.title(f'上位{n}トピックのパレート図', fontsize=14, fontweight='bold')
        plt.xticks(range(len(top_topics)), list(top_topics.keys()), rotation=45, ha='right')

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{int(height)}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'plots', f'pareto_chart_top_{n}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"パレート図を保存: {output_path}")

    def create_summary_dashboard(self):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('上位15トピック', 'トピック分布', '累積カバレッジ', '頻度分布'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )

        top_15 = self.get_top_topics(15)
        fig.add_trace(
            go.Bar(x=list(top_15.keys()), y=list(top_15.values()),
                   name="出現回数", showlegend=False),
            row=1, col=1
        )

        top_10 = self.get_top_topics(10)
        others_count = sum(self.topic_counts.values()) - sum(top_10.values())
        fig.add_trace(
            go.Pie(labels=list(top_10.keys()) + ['その他'],
                   values=list(top_10.values()) + [others_count],
                   showlegend=False),
            row=1, col=2
        )

        sorted_counts = sorted(self.topic_counts.values(), reverse=True)
        total_count = sum(sorted_counts)
        cumulative_percentages = [sum(sorted_counts[:i + 1]) / total_count * 100 for i in
                                  range(min(50, len(sorted_counts)))]
        fig.add_trace(
            go.Scatter(x=list(range(1, len(cumulative_percentages) + 1)),
                       y=cumulative_percentages,
                       mode='lines+markers', name="累積%", showlegend=False),
            row=2, col=1
        )

        frequencies = list(self.topic_counts.values())
        fig.add_trace(
            go.Histogram(x=frequencies, nbinsx=30, name="頻度分布", showlegend=False),
            row=2, col=2
        )

        fig.update_layout(
            title_text="トピック分析ダッシュボード",
            title_x=0.5,
            height=800,
            showlegend=False
        )

        fig.update_xaxes(tickangle=45, row=1, col=1)

        output_path = os.path.join(self.output_dir, 'interactive', 'summary_dashboard.html')
        fig.write_html(output_path)
        print(f"サマリーダッシュボードを保存: {output_path}")

    def export_analysis_report(self):
        total_topics = len(self.topic_counts)
        total_count = sum(self.topic_counts.values())
        top_10 = self.get_top_topics(10)
        top_40_percent = dict(list(self.topic_counts.most_common(int(total_topics * 0.4))))

        report_path = os.path.join(self.output_dir, 'reports', 'topic_analysis_report.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== トピック分析レポート ===\n")
            f.write(f"分析日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"出力ディレクトリ: {self.output_dir}\n\n")

            f.write(f"総データ件数: {total_count:,}\n")
            f.write(f"ユニークなトピック数: {total_topics:,}\n\n")

            f.write("=== 上位10トピック ===\n")
            for i, (topic, count) in enumerate(top_10.items(), 1):
                percentage = count / total_count * 100
                f.write(f"{i:2d}. {topic}: {count:,}回 ({percentage:.1f}%)\n")

            f.write(f"\n=== 主要トピック（上位40%）統計 ===\n")
            top_40_count = sum(top_40_percent.values())
            f.write(f"主要トピック数: {len(top_40_percent)}\n")
            f.write(f"カバレッジ: {top_40_count / total_count * 100:.1f}%\n")

            f.write(f"\n=== ロングテール統計 ===\n")
            single_occurrence = sum(1 for count in self.topic_counts.values() if count == 1)
            f.write(f"1回のみ出現するトピック: {single_occurrence}個 ({single_occurrence / total_topics * 100:.1f}%)\n")

        print(f"分析レポートを保存: {report_path}")

        df_topics = pd.DataFrame([
            {'rank': i + 1, 'topic': topic, 'count': count, 'percentage': count / total_count * 100}
            for i, (topic, count) in enumerate(self.topic_counts.most_common())
        ])
        csv_path = os.path.join(self.output_dir, 'data', 'topic_ranking.csv')
        df_topics.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"トピックランキングCSVを保存: {csv_path}")

    def run_all_analysis(self):
        print("=== 全分析を開始します ===")

        self.plot_top_topics_bar(n=20)
        self.plot_topic_distribution_pie(n=10)
        self.plot_cumulative_coverage(n=50)
        self.plot_pareto_chart(n=20)
        self.plot_topic_frequency_distribution()
        self.plot_interactive_treemap(n=30)
        self.create_summary_dashboard()
        self.export_analysis_report()

        print(f"\n=== すべての分析が完了しました！ ===")
        print(f"結果は以下のディレクトリに保存されました: {self.output_dir}")


# 実行部分
if __name__ == "__main__":
    file1_path = "/data/trial_data/topic_1_10000.csv"
    file2_path = "/data/trial_data/topic_1_10000.csv"

    analyzer = TopicAnalyzer(file1_path, file2_path)
    analyzer.run_all_analysis()