import os
import pandas as pd
from tqdm import tqdm
import re
from openai import AzureOpenAI

endpoint = "https://mulabo-saki.openai.azure.com/"
model_name = "gpt-4.1-mini"
deployment = "gpt-4.1-mini"

subscription_key = "API_KEY"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# キャッシュの初期化
classification_cache = {}
sentiment_cache = {}
subjectivity_cache = {}
person_name_cache = {}


def preprocess_review(review):
    """レビューの前処理"""
    if not review or not isinstance(review, str):
        return ""
    # 余分な空白や改行を削除
    return ' '.join(review.strip().split())


def classify_and_analyze_batch(reviews, batch_size=50):
    """レビューをバッチで分類、センチメント分析、主観性分析、人名抽出する関数"""
    all_topic_results = []
    all_sentiment_results = []
    all_subjectivity_results = []
    all_person_name_results = []

    # バッチに分割して処理
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(reviews))))

        # キャッシュ済みの項目を確認
        uncached_reviews = []
        uncached_indices = []
        batch_topic_results = [None] * len(batch)
        batch_sentiment_results = [None] * len(batch)
        batch_subjectivity_results = [None] * len(batch)
        batch_person_name_results = [None] * len(batch)

        for j, (idx, review) in enumerate(zip(batch_indices, batch)):
            if (review in classification_cache and
                    review in sentiment_cache and
                    review in subjectivity_cache and
                    review in person_name_cache):
                batch_topic_results[j] = classification_cache[review]
                batch_sentiment_results[j] = sentiment_cache[review]
                batch_subjectivity_results[j] = subjectivity_cache[review]
                batch_person_name_results[j] = person_name_cache[review]
            else:
                uncached_reviews.append(review)
                uncached_indices.append(j)

        # キャッシュにないレビューだけを処理
        if uncached_reviews:
            new_topic_results, new_sentiment_results, new_subjectivity_results, new_person_name_results = process_uncached_batch_combined(
                uncached_reviews)

            # 結果をキャッシュに追加し、バッチ結果に統合
            for j, (review, topic_result, sent_result, subj_result, person_result) in enumerate(
                    zip(uncached_reviews, new_topic_results, new_sentiment_results, new_subjectivity_results,
                        new_person_name_results)):
                classification_cache[review] = topic_result
                sentiment_cache[review] = sent_result
                subjectivity_cache[review] = subj_result
                person_name_cache[review] = person_result
                batch_topic_results[uncached_indices[j]] = topic_result
                batch_sentiment_results[uncached_indices[j]] = sent_result
                batch_subjectivity_results[uncached_indices[j]] = subj_result
                batch_person_name_results[uncached_indices[j]] = person_result

        all_topic_results.extend(batch_topic_results)
        all_sentiment_results.extend(batch_sentiment_results)
        all_subjectivity_results.extend(batch_subjectivity_results)
        all_person_name_results.extend(batch_person_name_results)

    return all_topic_results, all_sentiment_results, all_subjectivity_results, all_person_name_results


def process_uncached_batch_combined(reviews):
    """キャッシュにないレビューのバッチを処理（トピック、センチメント分析、主観性分析、人名抽出）"""
    # バッチ用プロンプトの作成
    batch_content = "\n".join([f"{j + 1}: {review}" for j, review in enumerate(reviews)])

    prompt_intro = (
        "Classify each of the following movie review sentences.\n"
        "For each sentence, provide the following 4 fields:\n"
        "- Topic: Choose exactly one topic label from the predefined list of 19 categories.\n"
        "  This represents the main aspect of the movie the sentence is about (e.g., acting, plot, music, etc.).\n"
        "  Choose the most content-specific label. Do not use vague terms or invent new labels.\n"
        "  Use emotion-related labels (viewing_experience, emotion, expectation) only if no other specific label fits.\n"
        "- Sentiment: A number from 1 (Very Negative) to 5 (Very Positive).\n"
        "- Subjectivity: 'S' for Subjective (opinion-based), 'O' for Objective (fact-based).\n"
        "- Person Name: Any person named (actor, director, composer, etc.), or 'None'.\n"
        "\n"
        "Return results in the following strict format:\n"
        "<number>: <Topic>|<Sentiment>|<Subjectivity>|<Person Name>\n"

    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an annotation system for sentence-level movie review analysis.\n"
                "For each sentence, provide a concise annotation including:\n\n"
                "TOPIC:\n"
                            
                        
"Your task is to assign exactly ONE topic label to each review sentence from the predefined 19 labels below.\n"
"Always return only one label per sentence.\n"
"Never create new labels.\n"
"If uncertain, choose the most relevant label based on content focus.\n"
"\n"
"---\n"
"\n"
"Label list and definitions:\n"
"\n"
"story_plot: Mentions of events, twists, narrative structure, or endings.\n"
"  Focus: What happens in the story.\n"
"  Examples: plot, twist, ending, storyline, narrative.\n"
"\n"
"character_development: Mentions of character traits, personalities, or growth.\n"
"  Focus: Who the characters are and how they change.\n"
"  Examples: character, protagonist, personality, arc, role.\n"
"\n"
"themes_messages: Mentions of themes, messages, morals, or symbolism.\n"
"  Focus: Deeper meanings and takeaways.\n"
"  Examples: theme, meaning, symbolism, message, philosophy.\n"
"\n"
"acting_performance: Mentions of how well actors acted.\n"
"  Focus: Acting skill, delivery, realism.\n"
"  Examples: acting, performance, delivery, actor name + impression.\n"
"\n"
"casting_choices: Mentions of whether the actor was appropriate for the role.\n"
"  Focus: Casting fit or misfit.\n"
"  Examples: casting, miscast, right choice, wrong choice.\n"
"\n"
"filmmaking_direction: Mentions of the director's creative influence or overall vision.\n"
"  Focus: Directing style or leadership.\n"
"  Examples: director, directing, filmmaker, vision.\n"
"\n"
"writing_dialogue: Mentions of script quality, dialogue, or screenplay.\n"
"  Focus: What is written or spoken.\n"
"  Examples: script, dialogue, lines, screenplay.\n"
"\n"
"technical_visuals: Mentions of camera work, CGI, VFX, or cinematography.\n"
"  Focus: Visual techniques and technologies.\n"
"  Examples: camera, CGI, VFX, lighting, cinematography.\n"
"\n"
"artistic_design: Mentions of visual art, costumes, sets, or makeup.\n"
"  Focus: Designed appearance or aesthetic.\n"
"  Examples: costume, set, design, makeup, visual style.\n"
"\n"
"audio_music: Mentions of music, score, sound effects, or audio design.\n"
"  Focus: What is heard.\n"
"  Examples: soundtrack, sound, music, audio.\n"
"\n"
"editing_pacing: Mentions of editing speed, rhythm, or scene transitions.\n"
"  Focus: Flow and timing.\n"
"  Examples: editing, pace, slow, fast, transition.\n"
"\n"
"viewing_experience: Mentions of entertainment value, enjoyment, or boredom.\n"
"  Focus: Overall watching experience, only if no other more specific topic applies.\n"
"  Examples: entertaining, boring, fun, immersive, dull.\n"
"\n"
"emotion: Mentions of personal emotional reaction to the film.\n"
"  Focus: Specific feelings, only if no other more specific topic applies.\n"
"  Examples: cried, laughed, scared, moved, thrilled.\n"
"\n"
"genre_style: Mentions of the movie genre or style.\n"
"  Focus: Type of movie.\n"
"  Examples: genre, thriller, comedy, action, drama, sci-fi.\n"
"\n"
"comparative_analysis: Mentions of comparisons to other works.\n"
"  Focus: Similarity or quality compared to other films.\n"
"  Examples: better than, similar to, like X, compared to Y.\n"
"\n"
"recommendation: Mentions of whether others should watch or avoid the film.\n"
"  Focus: Advice or recommendations.\n"
"  Examples: recommend, worth watching, skip, must-watch, avoid.\n"
"\n"
"expectation: Mentions of expectations before watching and how they were met.\n"
"  Focus: Expectations vs reality, only if no other more specific topic applies.\n"
"  Examples: expected, let down, exceeded expectations, surprised.\n"
"\n"
"commercial_context: Mentions of box office, franchise, sequels, or marketing.\n"
"  Focus: Commercial or business context.\n"
"  Examples: box office, success, flop, sequel, franchise, promoted.\n"
"\n"
"---\n"
"\n"
"Important distinctions:\n"
"- story_plot vs character_development:\n"
"  Plot = what happens. Character = who changes.\n"
"  Examples: \"The story twist surprised me\" → story_plot, \"He was a relatable hero\" → character_development\n"
"\n"
"- acting_performance vs casting_choices:\n"
"  Acting = how well they performed. Casting = whether they fit the role.\n"
"  Examples: \"She gave an amazing performance\" → acting_performance, \"He was miscast for the role\" → casting_choices\n"
"\n"
"- technical_visuals vs artistic_design:\n"
"  Visuals = camera or CGI. Design = costumes or sets.\n"
"  Examples: \"The CGI was impressive\" → technical_visuals, \"The costumes were beautiful\" → artistic_design\n"
"\n"
"- emotion vs viewing_experience:\n"
"  Emotion = specific feelings. Experience = overall enjoyment.\n"
"  Examples: \"It made me cry\" → emotion, \"I had a great time watching it\" → viewing_experience\n"
"\n"
"- story_plot vs themes_messages:\n"
"  Plot = the events. Themes = deeper meaning behind those events.\n"
"  Examples: \"The mystery kept me hooked\" → story_plot, \"It explores the meaning of justice\" → themes_messages\n"
"\n"
"- audio_music vs emotion:\n"
"  Audio = music or sound. Emotion = how it made the viewer feel.\n"
"  Examples: \"The soundtrack was great\" → audio_music, \"It felt haunting and lonely\" → emotion\n"
"\n"
"- editing_pacing vs story_plot:\n"
"  Editing = flow and pacing. Plot = story content.\n"
"  Examples: \"It dragged in the middle\" → editing_pacing, \"A murder occurs halfway through\" → story_plot\n"
"\n"
"- viewing_experience vs recommendation:\n"
"  Experience = personal reaction while watching. Recommendation = advice to others.\n"
"  Examples: \"I enjoyed it a lot\" → viewing_experience, \"You should definitely watch it\" → recommendation\n"
"\n"
"- expectation vs viewing_experience:\n"
"  Expectation = before watching. Experience = during or after watching.\n"
"  Examples: \"I thought it would be better\" → expectation, \"It was fun despite flaws\" → viewing_experience\n"
"\n"
"- commercial_context vs recommendation:\n"
"  Commercial = business/success. Recommendation = personal advice.\n"
"  Examples: \"It was a box office hit\" → commercial_context, \"I recommend it to anyone\" → recommendation\n"
"\n"
"- comparative_analysis vs expectation:\n"
"  Comparison = with other films. Expectation = based on personal hopes.\n"
"  Examples: \"It’s better than the first movie\" → comparative_analysis, \"I expected more action\" → expectation\n"
"\n"
"Note:\n"
"Labels related to emotions (viewing_experience, emotion, expectation) should be assigned only when no other more specific label applies.\n"
"For example, sentences like \"The acting was terrible\" or \"It was too boring\" should be categorized under the more specific topics (e.g., acting_performance) rather than simply viewing_experience.\n"
"\n"
"Output format:\n"
"Return only the label name from the list above (e.g., \"story_plot\").\n"
"Do not output any explanation or extra text.\n"

                

                
                "SENTIMENT:\n"
                "- 5: Very Positive (e.g., excellent, love, amazing)\n"
                "- 4: Positive (e.g., good, solid, enjoyable)\n"
                "- 3: Neutral (e.g., factual description)\n"
                "- 2: Negative (e.g., bad, weak, disappointing)\n"
                "- 1: Very Negative (e.g., terrible, worst, hate)\n"
                "- Only adjust from 3 if clear positive/negative expressions are present.\n"
                "NEUTRAL PATTERNS (keep as 3):\n"
                "- Conditional statements: 'if you want', 'when you need'\n"
                "- Pure facts without judgment\n\n"
    
    
                "SUBJECTIVITY:\n"
                "'S' for Subjective: Contains evaluations, judgments, or emotional expressions\n"
                "'O' for Objective: Describes facts, events, or content without evaluation\n"

                "- Ask: Is the sentence saying how something **is**, or how **good/bad** it is?\n\n"
                "PERSON NAME:\n"
                "- If a person's name (actor, director, composer, etc.) appears, extract it as-is.\n"
                "- If multiple names, separate with commas.\n"
                "- If no name is present, write 'None'.\n\n"
                "OUTPUT FORMAT - CRITICAL:\n"
                "1: score|4|S|Hans Zimmer\n"
                "2: plot|2|S|None\n"
                "3: cast|3|O|Tom Hanks, Meryl Streep\n"
                "4: direction|5|S|Christopher Nolan\n\n"
                "Format: NUMBER: TOPIC|SENTIMENT|SUBJECTIVITY|PERSON_NAME\n"
                "Use only this structure. Do NOT include explanations, examples, or extra text."
            )
        },
        {
            "role": "user",
            "content": prompt_intro + batch_content
        }
    ]

    # デバッグ: 送信するプロンプトを確認
    print(f"\n=== デバッグ: APIに送信するプロンプト ===")
    print(f"バッチサイズ: {len(reviews)}")
    print(f"最初のレビュー: {reviews[0][:100] if reviews else 'なし'}...")
    print("=" * 50)

    try:
        response = client.chat.completions.create(
            messages=messages,
            max_completion_tokens=200,
            temperature=0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment,
        )

        # デバッグ: APIからの応答を確認
        result_text = response.choices[0].message.content.strip()
        print(f"\n=== デバッグ: APIからの応答 ===")
        print(f"応答内容: {result_text}")
        print("=" * 50)

        # レスポンスをパース
        batch_topic_results, batch_sentiment_results, batch_subjectivity_results, batch_person_name_results = parse_combined_batch_results(
            result_text, len(reviews))

        # デバッグ: パース結果を確認
        print(f"\n=== デバッグ: パース結果 ===")
        print(f"トピック結果: {batch_topic_results}")
        print(f"センチメント結果: {batch_sentiment_results}")
        print(f"主観性結果: {batch_subjectivity_results}")
        print(f"人名結果: {batch_person_name_results}")
        print("=" * 50)

        return batch_topic_results, batch_sentiment_results, batch_subjectivity_results, batch_person_name_results

    except Exception as e:
        print(f"バッチ処理エラー: {e}")
        # エラー時は全てERRORとして扱う
        return ["ERROR"] * len(reviews), [3] * len(reviews), ["ERROR"] * len(reviews), ["None"] * len(reviews)


def parse_combined_batch_results(result_text, expected_count):
    """バッチ結果をパースしてトピック、センチメント、主観性、人名結果のリストを返す"""
    topic_results = ["ERROR"] * expected_count
    sentiment_results = [3] * expected_count  # デフォルトは中立
    subjectivity_results = ["ERROR"] * expected_count
    person_name_results = ["None"] * expected_count

    print(f"\n=== パース処理デバッグ ===")
    print(f"受信テキスト: '{result_text}'")
    print(f"期待件数: {expected_count}")

    # 先頭の余分なテキストを除去
    lines = result_text.strip().split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        # 番号:トピック|センチメント|主観性|人名 の形式の行のみ抽出
        if re.match(r'^\d+\s*:', line) and line.count('|') >= 3:
            clean_lines.append(line)

    # クリーンアップされたテキストを再構成
    clean_text = '\n'.join(clean_lines)
    print(f"クリーンアップ後: '{clean_text}'")

    # 新しいパターンで抽出（トピック|センチメント|主観性|人名）
    pattern = r'(\d+)\s*:\s*(\*?[a-zA-Z_\s]+)\s*\|\s*([1-5])\s*\|\s*([SO])\s*\|\s*([^|\n]+)'
    matches = re.findall(pattern, clean_text, re.IGNORECASE)
    print(f"マッチ結果: {matches}")

    for match in matches:
        try:
            index = int(match[0]) - 1
            if 0 <= index < expected_count:
                topic = match[1].strip().lower()
                sentiment_score = int(match[2])
                subjectivity = match[3].upper()
                person_name = match[4].strip()

                # トピックの設定
                if topic:
                    topic_results[index] = topic
                    print(f"インデックス {index} にトピック '{topic}' を設定")

                # センチメントスコアの検証
                if 1 <= sentiment_score <= 5:
                    sentiment_results[index] = sentiment_score
                    print(f"インデックス {index} にセンチメント '{sentiment_score}' を設定")

                # 主観性の検証
                if subjectivity in ['S', 'O']:
                    subjectivity_results[index] = subjectivity
                    print(f"インデックス {index} に主観性 '{subjectivity}' を設定")

                # 人名の設定
                if person_name and person_name.lower() != 'none':
                    person_name_results[index] = person_name
                    print(f"インデックス {index} に人名 '{person_name}' を設定")
                else:
                    person_name_results[index] = "None"
                    print(f"インデックス {index} に人名 'None' を設定")

        except (ValueError, IndexError) as e:
            print(f"パース エラー: {e}")
            continue

    print(f"最終トピック結果: {topic_results}")
    print(f"最終センチメント結果: {sentiment_results}")
    print(f"最終主観性結果: {subjectivity_results}")
    print(f"最終人名結果: {person_name_results}")
    print("=" * 50)
    return topic_results, sentiment_results, subjectivity_results, person_name_results


def classify_movie_reviews(input_path, output_path):
    """メイン処理関数"""
    print(f"入力ファイル '{input_path}' を読み込み中...")
    # 複数のエンコーディングを試して読み込み
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

    for encoding in encodings:
        try:
            df = pd.read_csv(input_path, encoding=encoding)
            if encoding != 'utf-8':
                print(f"ファイルを{encoding}エンコーディングで読み込みました")
            break
        except UnicodeDecodeError:
            continue
    else:
        # 全てのエンコーディングで失敗した場合、エラーを無視して読み込み
        print("標準エンコーディングで読み込めませんでした。エラーを無視して読み込みます...")
        df = pd.read_csv(input_path, encoding='utf-8', errors='ignore')

    # 11件目から20件目を処理（デバッグ用）
    original_count = len(df)
    df = df.iloc[1:1000]  # 10番目のインデックス（11件目）から20番目のインデックス（20件目）まで
    print(f"全{original_count}件中、11件目から20件目（19件）を処理します（デバッグモード）")

    # レビューの前処理
    processed_reviews = [preprocess_review(review) for review in df["review"]]

    # デバッグ: 実際のレビュー内容を確認
    print("\n=== デバッグ: 処理するレビューの最初の3件 ===")
    for i, review in enumerate(processed_reviews[:3]):
        print(f"{i + 1}: {review[:100]}..." if len(review) > 100 else f"{i + 1}: {review}")
    print("=" * 50)

    print(f"全{len(processed_reviews)}件のレビューを分析中...")
    # バッチサイズ10で処理（nano モデル用に調整）
    topic_results, sentiment_results, subjectivity_results, person_name_results = classify_and_analyze_batch(
        processed_reviews, batch_size=10)

    # 分析結果を追加
    df["topic"] = topic_results
    df["sentiment_score"] = sentiment_results
    df["subjectivity"] = subjectivity_results
    df["person_name"] = person_name_results

    # 結果を保存
    df.to_csv(output_path, index=False)
    print(f"分析結果を '{output_path}' に保存しました")

    # 統計情報表示
    print("\n=== トピック分析結果の統計 ===")
    topic_counts = pd.Series(topic_results).value_counts()
    print(f"出現したトピック数: {len(topic_counts)}種類")
    print("全トピックのランキング:")
    for rank, (topic_name, count) in enumerate(topic_counts.items(), 1):
        print(f"{rank:2d}. {topic_name}: {count}件 ({count / len(topic_results) * 100:.1f}%)")


    print("\n=== センチメント分析結果の統計 ===")
    sentiment_counts = pd.Series(sentiment_results).value_counts().sort_index()
    sentiment_labels = {
        1: "非常にネガティブ",
        2: "ネガティブ",
        3: "中立",
        4: "ポジティブ",
        5: "非常にポジティブ"
    }
    for score, count in sentiment_counts.items():
        label = sentiment_labels.get(score, f"スコア{score}")
        print(f"- {score} ({label}): {count}件 ({count / len(sentiment_results) * 100:.1f}%)")

    print("\n=== 主観性分析結果の統計 ===")
    subjectivity_counts = pd.Series(subjectivity_results).value_counts()
    subjectivity_labels = {
        'S': "主観的",
        'O': "客観的"
    }
    for subj, count in subjectivity_counts.items():
        label = subjectivity_labels.get(subj, f"分類{subj}")
        print(f"- {subj} ({label}): {count}件 ({count / len(subjectivity_results) * 100:.1f}%)")

    print("\n=== 人名抽出結果の統計 ===")
    person_name_counts = pd.Series(person_name_results).value_counts()
    print(f"- 人名が検出された件数: {len([p for p in person_name_results if p != 'None'])}件")
    print(f"- 人名が検出されなかった件数: {len([p for p in person_name_results if p == 'None'])}件")
    print("- 検出された人名（上位5件）:")
    for name, count in person_name_counts.head(5).items():
        if name != 'None':
            print(f"  * {name}: {count}件")

    # 詳細な分析結果表示
    print("\n=== 分析結果の詳細 ===")
    for i, (review, topic_result, sent_result, subj_result, person_result) in enumerate(
            zip(processed_reviews[:10], topic_results[:10], sentiment_results[:10], subjectivity_results[:10],
                person_name_results[:10])):
        sent_label = sentiment_labels.get(sent_result, f"スコア{sent_result}")
        subj_label = subjectivity_labels.get(subj_result, f"分類{subj_result}")
        display_review = review[:60] + "..." if len(review) > 60 else review
        print(
            f"{i + 1}. [{topic_result}|{sent_result}({sent_label})|{subj_result}({subj_label})|{person_result}] {display_review}")

    # キャッシュ統計
    print(f"\nキャッシュ統計:")
    print(f"- トピックキャッシュ: {len(classification_cache)}件")
    print(f"- センチメントキャッシュ: {len(sentiment_cache)}件")
    print(f"- 主観性キャッシュ: {len(subjectivity_cache)}件")
    print(f"- 人名キャッシュ: {len(person_name_cache)}件")


if __name__ == "__main__":
    input_file = "/Users/watanabesaki/PycharmProjects/sotsuron/review_per_movie/segmented/segmented_Final Destination 2000.csv"  # 入力ファイル名
    output_file = "/Users/watanabesaki/PycharmProjects/sotsuron/review_per_movie/topic/Final Destination 2000.csv"  # 出力ファイル名

    classify_movie_reviews(input_file, output_file)