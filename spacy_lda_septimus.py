import spacy
import os
import re
import math
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from gensim import corpora, models
from gensim.models import CoherenceModel
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Environment variables for temporary storage
os.environ["JOBLIB_TEMP_FOLDER"] = "C:/Temp"
os.environ["TEMP"] = "C:/Temp"
os.environ["TMP"] = "C:/Temp"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Preprocessing parameters
MIN_PARAGRAPH_WORDS = 100
MIN_SENTENCE_WORDS = 1
MIN_PMI = 1.0
MIN_BIGRAM_FREQ = 3
TOP_N_DOMAIN = 30

# Weak word detection parameters
min_doc_ratio = 0.1
min_tfidf = 0.01
min_word_length = 3
min_word_freq = 2

# LDA modeling parameters
NUM_TOPICS_RANGE = range(2, 11)
PASSES = 40
ITERATIONS = 500
RANDOM_STATE = 3037
ALPHA = 'asymmetric'
ETA = 'auto'

# Output directory
OUTPUT_BASE_DIR = "LDA_output/spacy_lda/septimus"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# Load spaCy model
MODEL_NAME = "en_core_web_sm"
nlp = spacy.load(MODEL_NAME, disable=['ner', 'parser'])
nlp.add_pipe('sentencizer')
nlp.max_length = max(nlp.max_length, 10_000_000)

# Parameter dictionary
params = {
    "MIN_PARAGRAPH_WORDS": MIN_PARAGRAPH_WORDS,
    "MIN_SENTENCE_WORDS": MIN_SENTENCE_WORDS,
    "MIN_PMI": MIN_PMI,
    "MIN_BIGRAM_FREQ": MIN_BIGRAM_FREQ,
    "TOP_N_DOMAIN": TOP_N_DOMAIN,
    "min_doc_ratio": min_doc_ratio,
    "min_tfidf": min_tfidf,
    "min_word_length": min_word_length,
    "min_word_freq": min_word_freq
}

# Text cleaning
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\'\"\-\—]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b\d+\b', '', text)
    return text.strip()

# Text preprocessing
def preprocess_text(input_file, params, output_dir):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    names_to_remove = [
        'clarissa','rezia','peter','sally','seton','septimus','dalloway',
        'richard','elizabeth','walsh','hugh','whitbread','bruton',
        'warren','smith','lucrezia','holmes','bradshaw','evelyn',
        'dempster','kilman','ellie','henderson','lucy','william',
        'peters','evans','geranium','blackberry'
    ]

    def detect_weak_words(doc_list):
        word_tf = Counter()
        doc_freq = Counter()
        total_tokens = 0
        for doc in doc_list:
            seen_in_doc = set()
            for token in doc:
                lemma = token.lemma_.lower()
                if (token.is_alpha and not token.is_stop
                    and len(lemma) >= params["min_word_length"]
                    and token.pos_ in ['NOUN','PROPN','ADJ','VERB']):
                    word_tf[lemma] += 1
                    total_tokens += 1
                    if lemma not in seen_in_doc:
                        doc_freq[lemma] += 1
                        seen_in_doc.add(lemma)
        total_docs = len(doc_list)
        tfidf_dict = {}
        for word in word_tf:
            tf = word_tf[word] / total_tokens if total_tokens > 0 else 0
            idf = math.log(total_docs / (doc_freq[word] + 1e-6)) if doc_freq[word] > 0 else 0
            tfidf_dict[word] = tf * idf
        weak_words = set()
        for word, count in word_tf.items():
            if word in names_to_remove:
                continue
            df_ratio = doc_freq[word] / total_docs
            tfidf_score = tfidf_dict.get(word, 0)
            if (tfidf_score < params["min_tfidf"] and df_ratio > params["min_doc_ratio"]) or count < params["min_word_freq"]:
                weak_words.add(word)
        return weak_words, tfidf_dict, word_tf

    def generate_domain_candidates(tfidf_dict, word_tf_dict):
        candidate_words = []
        for word, tfidf_score in tfidf_dict.items():
            if word_tf_dict.get(word,0) >= params["min_word_freq"] and len(word) >= params["min_word_length"]:
                score = tfidf_score * math.log(word_tf_dict[word]+1)
                candidate_words.append((word, score))
        candidate_words.sort(key=lambda x: x[1], reverse=True)
        return set([w for w,_ in candidate_words[:params["TOP_N_DOMAIN"]]])

    def get_high_pmi_phrases(texts_token_lists):
        bigram_counts = Counter()
        unigram_counts = Counter()
        total_tokens = 0
        for tokens in texts_token_lists:
            for i in range(len(tokens)):
                unigram_counts[tokens[i]] += 1
                total_tokens += 1
                if i < len(tokens)-1:
                    bigram_counts[(tokens[i], tokens[i+1])] += 1
        high_pmi_phrases = set()
        for (w1, w2), bigram_freq in bigram_counts.items():
            if bigram_freq < params["MIN_BIGRAM_FREQ"]:
                continue
            p_w1 = unigram_counts[w1] / total_tokens
            p_w2 = unigram_counts[w2] / total_tokens
            p_bigram = bigram_freq / total_tokens
            if p_bigram > 0 and p_w1 > 0 and p_w2 > 0:
                pmi = math.log2(p_bigram / (p_w1 * p_w2))
                if pmi >= params["MIN_PMI"]:
                    high_pmi_phrases.add(f"{w1}_{w2}")
        return high_pmi_phrases

    def process_sentence(sent_span, weak_words, high_pmi_phrases):
        processed_words = []
        tokens = [t.lemma_.lower() for t in sent_span 
                if t.is_alpha and not t.is_stop 
                and t.pos_ in ['NOUN','PROPN','ADJ','VERB']
                and t.lemma_.lower() not in weak_words]

        i = 0
        while i < len(tokens):
            if i < len(tokens)-1:
                bigram = f"{tokens[i]}_{tokens[i+1]}"
                if bigram in high_pmi_phrases:
                    processed_words.append(bigram)
                    i += 2
                    continue
            processed_words.append(tokens[i])
            i += 1

        if len(processed_words) >= params["MIN_SENTENCE_WORDS"]:
            return " ".join(processed_words)
        return ""

    def strict_merge_paragraphs(sentences):
        sentences = [s for s in sentences if s and s.strip()]
        paragraphs = []
        current_para = []
        current_word_count = 0
        for sent in sentences:
            sent_words = sent.split()
            if not sent_words: continue
            current_para.append(sent)
            current_word_count += len(sent_words)
            if current_word_count >= params["MIN_PARAGRAPH_WORDS"]:
                paragraphs.append(" ".join(current_para).strip())
                current_para = []
                current_word_count = 0
        if current_para:
            if paragraphs:
                paragraphs[-1] += " " + " ".join(current_para).strip()
            else:
                paragraphs.append(" ".join(current_para).strip())
        return [p for p in paragraphs if p.strip()]

    with open(input_file,'r',encoding='utf-8') as f:
        full_text = clean_text(f.read())

    doc = nlp(full_text)
    sentences = [s for s in doc.sents if len(s.text.split()) > 1]
    sent_docs = list(nlp.pipe([s.text for s in sentences]))
    weak_words, tfidf_dict, word_tf_dict = detect_weak_words(sent_docs)
    domain_candidates = generate_domain_candidates(tfidf_dict, word_tf_dict)
    token_lists = [[t.lemma_.lower() for t in s if t.is_alpha] for s in sent_docs]
    high_pmi_phrases = get_high_pmi_phrases(token_lists)
    processed_sentences = [process_sentence(s, weak_words, high_pmi_phrases) for s in sentences]
    processed_sentences = [s for s in processed_sentences if s]
    paragraphs = strict_merge_paragraphs(processed_sentences)

    para_lengths = [len(p.split()) for p in paragraphs]
    valid_paragraphs = [p for p in paragraphs if len(p.split()) >= params["MIN_SENTENCE_WORDS"]]

    text_stats = {
        "total_words": sum(para_lengths),
        "num_paragraphs": len(valid_paragraphs),
        "min_paragraph_length": min(para_lengths) if para_lengths else 0,
        "max_paragraph_length": max(para_lengths) if para_lengths else 0,
        "avg_paragraph_length": sum(para_lengths)/len(para_lengths) if para_lengths else 0,
        "original_paragraphs": len(paragraphs),
        "filtered_paragraphs": len(paragraphs) - len(valid_paragraphs),
        "weak_words_count": len(weak_words),
        "high_pmi_phrases_count": len(high_pmi_phrases)
    }

    processed_file = os.path.join(output_dir, f"{base_name}_processed.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(processed_file,'w',encoding='utf-8') as f:
        for para in paragraphs:
            f.write(para+"\n\n")   

    return processed_file, text_stats, weak_words, high_pmi_phrases

# Generating publication-ready topic tables
def save_topics_table(best_model, best_num_topics, output_dir, base_name, format="markdown"):
    table_ext = 'md' if format=='markdown' else 'tex'
    table_path = os.path.join(output_dir, f"{base_name}_{'Markdown' if format=='markdown' else 'LaTeX'}.{table_ext}")
    topics_data = []
    for idx in range(best_num_topics):
        terms = best_model.show_topic(idx, topn=10)
        topics_data.append([f"{w:.3f}*\"{t}\"" for t, w in terms])
    if format == "markdown":
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write("| " + " | ".join([f"Topic {i}" for i in range(best_num_topics)]) + " |\n")
            f.write("|" + "|".join(["---"]*best_num_topics) + "|\n")
            for i in range(10):
                row = []
                for topic_terms in topics_data:
                    row.append(topic_terms[i] if i < len(topic_terms) else "")
                f.write("| " + " | ".join(row) + " |\n")
    else:  
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write("\\begin{tabular}{%s}\n" % ("l"*best_num_topics))
            f.write(" & ".join([f"Topic {i}" for i in range(best_num_topics)]) + " \\\\\n")
            f.write("\\hline\n")
            for i in range(10):
                row = []
                for topic_terms in topics_data:
                    row.append(topic_terms[i] if i < len(topic_terms) else "")
                f.write(" & ".join(row) + " \\\\\n")
            f.write("\\end{tabular}\n")

# LDA topic modeling
def perform_lda(input_file):
    output_dir = OUTPUT_BASE_DIR
    base_name = os.path.splitext(os.path.basename(input_file))[0].replace("_spacy", "")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        documents = [line.strip().split() for line in f if line.strip()]
    if not documents:
        print("No documents to model.")
        return

    dictionary = corpora.Dictionary(documents)
    dictionary.filter_extremes(no_below=1, no_above=0.9)
    corpus = [dictionary.doc2bow(doc) for doc in documents if len(doc) > 0]

    all_metrics = []
    for num_topics in NUM_TOPICS_RANGE:
        lda_model = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=RANDOM_STATE,
            passes=PASSES,
            iterations=ITERATIONS,
            alpha=ALPHA,
            eta=ETA,
            per_word_topics=True,
            eval_every=0
        )
        perplexity = lda_model.log_perplexity(corpus)
        coherence_model = CoherenceModel(model=lda_model, texts=documents, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        all_metrics.append({'num_topics': num_topics, 'perplexity': perplexity, 'coherence': coherence, 'model': lda_model})
        print(f"num_topics={num_topics}: Perplexity={perplexity:.4f}, Coherence={coherence:.4f}")

    best_metric = max(all_metrics, key=lambda x: (x['coherence'], -x['perplexity']))
    best_model = best_metric['model']
    best_num_topics = best_metric['num_topics']
    print(f"\n=== Best model selected: num_topics={best_num_topics}, Perplexity={best_metric['perplexity']:.4f}, Coherence={best_metric['coherence']:.4f} ===\n")

    txt_path = os.path.join(output_dir, f"{base_name}_spacy_lda.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=== All topic models metrics ===\n")
        for m in all_metrics:
            f.write(f"num_topics={m['num_topics']}, Perplexity={m['perplexity']:.4f}, Coherence={m['coherence']:.4f}\n")
        f.write(f"\n=== Best model (num_topics={best_num_topics}) ===\n")
        f.write(f"Perplexity={best_metric['perplexity']:.4f}, Coherence={best_metric['coherence']:.4f}\n\n")
        f.write("--- Top 10 Words per Topic ---\n")
        for idx in range(best_num_topics):
            terms = best_model.show_topic(idx, topn=10)
            words_str = " + ".join([f"{w:.3f}*\"{t}\"" for t, w in terms])
            f.write(f"Topic: {idx}\nWords: {words_str}\n\n")
        f.write("--- Document Topic Distribution ---\n")
        for i, doc in enumerate(corpus):
            f.write(f"\nDocument {i+1} Topic Distribution:\n")
            f.write(str(best_model.get_document_topics(doc)) + "\n")

    seed_df = pd.DataFrame(all_metrics).sort_values(by='num_topics')
    plt.figure(figsize=(12,6))
    plt.rcParams['font.size'] = 12
    plt.subplot(1,2,1)
    plt.plot(seed_df['num_topics'], seed_df['perplexity'], marker='o', color='blue')
    plt.xticks(seed_df['num_topics'])
    plt.title('Perplexity vs Topics')
    plt.xlabel('Number of Topics')
    plt.ylabel('Log Perplexity')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.subplot(1,2,2)
    plt.plot(seed_df['num_topics'], seed_df['coherence'], marker='o', color='red')
    plt.xticks(seed_df['num_topics'])
    plt.title('Coherence vs Topics')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score (c_v)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{base_name}_lda_metrics.tif")
    plt.savefig(plot_path, dpi=600, format='tif', bbox_inches='tight')
    plt.close()

    save_topics_table(best_model, best_num_topics, output_dir, base_name, format="markdown")
    save_topics_table(best_model, best_num_topics, output_dir, base_name, format="latex")

    print(f"Analysis completed. Files saved in {output_dir}")

if __name__ == "__main__":
    INPUT_FILE = "data/septimus.txt"
    processed_file_path, text_stats, weak_words, high_pmi_phrases = preprocess_text(INPUT_FILE, params, OUTPUT_BASE_DIR)
    
    print(f"High PMI bigrams: {len(high_pmi_phrases)}, "
          f"Weak words: {len(weak_words)}, "
          f"Paragraphs: {text_stats['num_paragraphs']}, "
          f"Total words: {text_stats['total_words']}, "
          f"Paragraph length: min={text_stats['min_paragraph_length']}, "
          f"max={text_stats['max_paragraph_length']}, "
          f"avg={text_stats['avg_paragraph_length']:.1f}")
    
    perform_lda(processed_file_path)