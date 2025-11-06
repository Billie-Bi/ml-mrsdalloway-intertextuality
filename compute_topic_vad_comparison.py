import os
import re
import ast
import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

# Setting global plotting parameters
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Defining color scheme
COLORS = {
    'Clarissa': '#d62728',  # red
    'Septimus': '#1f77b4'   # blue
}

# Specifying input/output paths
clarissa_txt = "data/clarissa_processed.txt"
septimus_txt = "data/septimus_processed.txt"
clarissa_doc_topic_txt = "data/clarissa_document_topic_distribution.txt"
septimus_doc_topic_txt = "data/septimus_document_topic_distribution.txt"
topic_alignment_csv = "data/cross_character_topic_clusters.csv"
nrc_vad_path = "data/NRC-VAD-Lexicon-v2.1.txt"

output_folder = "VAD_output/vad_topic_comparison_copy"
os.makedirs(output_folder, exist_ok=True)

# Checking file existence
for file_path in [clarissa_txt, septimus_txt, clarissa_doc_topic_txt, septimus_doc_topic_txt, topic_alignment_csv, nrc_vad_path]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found: {file_path}")

# Loading spaCy model
nlp = spacy.load("en_core_web_sm")

# Parsing document-topic distributions
def parse_doc_topic_all_probs(txt_file):
    all_docs = []
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('['):
            matches = re.findall(r'\((\d+),\s*([0-9\.]+)\)', line)
            if matches:
                all_docs.append([(int(t), float(p)) for t, p in matches])
    return all_docs

clarissa_doc_topics_all = parse_doc_topic_all_probs(clarissa_doc_topic_txt)
septimus_doc_topics_all = parse_doc_topic_all_probs(septimus_doc_topic_txt)

# Reading text paragraphs
with open(clarissa_txt, 'r', encoding='utf-8') as f:
    clarissa_paras = [p.strip() for p in f.read().split('\n\n') if p.strip()]
with open(septimus_txt, 'r', encoding='utf-8') as f:
    septimus_paras = [p.strip() for p in f.read().split('\n\n') if p.strip()]

# Reading topic alignment table
alignment_df = pd.read_csv(topic_alignment_csv)
alignment_df['Clarissa_Topics'] = alignment_df['Clarissa_Topics'].apply(lambda x: ast.literal_eval(x))
alignment_df['Septimus_Topics'] = alignment_df['Septimus_Topics'].apply(lambda x: ast.literal_eval(x))

# Loading NRC-VAD lexicon
vad_dict = {}
with open(nrc_vad_path, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        try:
            word = parts[0].lower()
            V, A, D = map(float, parts[1:4])
            vad_dict[word] = (V, A, D)
        except:
            continue

# Computing paragraph VAD scores and coverage
def compute_vad(text):
    words = text.split()
    tokens = []
    i = 0
    while i < len(words):
        matched = False
        for n in [3, 2, 1]:
            if i + n <= len(words):
                phrase = ' '.join(words[i:i+n]).lower()
                if phrase in vad_dict:
                    tokens.append(phrase)
                    i += n
                    matched = True
                    break
        if not matched:
            doc = nlp(words[i])
            tokens.extend([t.lemma_.lower() for t in doc if t.is_alpha])
            i += 1
    vals = [vad_dict[t] for t in tokens if t in vad_dict]
    coverage = len(vals) / len(tokens) if tokens else 0
    return (np.mean(vals, axis=0) if vals else (np.nan, np.nan, np.nan), coverage)

# Aggregating weighted VAD scores and coverage
def weighted_vad(paras, doc_topics_all, character):
    topic_vad = {}
    topic_coverage = {}
    for para, topic_probs in zip(paras, doc_topics_all):
        (V, A, D), cov = compute_vad(para)
        for topic, prob in topic_probs:
            topic_id = f"{character}{topic}"
            if topic_id not in topic_vad:
                topic_vad[topic_id] = {'V': 0, 'A': 0, 'D': 0, 'weight': 0}
                topic_coverage[topic_id] = {'cov_sum': 0, 'weight': 0}
            if not np.isnan(V):
                topic_vad[topic_id]['V'] += V * prob
                topic_vad[topic_id]['A'] += A * prob
                topic_vad[topic_id]['D'] += D * prob
                topic_vad[topic_id]['weight'] += prob
                topic_coverage[topic_id]['cov_sum'] += cov * prob
                topic_coverage[topic_id]['weight'] += prob
    for t in topic_vad:
        w = topic_vad[t]['weight']
        if w > 0:
            topic_vad[t]['V'] /= w
            topic_vad[t]['A'] /= w
            topic_vad[t]['D'] /= w
            topic_coverage[t]['cov'] = topic_coverage[t]['cov_sum'] / w
        else:
            print(f"Warning: Topic {t} has no valid VAD scores (weight=0)")
            topic_coverage[t]['cov'] = 0
    return topic_vad, topic_coverage

clarissa_topic_vad, clarissa_topic_coverage = weighted_vad(clarissa_paras, clarissa_doc_topics_all, character='C')
septimus_topic_vad, septimus_topic_coverage = weighted_vad(septimus_paras, septimus_doc_topics_all, character='S')

# Computing cluster-level average VAD scores and coverage
def average_cluster_vad(topics, topic_vad_dict, topic_coverage_dict):
    vs, a_s, d_s, covs = [], [], [], []
    for t in topics:
        if t in topic_vad_dict:
            vs.append(topic_vad_dict[t]['V'])
            a_s.append(topic_vad_dict[t]['A'])
            d_s.append(topic_vad_dict[t]['D'])
            covs.append(topic_coverage_dict[t]['cov'])
    if not vs:
        return {'V': np.nan, 'A': np.nan, 'D': np.nan, 'cov': np.nan}
    return {'V': np.mean(vs), 'A': np.mean(a_s), 'D': np.mean(d_s), 'cov': np.mean(covs)}

cluster_rows = []
for _, row in alignment_df.iterrows():
    clar_vad = average_cluster_vad(row['Clarissa_Topics'], clarissa_topic_vad, clarissa_topic_coverage)
    sept_vad = average_cluster_vad(row['Septimus_Topics'], septimus_topic_vad, septimus_topic_coverage)
    cluster_rows.append({
        'Cluster_ID': row['Cluster_ID'],
        'V_Clarissa': clar_vad['V'],
        'A_Clarissa': clar_vad['A'],
        'D_Clarissa': clar_vad['D'],
        'Coverage_Clarissa': clar_vad['cov'],
        'V_Septimus': sept_vad['V'],
        'A_Septimus': sept_vad['A'],
        'D_Septimus': sept_vad['D'],
        'Coverage_Septimus': sept_vad['cov']
    })

df_cluster_vad = pd.DataFrame(cluster_rows)
df_cluster_vad.to_csv(os.path.join(output_folder, "cluster_vad_results.csv"), index=False)
print("Cluster VAD and coverage results saved.")

# Plotting cluster-level bar charts
def plot_cluster_bar(df, output_folder):
    for _, row in df.iterrows():
        labels = ['Valence','Arousal','Dominance']
        clar_vals = [row['V_Clarissa'],row['A_Clarissa'],row['D_Clarissa']]
        sept_vals = [row['V_Septimus'],row['A_Septimus'],row['D_Septimus']]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(x - width/2, clar_vals, width, color=COLORS['Clarissa'], label='Clarissa')
        ax.bar(x + width/2, sept_vals, width, color=COLORS['Septimus'], label='Septimus')
        
        for i, c_val in enumerate(clar_vals):
            if c_val >= 0:
                va = 'bottom'
                offset = 0.01
            else:
                va = 'top'
                offset = -0.01
            ax.text(i - width/2, c_val + offset, f'{c_val:.3f}', ha='center', va=va, fontsize=10)
        
        for i, s_val in enumerate(sept_vals):
            if s_val >= 0:
                va = 'bottom'
                offset = 0.01
            else:
                va = 'top'
                offset = -0.01
            ax.text(i + width/2, s_val + offset, f'{s_val:.3f}', ha='center', va=va, fontsize=10)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("VAD Score")
        cluster_id_str = str(int(row['Cluster_ID']))
        ax.set_title(f"Cluster {cluster_id_str} VAD Comparison", fontweight='bold')
        ax.set_ylim(-0.25,0.5)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', frameon=False)
        ax.set_facecolor('#F8FAFC')
        
        plt.savefig(os.path.join(output_folder, f"bar_cluster_{row['Cluster_ID']}.tif"), format='tif', dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Bar plot for Cluster {row['Cluster_ID']} saved.")

# Generating plots
plot_cluster_bar(df_cluster_vad, output_folder)