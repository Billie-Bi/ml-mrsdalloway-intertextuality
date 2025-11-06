import re
import spacy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import os

lda_top_words_clarissa = "data/clarissa_top10.txt"
lda_top_words_septimus = "data/septimus_top10.txt"

output_dir = "LDA_output/topic_alignment_results"
os.makedirs(output_dir, exist_ok=True)

nlp = spacy.load("en_core_web_sm")

# Parsing topic keywords with weights from file
def parse_top_words_with_weights(file_path):
    topic_words = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    topic_blocks = re.split(r'Topic: (\d+)', content)
    for i in range(1, len(topic_blocks), 2):
        topic_id = int(topic_blocks[i])
        words_line = topic_blocks[i+1]
        words_weights = re.findall(r'([\d\.]+)\*"(.*?)"', words_line)
        words_weights = [(word, float(weight)) for weight, word in words_weights]
        topic_words[topic_id] = words_weights
    return topic_words

clarissa_topics = parse_top_words_with_weights(lda_top_words_clarissa)
septimus_topics = parse_top_words_with_weights(lda_top_words_septimus)

# Computing weighted average vector for topic keywords
def topic_to_weighted_vector(words_weights, nlp):
    vecs = []
    weights = []
    for word, weight in words_weights:
        token = nlp(word)
        if token.has_vector:
            vecs.append(token.vector)
            weights.append(weight)
    if vecs:
        vecs = np.array(vecs)
        weights = np.array(weights).reshape(-1,1)
        return np.sum(vecs * weights, axis=0) / np.sum(weights)
    else:
        return np.zeros(nlp.vocab.vectors_length)

clarissa_vectors = {tid: topic_to_weighted_vector(words_weights, nlp)
                    for tid, words_weights in clarissa_topics.items()}
septimus_vectors = {tid: topic_to_weighted_vector(words_weights, nlp)
                    for tid, words_weights in septimus_topics.items()}

# Constructing cosine similarity matrix between topic vectors
clarissa_ids = sorted(clarissa_vectors.keys())
septimus_ids = sorted(septimus_vectors.keys())
similarity_matrix = np.zeros((len(clarissa_ids), len(septimus_ids)))

for i, cid in enumerate(clarissa_ids):
    for j, sid in enumerate(septimus_ids):
        similarity_matrix[i,j] = cosine_similarity(
            clarissa_vectors[cid].reshape(1,-1),
            septimus_vectors[sid].reshape(1,-1)
        )[0,0]

df_sim = pd.DataFrame(similarity_matrix,
                      index=[f"C{tid}" for tid in clarissa_ids],
                      columns=[f"S{tid}" for tid in septimus_ids])

similarity_matrix_csv = os.path.join(output_dir, "topic_similarity_matrix_weighted.csv")
df_sim.to_csv(similarity_matrix_csv)
print(f"Weighted similarity matrix saved to {similarity_matrix_csv}")

# Generating cross-character topic clusters via threshold-based graph
threshold = 0.91
G = nx.Graph()

for i, cid in enumerate(clarissa_ids):
    for j, sid in enumerate(septimus_ids):
        sim = similarity_matrix[i,j]
        if sim >= threshold:
            G.add_edge(f"C{cid}", f"S{sid}", weight=sim)

clusters = list(nx.connected_components(G))
cluster_list = []
for idx, cluster in enumerate(clusters):
    clarissa_nodes = sorted([n for n in cluster if n.startswith("C")])
    septimus_nodes = sorted([n for n in cluster if n.startswith("S")])
    cluster_list.append({
        "Cluster_ID": idx+1,
        "Clarissa_Topics": clarissa_nodes,
        "Septimus_Topics": septimus_nodes
    })

cluster_df = pd.DataFrame(cluster_list)
cluster_csv_path = os.path.join(output_dir, "cross_character_topic_clusters.csv")
cluster_df.to_csv(cluster_csv_path, index=False)
print(f"Cross-character topic clusters saved to {cluster_csv_path}")

# Visualizing topic similarity through heatmap
plt.figure(figsize=(8,6))
plt.rcParams['font.size'] = 12
sns.heatmap(df_sim, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Clarissa ↔ Septimus Topic Similarity (Weighted Vectors)")
plt.xlabel("Septimus Topic")
plt.ylabel("Clarissa Topic")

heatmap_path = os.path.join(output_dir, "topic_similarity_heatmap.tif")
plt.tight_layout()
plt.savefig(heatmap_path, dpi=600, format='tif', bbox_inches='tight')
plt.show()
print(f"Heatmap saved to {heatmap_path}")