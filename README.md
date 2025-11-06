# ml-mrsdalloway-intertextuality
A unified machine learning framework for thematic, emotional, and semantic analysis of Clarissa and Septimus’ narrative intertextuality in Mrs Dalloway.

---
## Overview

This repository contains the **analysis scripts** for a machine learning–enhanced computational study of narrative intertextuality between Clarissa Dalloway and Septimus Warren Smith in *Virginia Woolf’s Mrs Dalloway*.
The framework integrates NLP tools such as **spaCy**, **Gensim**, **Sentence-BERT**, **UMAP**, **HDBSCAN**, and the **NRC VAD Lexicon** to quantify thematic structures, emotional profiles, and semantic spaces, revealing patterns in **psychological resonance** and **modernist narrative technique**.
It includes scripts for text preprocessing, LDA topic modeling, topic alignment, VAD emotional analysis, and semantic space visualization, ensuring full methodological transparency and reproducibility.

This repository accompanies the paper:
**“Machine Learning Study of Narrative Intertextuality: Integrated Analysis of Thematic, Emotional, and Semantic Dimensions in Mrs Dalloway.”**

**Note:**
To maintain the integrity of the peer review process, all raw and processed data are withheld during review.
Upon acceptance, both datasets will be released through **Dryad** and linked here.
All analysis scripts, environment settings, and processing logic are fully available in this repository.
---

## Repository Structure
```
ml-narrative-intertextuality-analysis/
├── spacy_lda_clarissa.py                            # Preprocessing and LDA for Clarissa's text
├── spacy_lda_septimus.py                            # Preprocessing and LDA for Septimus's text
├── topic_alignment_results.py                       # Topic alignment and similarity analysis
├── compute_topic_vad_comparison.py                  # VAD emotional analysis on topics
├── semantic_space_analysis.py                       # Semantic embedding, clustering, and visualization
├── data/                                            # (empty) Placeholder for source texts
│ └── .gitkeep
├── LDA_output/                                      # (empty) Placeholder for LDA results
│ └── .gitkeep
├── VAD_output/                                      # (empty) Placeholder for VAD results
│ └── .gitkeep
├── semantic_space_analysis_output/                  # (empty) Placeholder for semantic results
│ └── .gitkeep
├── spacy_lda_vad.yaml                               # Python dependencies
├── semantic_space.yaml                              # Python dependencies
├── .gitignore                                       # Python-specific ignores
├── LICENSE                                          # MIT License
└── README.md                                        # This file
```
---

## Installation
### Prerequisites
- Python 3.10 or above
- Git (for cloning the repository)
### Setup
```
git clone https://github.com/yourusername/ml-narrative-intertextuality-analysis.git
cd ml-narrative-intertextuality-analysis
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
conda env create -f spacy_lda_vad.yaml
conda env create -f semantic_space.yaml 
python -m spacy download en_core_web_sm
```
---

## Usage
### 1. Text Preprocessing and LDA for Clarissa
Preprocesses Clarissa's text and performs LDA topic modeling.
```
python spacy_lda_clarissa.py
```
### 2. Text Preprocessing and LDA for Septimus
Preprocesses Septimus's text and performs LDA topic modeling.
```
python spacy_lda_septimus.py
```
### 3. Topic Alignment
Aligns topics between Clarissa and Septimus using cosine similarity, generates clusters and visualizations.
```
python topic_alignment_results.py
```
### 4. VAD Emotional Analysis
Computes VAD profiles for topics and clusters, generates bar plots.
```
python compute_topic_vad_comparison.py
```
### 5. Semantic Space Analysis
Performs embedding, UMAP reduction, HDBSCAN clustering, and semantic matching.
```
python semantic_space_analysis.py
```
---

## Data Availability

The novel text and all processed data are currently withheld during peer review to ensure double-blind integrity.  
Upon acceptance, both the raw and processed datasets will be publicly released via **Zenodo**, and this section will be updated with the DOI.
---

## Citation
If you use or adapt the analysis scripts, please cite:
```
@article{bi_wu_pan_2025,
  title = {Machine Learning Study of Narrative Intertextuality: Integrated Analysis of Thematic, Emotional, and Semantic Dimensions in Mrs Dalloway},
  author = {Bi, Liqi and Wu, Guanting and Pan, Yan},
  year = {2025},
  note = {Code available at: https://github.com/Billie-Bi/ml-mrsdalloway-intertextuality}
}
```
---

## License
This project is licensed under the MIT License — see the LICENSE file for details.
---

## Acknowledgments
Tools: spaCy, Gensim, Sentence-Transformers, UMAP, HDBSCAN, scikit-learn, NetworkX, matplotlib, pandas, etc.
Data: Virginia Woolf’s Mrs Dalloway (public domain via Project Gutenberg, to be released post-review).
