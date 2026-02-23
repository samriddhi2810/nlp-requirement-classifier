📌 Semantic Classifier
An Empirical Study of Cross-Domain Robustness in Functional Requirement Classification
📖 Overview

This project investigates the robustness of machine learning models for Functional vs Non-Functional Requirement (FR/NFR) classification under cross-domain conditions.

While many published approaches report strong in-domain performance, their effectiveness under domain shift remains underexplored. This study provides a systematic evaluation of:

Lexical representations (TF-IDF)

Semantic representations (Sentence-BERT)

Cross-domain performance degradation

Few-shot domain adaptation

The goal is to understand how requirement classifiers behave when deployed across heterogeneous software domains.

🎯 Research Questions

How do lexical and semantic models behave under domain shift?

How severe is cross-domain performance degradation?

Do sentence-level semantic embeddings improve robustness?

How much target-domain supervision is required to recover performance?

📂 Datasets
1️⃣ PROMISE Dataset

~600 labeled requirements

Highly imbalanced (~89% Non-Functional)

Small-scale academic benchmark

2️⃣ PURE Dataset

~11,000+ labeled requirements

Opposite label distribution (~83% Functional)

Large-scale industrial dataset

This label distribution inversion introduces significant domain shift.

🧠 Methodology
1️⃣ Lexical Baseline

TF-IDF (unigrams + bigrams)

Linear SVM classifier

Evaluated using Macro F1-score

2️⃣ Semantic Modeling

Sentence-BERT embeddings (all-MiniLM-L6-v2)

Frozen embedding extraction

Linear SVM classifier on top

Cross-domain evaluation (PROMISE → PURE)

3️⃣ Few-Shot Domain Adaptation

Train on PROMISE (source domain)

Incrementally add labeled samples from PURE (target domain)

Measure Macro F1 recovery curve

📊 Key Results
Cross-Domain Robustness (PROMISE → PURE)
Model	Macro F1
TF-IDF	0.157
SBERT	0.282

Sentence-level semantic embeddings improve cross-domain Macro F1 relative to lexical features.

Label Distribution Shift
Dataset	Functional	Non-Functional
PROMISE	11%	89%
PURE	83%	17%

The dominant class flips across datasets, significantly impacting model generalization.

Few-Shot Adaptation Results
Labeled PURE Samples	Macro F1
0	0.266
10	0.409
50	0.529
100	0.574
500	0.654

Observation:

Even small amounts (50–100 samples) of labeled target data substantially recover cross-domain performance.

🔎 Core Findings

Lexical models degrade sharply under domain shift.

Sentence-level semantic embeddings provide improved robustness.

Label distribution shift is a major contributing factor to cross-domain failure.

Few-shot supervision is highly effective in mitigating performance loss.📈 Evaluation Metric

Macro F1-score is used due to severe class imbalance in both datasets.

Macro F1 treats each class equally and avoids bias toward majority-class dominance.

```
🧪 Project Structure
semantic_classifier/
│
├── data/
├── src/
│   ├── train_svm.py
│   ├── sbert.py
│   ├── fewshot_domain_adaptation.py
│   ├── analyze_label_distribution.py
│   └── plot_fewshot_curve.py
│
├── fewshot_adaptation_curve.png
├── confusion_matrix.png
├── f1_scores.png
└── README.md
🚀 How to Run
```


Install dependencies:

pip install -r requirements.txt

Run few-shot experiment:

python src/fewshot_domain_adaptation.py

Generate adaptation curve:

python src/plot_fewshot_curve.py
📌 Conclusion

This study demonstrates that:

Semantic sentence embeddings improve cross-domain robustness in requirement classification.

Label distribution shift significantly affects generalization.

Limited target-domain supervision can efficiently recover degraded performance.

These findings highlight the importance of domain awareness and data efficiency in practical NLP-based requirement engineering systems.
