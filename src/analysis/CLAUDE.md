# src/analysis/ — Model Dissection Analysis Module

## Role
Analyze trained models from "black box → gray box" perspective.
Reverse-engineer "what the model knows" through training data profiling,
Membership Inference, and embedding space analysis.

## Core Files

### data_profiler.py — Training Data Profiling
- Input sentences from various domains (news, medical, legal, religious, colloquial, etc.)
  and analyze translation quality variance to infer training data domain distribution
- Per-language performance mapping: systematic quality measurement across all supported languages
- Output: per-domain/per-language performance heatmap

### membership_inference.py — MIA (Membership Inference Attack)
- Probabilistically determine whether a given source-target sentence pair was included in training data
- Method: cross-entropy loss based threshold discrimination
- Cross-reference with public datasets (OPUS, CCMatrix, WikiMatrix) for known samples
- Output: per-sentence membership probability + ROC curve

### embedding_analyzer.py — Embedding Space Analysis
- Extract encoder embeddings and visualize with t-SNE/UMAP
- Analyze per-language cluster density → infer data richness
- Inter-language distance matrix → infer transfer learning paths
- Output: visualization images + quantitative analysis results JSON

## Ethical Considerations
- This module's purpose is **research understanding**, not intellectual property infringement
- Analysis targets are limited to **publicly available open-source models**
- Redistribution of extracted information is prohibited
- All analysis results must note: "This is probabilistic inference, not a definitive conclusion"

## Rules
- Analysis results must always be reported with confidence intervals
- Large-scale analyses (e.g., 600-language probing) must implement checkpointing for intermediate saves
- Visualizations use matplotlib + seaborn with a consistent style sheet
