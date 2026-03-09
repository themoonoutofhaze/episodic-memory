# Episodic Memory: Textual Answer Extraction from Egocentric Videos via NLQ

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-ee4c2c)
![Dataset](https://img.shields.io/badge/Dataset-Ego4D-green)

## 📌 Overview
Natural Language Querying (NLQ) tasks for video retrieval typically require viewers to manually watch retrieved video segments to find answers. This is especially challenging with **egocentric (first-person) videos**, which are often lengthy and highly unstructured.

This repository proposes a novel **two-step methodology** to directly extract concise textual answers from egocentric videos. By localizing the relevant segments first and only passing the essential frames to a Vision-Language Model (VLM), our strategy significantly reduces computational load while maintaining high answer quality.

## 🧠 Methodology
Our approach is divided into two main stages:

1. **Video Segment Localization (VSLNet & EgoVLP)**
   * We utilize **EgoVLP** pre-extracted features combined with **VSLNet**.
   * The model is trained on the **Ego4D dataset's** NLQ task to identify and predict the most relevant video segments corresponding to the natural language query.
   * From these predictions, we filter and select the top 50 most successful localized segments.

2. **Textual Answer Generation (Video-LLaVA)**
   * The selected top video segments, alongside the user's original query, are processed by **Video-LLaVA** (a powerful Vision-Language Model).
   * Video-LLaVA "watches" the shortened segments and directly generates a concise textual answer, improving both retrieval efficiency and the end-user experience.

## 📊 Evaluation & Performance
The generated textual answers are rigorously evaluated against multiple standard NLP benchmarks to ensure accuracy and contextual relevance. Our metrics include:
* **F1 Score**
* **BLEU**
* **ROUGE-L**
* **BERTScore**
* **METEOR**

*Results demonstrate that this approach effectively provides accurate, concise answers to complex natural language queries in unstructured egocentric environments.*

## ⚙️ Getting Started

### Prerequisites
* Python 3.8+
* PyTorch
* [Other requirements...]

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/themoonoutofhaze/episodic-memory.git
   cd episodic-memory
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Preparation
1. Download the **Ego4D dataset** and pre-extracted **EgoVLP** features.
2. Place the data in the `/data` directory as structured below:
   ```text
   /data
     ├── ego4d_features/
     ├── annotations/
   ```

### Usage
*(Add your specific run commands here)*

**To run the segment localization (VSLNet):**
```bash
python run_localization.py --config config.yaml
```

**To generate answers using Video-LLaVA:**
```bash
python generate_answers.py --query "Where did I leave my keys?" --video_path /path/to/video
```

## 🤝 Acknowledgments
* **Ego4D Dataset**: For providing the extensive egocentric video data and NLQ benchmarks.
* **Video-LLaVA**: For the underlying VLM architecture.
* **EgoVLP & VSLNet**: For video-language pre-training and localization.

