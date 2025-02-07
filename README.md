# LoDA-BERTopic: Low-Rank Domain Adaptation BERTopic

This repository contains the implementation of LoDA-BERTopic, a domain-adaptive topic modeling framework
built on BERTopic. The model leverages Low-Rank Adaptation (LoRA) to fine-tune sentence
embeddings for improved topic coherence and diversity in specialized discussions.

---

## Table of Contents
1. [Overview](#overview)
2. [Setup](#setup)
3. [Usage](#usage)
   - [Reddit Post Collection](#reddit-post-collection)
   - [Reddit Post Preprocessing](#reddit-post-preprocessing)
   - [Fine-Tuning with LoRA](#fine-tuning-with-lora)
   - [Topic Modeling](#topic-modeling)
5. [Results](#results)


---

## Overview

The project consists of the following steps:
1. **Reddit Post Collection**: Collect climate change-related posts from Reddit using the Reddit API.
2. **Reddit Post Preprocessing**: Clean and preprocess the collected Reddit posts.
3. **Fine-Tuning with LoRA**: Fine-tune the `all-MiniLM-L6-v2` sentence transformer using LoRA on Multi-Head Self-Attention (MHSA) and on both (MHSA) and Feed-Forward Network (FFN) layers.
4. **Topic Modeling**: Apply BERTopic, LDA, and SeqLDA to the preprocessed data and evaluate their performance using topic coherence (C_v, UMass) and topic diversity (TD).

---

## Setup

1. Clone the repository:
   git clone https://github.com/Kadri-Wadie/LoDA-BERTopic.git
   cd loda-bertopic
2. Install the required dependencies:
   !pip install -r requirements.txt

## Usage
1. Reddit Post Collection
   Run the script to collect Reddit posts: 
   python scripts/reddit_crawler.py
2. Reddit Post Preprocessing
   Preprocess the collected Reddit posts:
   python scripts/preprocessing.py
   
3. Fine-Tuning with LoRA
   Fine-tune the sentence transformer using LoRA:
   python scripts/finetuning.py
   
4. Topic Modeling
   Run the topic modeling scripts:
  *BERTopic with base model:
   python scripts/bertopic_base.py
   
  *BERTopic with distilroberta:
   python scripts/bertopic_distilroberta.py
   
  *BERTopic with all-mpnet:
   python scripts/bertopic_mpnet.py
   
  *LDA:
   python scripts/lda_modeling.py
   
  *SeqLDA:
   python scripts/seqlda_modeling.py
   
## Results
   The results, including coherence scores (C_v, UMass),
   topic diversity (TD), and visualizations, are saved
   in the results/ folder.
   

   

   


