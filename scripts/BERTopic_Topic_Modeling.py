# Install required libraries
!pip install bertopic gensim

# Import necessary libraries
import pandas as pd
from bertopic import BERTopic
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

# Load preprocessed Reddit data
# Assuming `df` is a DataFrame containing the preprocessed text and timestamps
reddits = df['text'].tolist()
timestamps = pd.to_datetime(df['Date']).tolist()

# Step 1: Train BERTopic model
# Initialize BERTopic with the "all-MiniLM-L6-v2" sentence transformer
topic_model = BERTopic(embedding_model="all-MiniLM-L6-v2", min_topic_size=35, verbose=True)

# Fit the model on the Reddit data
topics, probs = topic_model.fit_transform(reddits)

# Display topic information
topic_info = topic_model.get_topic_info()
print("Top 10 Topics:")
print(topic_info.head(10))

# Inspect a specific topic (e.g., topic 4)
print("\nTopic 4:")
print(topic_model.get_topic(4))

# Inspect outlier topics (topic -1)
print("\nOutlier Topics:")
print(topic_model.get_topic(-1))

# Visualize topics
topic_model.visualize_topics()

# Visualize topics over time
topics_over_time = topic_model.topics_over_time(reddits, timestamps, global_tuning=True, evolution_tuning=True, nr_bins=15)
topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=15)

# Step 2: Calculate topic coherence
# Tokenize texts for coherence calculation
tokenized_texts = [simple_preprocess(text) for text in reddits]

def calculate_coherence(model, texts):
    """
    Calculates topic coherence for a BERTopic model.
    Args:
        model: Trained BERTopic model.
        texts (list): List of tokenized texts.
    Returns:
        coherence_cv (float): C_v coherence score.
        coherence_umass (float): U_Mass coherence score.
    """
    # Create a dictionary and corpus for coherence calculation
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Extract topic words
    topics = model.get_topics()
    topic_words = [[word for word, _ in topics[topic_id]] for topic_id in topics if topic_id != -1]

    # Calculate C_v coherence
    coherence_model_cv = CoherenceModel(
        topics=topic_words, texts=texts, dictionary=dictionary, coherence='c_v'
    )
    coherence_cv = coherence_model_cv.get_coherence()

    # Calculate U_Mass coherence
    coherence_model_umass = CoherenceModel(
        topics=topic_words, texts=texts, dictionary=dictionary, corpus=corpus, coherence='u_mass'
    )
    coherence_umass = coherence_model_umass.get_coherence()

    return coherence_cv, coherence_umass

# Step 3: Calculate topic diversity
def calculate_topic_diversity(model, top_n=10):
    """
    Calculates topic diversity for a BERTopic model.
    Args:
        model: Trained BERTopic model.
        top_n (int): Number of top words to consider for each topic.
    Returns:
        td_score (float): Topic Diversity score.
    """
    # Extract topics and their top words
    topics = model.get_topics()
    topic_words = [model.get_topic(topic_id)[:top_n] for topic_id in topics if topic_id != -1]

    # Flatten the list of words and count unique words
    all_words = [word for topic in topic_words for word, _ in topic]
    unique_words = set(all_words)

    # Calculate Topic Diversity (TD) score
    td_score = len(unique_words) / len(all_words)
    return td_score

# Calculate coherence scores
coherence_cv, coherence_umass = calculate_coherence(topic_model, tokenized_texts)

# Calculate Topic Diversity (TD) score
td_score = calculate_topic_diversity(topic_model, top_n=10)

# Print evaluation metrics
print(f"Topic Coherence (C_v): {coherence_cv:.4f}")
print(f"Topic Coherence (U_Mass): {coherence_umass:.4f}")
print(f"Topic Diversity (TD): {td_score:.4f}")