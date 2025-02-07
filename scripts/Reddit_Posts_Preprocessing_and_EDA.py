# Import necessary libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources (stopwords and tokenizer)
nltk.download('stopwords')
nltk.download('punkt')

# Load Reddit posts from CSV file
csv_path = 'reddit_posts.csv'  # Replace with your CSV file path
df = pd.read_csv(csv_path)

# Display basic information about the dataset
print("Dataset Overview:")
print(df.head())  # Display first 5 rows
print(df.tail())  # Display last 5 rows
print("\nColumns in the dataset:")
print(df.columns)
print("\nData types:")
print(df.dtypes)

# Exploratory Data Analysis (EDA): Sentence Lengths Before Preprocessing
# Extract comments and compute sentence lengths
text_comm = df['Comment'].tolist()
sentence_lengths_before = [len(sentence.split()) for sentence in text_comm]

# Create a DataFrame for visualization
data_before = pd.DataFrame({'Sentence Length': sentence_lengths_before})

# Plot histogram of sentence lengths before preprocessing
plt.figure(figsize=(10, 6))
sns.histplot(data_before['Sentence Length'], bins=30, kde=True, color='blue')
plt.title('Distribution of Sentence Lengths (Before Preprocessing)')
plt.xlabel('Sentence Length (Number of Words)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Print max and min sentence lengths before preprocessing
max_length_before = data_before['Sentence Length'].max()
min_length_before = data_before['Sentence Length'].min()
print(f"Maximum Sentence Length (Before Preprocessing): {max_length_before} words")
print(f"Minimum Sentence Length (Before Preprocessing): {min_length_before} words")

# Text Preprocessing Function
def preprocess_text(text):
    """
    Preprocesses text by:
    1. Removing URLs
    2. Converting to lowercase
    3. Removing mentions (@)
    4. Removing non-alphabetic characters
    5. Removing stopwords
    6. Filtering out short sentences (less than 5 words)
    """
    if isinstance(text, str):  # Ensure the input is a string
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove mentions
        text = " ".join(filter(lambda x: x[0] != "@", text.split()))
        # Remove non-alphabetic characters
        text = " ".join(re.sub("[^a-zA-Z]+", " ", text).split())
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        text = " ".join(filtered_text)
        # Filter out short sentences
        if len(text.split()) > 4:  # Keep sentences with more than 4 words
            return text
    return ""  # Return empty string for short or invalid sentences

# Apply preprocessing to the 'Comment' column
df['text'] = df['Comment'].apply(preprocess_text)

# Filter out rows where 'text' is empty
df = df[df['text'] != ""]

# Save the preprocessed data to a new CSV file
preprocessed_csv_path = 'reddit_posts_preprocessed.csv'
df.to_csv(preprocessed_csv_path, index=False)
print(f"\nPreprocessed data saved to: {preprocessed_csv_path}")

# Exploratory Data Analysis (EDA): Sentence Lengths After Preprocessing
# Compute sentence lengths after preprocessing
sentence_lengths_after = [len(sentence.split()) for sentence in df['text']]

# Create a DataFrame for visualization
data_after = pd.DataFrame({'Sentence Length': sentence_lengths_after})

# Plot histogram of sentence lengths after preprocessing
plt.figure(figsize=(10, 6))
sns.histplot(data_after['Sentence Length'], bins=30, kde=True, color='green')
plt.title('Distribution of Sentence Lengths (After Preprocessing)')
plt.xlabel('Sentence Length (Number of Words)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Print max and min sentence lengths after preprocessing
max_length_after = data_after['Sentence Length'].max()
min_length_after = data_after['Sentence Length'].min()
print(f"Maximum Sentence Length (After Preprocessing): {max_length_after} words")
print(f"Minimum Sentence Length (After Preprocessing): {min_length_after} words")