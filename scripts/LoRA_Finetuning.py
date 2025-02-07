# Install required libraries
!pip install peft transformers sentence-transformers datasets wandb

# Import necessary libraries
import os
import re
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertTokenizer
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model
import wandb

# Initialize W&B for experiment tracking
wandb.init(project='lora-embedding-tuning', config={
    'r_values': [4, 6, 8, 16],
    'dropout_values': [0.1],
    'epoch_values': [100]
})

# Load datasets
def load_and_preprocess_datasets():
    """
    Load and preprocess climate-related datasets from Hugging Face.
    Returns:
        df_sentences_train (pd.DataFrame): Preprocessed training sentences.
        df_sentences_test (pd.DataFrame): Preprocessed test sentences.
    """
    # Load Net-Zero dataset
    df_net_zero_reduc = pd.read_csv("hf://datasets/climatebert/netzero_reduction_data/targets_final.csv")
    df_net_zero_reduc_cleaned = df_net_zero_reduc[df_net_zero_reduc['target'] != 'none']

    # Load TCFD dataset
    dataset_tcfd = load_dataset("climatebert/tcfd_recommendations")
    df_tcfd_train = dataset_tcfd['train'].to_pandas()
    df_tcfd_test = dataset_tcfd['test'].to_pandas()

    # Load Climate-Specificity dataset
    dataset_climate_speci = load_dataset("climatebert/climate_specificity")
    df_climate_speci_train = dataset_climate_speci['train'].to_pandas()
    df_climate_speci_test = dataset_climate_speci['test'].to_pandas()

    # Load Environmental Claims dataset
    dataset_claims = load_dataset("climatebert/environmental_claims")
    df_claims_train = dataset_claims["train"].to_pandas()
    df_claims_test = dataset_claims["test"].to_pandas()

    # Load Climate Commitments Actions dataset
    dataset_commitments = load_dataset("climatebert/climate_commitments_actions")
    df_commitments_train = dataset_commitments["train"].to_pandas()
    df_commitments_test = dataset_commitments["test"].to_pandas()

    # Merge and clean datasets
    df_all_train = pd.concat([
        df_commitments_train['text'], df_claims_train['text'],
        df_climate_speci_train['text'], df_tcfd_train['text'],
        df_net_zero_reduc_cleaned['text']
    ], ignore_index=True)
    df_all_train_cleaned = df_all_train.apply(clean_text)

    df_all_test = pd.concat([
        df_commitments_test['text'], df_claims_test['text'],
        df_climate_speci_test['text'], df_tcfd_test['text']
    ], ignore_index=True)
    df_all_test_cleaned = df_all_test.apply(clean_text)

    # Remove duplicates and create final datasets
    df_sentences_train = pd.DataFrame({'sentences': df_all_train_cleaned}).drop_duplicates()
    df_sentences_test = pd.DataFrame({'sentences': df_all_test_cleaned}).drop_duplicates()

    return df_sentences_train, df_sentences_test

# Text cleaning function
def clean_text(text):
    """
    Clean text by:
    1. Converting to lowercase
    2. Removing non-alphanumeric characters (except spaces)
    3. Removing stopwords
    """
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphanumeric characters
        text = ' '.join(word for word in text.split() if word not in ENGLISH_STOP_WORDS)  # Remove stopwords
    return text

# Tokenization function
def tokenize_texts(texts, tokenizer, max_length=128):
    """
    Tokenize texts using a tokenizer.
    Args:
        texts (list): List of text sentences.
        tokenizer: Pre-trained tokenizer.
        max_length (int): Maximum sequence length.
    Returns:
        dict: Tokenized encodings.
    """
    return tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')

# Custom Dataset class
class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for tokenized text data.
    """
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

# Custom loss function
def custom_loss(embeddings):
    """
    Compute custom loss for embeddings.
    Args:
        embeddings (torch.Tensor): Embeddings tensor.
    Returns:
        torch.Tensor: Computed loss.
    """
    if embeddings.shape[0] <= 1:
        return torch.tensor(0.0, device=embeddings.device)
    sim_matrix = torch.cdist(embeddings, embeddings, p=2)  # Pairwise distance
    return sim_matrix.mean()  # Mean pairwise distance

# Apply LoRA to the model
def apply_lora_to_model(base_model, r, alpha, dropout):
    """
    Apply LoRA to the base model.
    Args:
        base_model: Pre-trained model.
        r (int): Rank for LoRA.
        alpha (int): Scaling factor for LoRA.
        dropout (float): Dropout rate.
    Returns:
        model: Model with LoRA applied.
    """
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["query", "key", "value", "feedforward"],  # Apply LoRA to MHA and FFN
        lora_dropout=dropout,
        bias="none"
    )
    model = get_peft_model(base_model, config)
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False  # Freeze non-LoRA parameters
    return model

# Main training and evaluation loop
def train_and_evaluate(base_model, train_dataloader, test_dataloader, r_values, alpha_ranges, dropout_fixed, epochs_fixed):
    """
    Train and evaluate the model with different LoRA configurations.
    Args:
        base_model: Pre-trained model.
        train_dataloader: DataLoader for training data.
        test_dataloader: DataLoader for test data.
        r_values (list): List of rank values.
        alpha_ranges (dict): Dictionary of alpha values for each rank.
        dropout_fixed (float): Fixed dropout rate.
        epochs_fixed (int): Fixed number of epochs.
    Returns:
        results (list): List of results for each configuration.
        best_model: Best model based on test loss.
    """
    results = []
    best_loss = float('inf')
    best_model = None

    for r in r_values:
        for alpha in alpha_ranges[r]:
            print(f"Training with rank {r}, alpha {alpha}, and dropout {dropout_fixed}")
            model = apply_lora_to_model(base_model, r, alpha, dropout_fixed)
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

            # Training loop
            for epoch in range(epochs_fixed):
                model.train()
                epoch_loss = 0
                for batch in train_dataloader:
                    optimizer.zero_grad()
                    embeddings = model.encode(batch['input_ids'], attention_mask=batch['attention_mask'], convert_to_tensor=True)
                    loss = custom_loss(embeddings)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_epoch_loss = epoch_loss / len(train_dataloader)
                print(f"Epoch {epoch + 1}/{epochs_fixed} - Loss: {avg_epoch_loss}")
                wandb.log({'epoch': epoch + 1, 'train_loss': avg_epoch_loss})

            # Evaluation
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch in test_dataloader:
                    embeddings = model.encode(batch['input_ids'], attention_mask=batch['attention_mask'], convert_to_tensor=True)
                    test_loss += custom_loss(embeddings).item()

            avg_test_loss = test_loss / len(test_dataloader)
            print(f"Test Loss - Rank {r}, Alpha {alpha}: {avg_test_loss}")
            wandb.log({'test_loss': avg_test_loss})
            results.append({'rank': r, 'alpha': alpha, 'test_loss': avg_test_loss})

            # Save the best model
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                best_model = model
                model.save_pretrained(f"fine_tuned_model_r_{r}_alpha_{alpha}")

    return results, best_model

# Load and preprocess datasets
df_sentences_train, df_sentences_test = load_and_preprocess_datasets()

# Initialize tokenizer and tokenize datasets
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenize_texts(df_sentences_train['sentences'].tolist(), tokenizer)
test_encodings = tokenize_texts(df_sentences_test['sentences'].tolist(), tokenizer)

# Create datasets and dataloaders
train_dataset = CustomDataset(train_encodings)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
test_dataset = CustomDataset(test_encodings)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=32)

# Load the base model
base_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define hyperparameters
r_values = [4, 6, 8, 16]
alpha_ranges = {4: [8], 6: [12], 8: [16], 16: [32]}
dropout_fixed = 0.1
epochs_fixed = 30

# Train and evaluate the model
results, best_model = train_and_evaluate(base_model, train_dataloader, test_dataloader, r_values, alpha_ranges, dropout_fixed, epochs_fixed)

# Visualize results
results_df = pd.DataFrame(results)
plt.figure(figsize=(10, 6))
sns.lineplot(data=results_df, x='alpha', y='test_loss', hue='rank', markers=True, dashes=False)
plt.title('Test Loss across Different Ranks and Alpha Values')
plt.xlabel('Alpha')
plt.ylabel('Test Loss')
plt.legend(title='Rank')
plt.grid(True)
plt.show()

# Print the best configuration
best_result = results_df.loc[results_df['test_loss'].idxmin()]
print(f"Best Configuration - Rank: {best_result['rank']}, Alpha: {best_result['alpha']}, Test Loss: {best_result['test_loss']}")