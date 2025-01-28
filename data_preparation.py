import pandas as pd
from itertools import combinations
from sklearn.utils import shuffle
from sentence_transformers import SentenceTransformer,util, InputExample

def load_data(file_path):
    """
    Load the dataset from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        data = pd.read_csv(file_path)
        if not all(col in data.columns for col in ['book_name', 'summaries', 'categories']):
            raise ValueError("Dataset must contain 'book_name', 'summaries', and 'categories' columns.")
        return data
    except Exception as e:
        raise FileNotFoundError(f"Error loading file: {e}")

def preprocess_data(data):
    """
    Preprocess the dataset by:
    - Dropping rows with missing values in critical columns.
    - Removing duplicates based on 'book_name' and 'categories'.
    - Grouping categories for each book.
    - Adding a combined text column with summaries and categories.

    Args:
        data (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Remove duplicates where both 'book_name' and 'categories' are identical
    data = data.drop_duplicates(subset=['book_name', 'categories'], keep='first').reset_index(drop=True)
    # Drop rows with missing values in 'book_name' and 'summaries'
    missing_or_empty_rows = data[
    (data['book_name'].isna() | (data['book_name'].str.strip() == "") | (data['book_name'].str.strip() == ".")) |
    (data['summaries'].isna() | (data['summaries'].str.strip() == "") | (data['summaries'].str.strip() == "."))]

    # Drop these rows
    data = data.drop(missing_or_empty_rows.index)

    # Reset the index after dropping rows (optional)
    data.reset_index(drop=True, inplace=True)


    # Group categories for each book
    data = data.groupby('book_name', as_index=False).agg({
        'summaries': 'first',  # Retain the first summary
        'categories': ', '.join  # Combine categories into a single string
    })

    # Split categories into a list
    data['categories_list'] = data['categories'].str.split(', ')

    return data


def generate_embeddings(data, model_name='all-MiniLM-L6-v2'):
    """
    Generate embeddings for book summaries using a pre-trained SBERT model.

    Args:
        data (pd.DataFrame): Preprocessed dataset with 'summaries' column.
        model_name (str): Name of the pre-trained SBERT model.

    Returns:
        np.ndarray: Array of embeddings.
    """
    model = SentenceTransformer(model_name)
    data['embeddings'] = data['summaries'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return model, data
def generate_pairs(data, num_samples=1000):
    pairs = []
    sampled_combinations = combinations(data.iterrows(), 2)
    for (idx1, row1), (idx2, row2) in sampled_combinations:
        # Compute Jaccard similarity
        common_categories = len(set(row1['categories_list']) & set(row2['categories_list']))
        total_categories = len(set(row1['categories_list']) | set(row2['categories_list']))
        jaccard_similarity = common_categories / total_categories


        # Reuse precomputed embeddings
        semantic_similarity = util.pytorch_cos_sim(row1['embeddings'], row2['embeddings']).item()

        # Final similarity (weighted average)
        combined_similarity = 0.9 * semantic_similarity + 0.1 * jaccard_similarity

        # Append the pair with book names
        pairs.append({
            "book1": row1['book_name'],  # Book name for text1
            "book2": row2['book_name'],  # Book name for text2
            "text1": row1['summaries'] ,
            "text2": row2['summaries'] ,
            "similarity": combined_similarity
        })

    return pd.DataFrame(pairs)
# Define bins for similarity scores
def stratify_data(pairs_df, high_threshold=0.5, low_threshold=0.3, samples_per_bin=5000):
    """
    Stratifies pairs_df into bins based on similarity scores and samples equally from each bin.

    Args:
        pairs_df (pd.DataFrame): DataFrame containing the similarity scores.
        high_threshold (float): Threshold for high similarity.
        low_threshold (float): Threshold for low similarity.
        samples_per_bin (int): Number of samples to draw from each bin.

    Returns:
        pd.DataFrame: Stratified and sampled DataFrame.
    """
    # Define bins
    high_similarity = pairs_df[pairs_df['similarity'] >= high_threshold]
    moderate_similarity = pairs_df[(pairs_df['similarity'] < high_threshold) & (pairs_df['similarity'] >= low_threshold)]
    low_similarity = pairs_df[pairs_df['similarity'] < low_threshold]

    # Sample equally from each bin
    high_sample = high_similarity.sample(min(len(high_similarity), samples_per_bin), random_state=42)
    moderate_sample = moderate_similarity.sample(min(len(moderate_similarity), samples_per_bin), random_state=42)
    low_sample = low_similarity.sample(min(len(low_similarity), samples_per_bin), random_state=42)

    # Combine samples and shuffle
    stratified_data = pd.concat([high_sample, moderate_sample, low_sample])
    return shuffle(stratified_data, random_state=42)

