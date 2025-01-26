import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr


### Step 1: Data Loading ###
def load_data(file_path):
    """
    Load the dataset from a CSV file and return a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise FileNotFoundError(f"Error loading file: {e}")


### Step 2: Preprocessing ###
def preprocess_data(data):
    """
    Preprocess the dataset by handling missing values, removing duplicates, and formatting categories.

    Args:
        data (pd.DataFrame): Raw dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Drop missing values
    data = data.dropna(subset=['book_name', 'summaries']).reset_index(drop=True)

    # Remove duplicates
    data = data.drop_duplicates(subset=['book_name', 'categories'], keep='first').reset_index(drop=True)

    # Group categories for each book
    data = data.groupby('book_name', as_index=False).agg({
        'summaries': 'first',  # Retain the first summary
        'categories': ', '.join  # Combine categories into a single string
    })

    # Add additional features
    data['categories_list'] = data['categories'].str.split(', ')
    data['combined_text'] = data.apply(
        lambda row: row['summaries'] + " " + " ".join(row['categories_list']),
        axis=1
    )

    return data


### Step 3: Embedding Loading ###
def load_embeddings(embedding_path):
    """
    Load precomputed embeddings from a file.

    Args:
        embedding_path (str): Path to the embeddings file.

    Returns:
        np.ndarray: Loaded embeddings.
    """
    try:
        embeddings = np.load(embedding_path)
        return embeddings
    except Exception as e:
        raise FileNotFoundError(f"Error loading embeddings: {e}")


### Step 4: Recommendation Generation ###
def recommend_books_with_category_filter(book_title, data, embeddings, top_n=5):
    """
    Recommend books similar to the input book with category filtering.

    Args:
        book_title (str): The title of the book to recommend similar books for.
        data (pd.DataFrame): The dataset containing book information.
        embeddings (np.ndarray): Precomputed embeddings.
        top_n (int): Number of recommendations to return.

    Returns:
        list: Recommended book names.
    """
    if book_title not in data['book_name'].values:
        return ["Book not found in the dataset."]

    # Find the index of the input book
    input_idx = data[data['book_name'] == book_title].index[0]
    input_embedding = embeddings[input_idx]
    input_categories = set(data.loc[input_idx, 'categories_list'])

    # Compute cosine similarity
    similarity_scores = cosine_similarity([input_embedding], embeddings).flatten()
    similarity_scores[input_idx] = -1  # Exclude the input book

    # Add similarity scores to the DataFrame
    data['similarity'] = similarity_scores

    # Filter books by category overlap
    data_filtered = data[data['categories_list'].apply(lambda x: len(set(x) & input_categories) > 0)]

    # Sort by similarity score and select top_n recommendations
    recommended_books = data_filtered.sort_values(by='similarity', ascending=False).head(top_n)

    return recommended_books[['book_name', 'similarity']].values.tolist()


### Main Workflow ###


def recommend_ui(book_title):
    """
    Gradio wrapper for the recommendation system.
    
    Args:
        book_title (str): User input book title.

    Returns:
        list: Recommended books.
    """
    recommendations = recommend_books_with_category_filter(book_title, data, embeddings, top_n=5)
    if recommendations[0] == "Book not found in the dataset.":
        return "Book not found in the dataset. Please try another title."
    return [f"{rec[0]} (Similarity: {rec[1]:.4f})" for rec in recommendations]

# Gradio interface
iface = gr.Interface(
    fn=recommend_ui,
    inputs=gr.Textbox(label="Enter Book Title"),
    outputs=gr.Textbox(label="Recommended Books"),
    title="Book Recommendation System",
    description="Enter a book title to get 5 similar recommendations based on content."
)

if __name__ == "__main__":
    iface.launch()
