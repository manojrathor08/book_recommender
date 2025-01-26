import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from rapidfuzz import process


### Step 1: Data Loading ###
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if not all(col in data.columns for col in ['book_name', 'summaries', 'categories']):
            raise ValueError("Dataset must contain 'book_name', 'summaries', and 'categories' columns.")
        return data
    except Exception as e:
        raise FileNotFoundError(f"Error loading file: {e}")


### Step 2: Preprocessing ###
def preprocess_data(data):
    data = data.dropna(subset=['book_name', 'summaries']).reset_index(drop=True)
    data = data.drop_duplicates(subset=['book_name', 'categories'], keep='first').reset_index(drop=True)
    data = data.groupby('book_name', as_index=False).agg({
        'summaries': 'first',
        'categories': ', '.join
    })
    data['categories_list'] = data['categories'].str.split(', ')
    data['combined_text'] = data.apply(
        lambda row: row['summaries'] + " " + " ".join(row['categories_list']),
        axis=1
    )
    return data


### Step 3: Embedding Loading ###
def load_embeddings(embedding_path):
    try:
        embeddings = np.load(embedding_path)
        return embeddings
    except Exception as e:
        raise FileNotFoundError(f"Error loading embeddings: {e}")


### Step 4: Recommendation Generation ###

def recommend_books_with_category_filter(book_title, data, embeddings, top_n=5):
    # Normalize book titles to lowercase
    book_title = book_title.lower()
    data['book_name'] = data['book_name'].str.lower()

    # Check for exact match
    if book_title not in data['book_name'].values:
        # Use fuzzy matching to find the closest match
        closest_match, score = process.extractOne(book_title, data['book_name'].values)
        if score < 70:  # Set a threshold for similarity
            return [f"Book not found in the dataset. Did you mean '{closest_match}'?"]
        book_title = closest_match

    # Find the index of the input book
    input_idx = data[data['book_name'] == book_title].index[0]
    input_embedding = embeddings[input_idx]
    input_categories = set(data.loc[input_idx, 'categories_list'])

    # Compute cosine similarity
    similarity_scores = cosine_similarity([input_embedding], embeddings).flatten()
    similarity_scores[input_idx] = -1  # Exclude the input book

    # Add similarity scores to a copy of the data
    data_copy = data.copy()
    data_copy['similarity'] = similarity_scores

    # Filter books by category overlap
    data_filtered = data_copy[data_copy['categories_list'].apply(lambda x: len(set(x) & input_categories) > 0)]

    # Sort by similarity score and select top_n recommendations
    recommended_books = data_filtered.sort_values(by='similarity', ascending=False).head(top_n)

    return recommended_books[['book_name', 'similarity']].values.tolist()


### Main Workflow ###
# Load data and embeddings
data = preprocess_data(load_data('books_summary.csv'))
embeddings = load_embeddings('book_embeddings.npy')

def recommend_ui(book_title):
    print('The book you entered is:', book_title)
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
