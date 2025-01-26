import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import gradio as gr
from rapidfuzz import fuzz, process
from joblib import Parallel, delayed

### Step 1: Load processed data ###
data = pd.read_csv('preprocessed_books_data.csv')  # Load from CSV

# Convert categories to sets during initialization for faster filtering
data['categories_list'] = data['categories_list'].apply(eval).apply(set)

### Step 2: Embedding Loading ###
def load_embeddings(embedding_path):
    try:
        embeddings = np.load(embedding_path)
        return embeddings
    except Exception as e:
        raise FileNotFoundError(f"Error loading embeddings: {e}")

# Step 3: Load precomputed similarity matrix ###
def load_similarity_matrix(similarity_path, embeddings, reduce_dim=True):
    try:
        return np.load(similarity_path)
    except FileNotFoundError:
        # If similarity matrix is missing, compute it
        if reduce_dim:
            print("Reducing embedding dimensions using PCA...")
            pca = PCA(n_components=128)
            embeddings = pca.fit_transform(embeddings)
            np.save("reduced_embeddings.npy", embeddings)

        print("Computing similarity matrix...")
        similarity_matrix = cosine_similarity(embeddings)
        np.save(similarity_path, similarity_matrix)
        return similarity_matrix

embeddings = load_embeddings('book_embeddings.npy')
similarity_matrix = load_similarity_matrix("similarity_matrix.npy", embeddings)

### Step 4: Recommendation Cache ###
recommendation_cache = {}

def recommend_books_with_category_filter(book_title, data, similarity_matrix, top_n=5, min_similarity=60):
    # Check cache first
    if book_title in recommendation_cache:
        return recommendation_cache[book_title]

    # Normalize book titles to lowercase
    book_title = book_title.lower()

    # Adjust the similarity threshold for numeric titles
    if book_title.isdigit():
        min_similarity = 50

    # Fuzzy matching
    if book_title not in data['book_name'].values:
        # Narrow down candidates with substring filtering
        candidates = [name for name in data['book_name'] if book_title in name]
        if not candidates:
            candidates = data['book_name'].values

        closest_match = process.extractOne(
            book_title,
            candidates,
            scorer=fuzz.token_sort_ratio
        )

        if closest_match is None or closest_match[1] < min_similarity:
            return [f"No close match found for '{book_title}'. Please try another title."], book_title

        book_title = closest_match[0]
        print(f"Giving results for: {book_title}")

    # Find the index of the input book
    input_idx = data[data['book_name'] == book_title].index[0]

    # Use precomputed similarities
    similarity_scores = similarity_matrix[input_idx]
    similarity_scores[input_idx] = -1  # Exclude the input book

    # Filter by categories with NumPy
    input_categories = data.loc[input_idx, 'categories_list']
    category_filter = np.array([
        len(input_categories & categories) > 0 for categories in data['categories_list']
    ])
    data_filtered = data.loc[category_filter].copy()
    data_filtered['similarity'] = similarity_scores[category_filter]

    # Get top recommendations
    recommended_books = data_filtered.sort_values(by='similarity', ascending=False).head(top_n)
    recommendations = recommended_books[['book_name', 'similarity']].values.tolist()

    # Cache the results
    recommendation_cache[book_title] = (recommendations, book_title)
    return recommendations, book_title

### Step 5: Recommendation UI ###
def recommend_ui(book_title):
    recommendations, book_name = recommend_books_with_category_filter(book_title, data, similarity_matrix, top_n=5)
    
    if len(recommendations) < 2:
        return "Book not found in the dataset. Please try another title."
    
    output_message = f"Giving results for: {book_name}\n\nRecommended Books:\n"
    recommendations_list = "\n".join([f"{rec[0]}" for rec in recommendations])
    return output_message + recommendations_list

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
