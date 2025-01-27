import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr
from rapidfuzz import fuzz, process

### Step 1: Load Processed Data ###
data = pd.read_csv('preprocessed_books_data.csv')  # Load preprocessed data
data['categories_list'] = data['categories_list'].apply(eval).apply(set)  # Convert categories to sets

### Step 2: Embedding Loading ###
def load_embeddings(embedding_path):
    """
    Load embeddings from a file.
    """
    try:
        embeddings = np.load(embedding_path).astype('float32')  # Ensure float32 for Faiss compatibility
        return embeddings
    except Exception as e:
        raise FileNotFoundError(f"Error loading embeddings: {e}")

embeddings = load_embeddings('book_embeddings.npy')

### Step 3: Build or Load Faiss Index ###
def build_faiss_index(embeddings, index_path="faiss_index.bin"):
    """
    Build a Faiss index for fast nearest-neighbor searches and save it to a file.
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
    index.add(embeddings)  # Add embeddings to the index
    faiss.write_index(index, index_path)  # Save the index
    return index

def load_faiss_index(index_path="faiss_index.bin"):
    """
    Load a prebuilt Faiss index from a file.
    """
    return faiss.read_index(index_path)

try:
    faiss_index = load_faiss_index("faiss_index.bin")
except FileNotFoundError:
    faiss_index = build_faiss_index(embeddings, "faiss_index.bin")

### Step 4: Recommendation Function ###
def recommend_books_with_faiss(book_title, data, faiss_index, embeddings, top_n=5, min_similarity=60):
    """
    Recommend books similar to the input book using Faiss for nearest-neighbor search.
    """
    # Normalize book titles to lowercase
    book_title = book_title.lower()

    # Fuzzy matching for book title
    if book_title not in data['book_name'].values:
        closest_match = process.extractOne(
            book_title,
            data['book_name'].values,
            scorer=fuzz.token_sort_ratio
        )
        if closest_match is None or closest_match[1] < min_similarity:
            return [f"No close match found for '{book_title}'. Please try another title."], book_title
        
        book_title = closest_match[0]
        print(f"Giving results for: {book_title}")

    # Find the index of the input book
    input_idx = data[data['book_name'] == book_title].index[0]
    input_embedding = embeddings[input_idx].reshape(1, -1)  # Reshape for Faiss compatibility

    # Use Faiss to find the nearest neighbors
    distances, indices = faiss_index.search(input_embedding, top_n + 1)  # +1 to exclude itself
    indices = indices.flatten()
    distances = distances.flatten()

    # Exclude the input book itself
    indices = indices[1:]
    distances = distances[1:]

    # Convert distances to cosine similarity
    cosine_similarities = 1 - (distances / 2)

    # Filter by categories
    input_categories = data.loc[input_idx, 'categories_list']
    filtered_books = []
    for idx, sim in zip(indices, cosine_similarities):
        if len(input_categories & data.loc[idx, 'categories_list']) > 0:  # Category overlap
            filtered_books.append((data.loc[idx, 'book_name'], sim))
        if len(filtered_books) >= top_n:
            break

    # Fallback: Add recommendations without category filtering
    if len(filtered_books) < top_n:
        remaining_indices = [idx for idx in indices if idx not in [rec[0] for rec in filtered_books]]
        for idx, sim in zip(remaining_indices, cosine_similarities[len(filtered_books):]):
            filtered_books.append((data.loc[idx, 'book_name'], sim))
            if len(filtered_books) >= top_n:
                break

    return filtered_books[:top_n], book_title

### Step 5: Recommendation UI ###
def recommend_ui(book_title):
    recommendations, book_name = recommend_books_with_faiss(book_title, data, faiss_index, embeddings, top_n=5)
    
    if len(recommendations) == 0:
        return "Book not found in the dataset. Please try another title."
    
    output_message = f"Giving results for: {book_name}\n\nRecommended Books:\n"
    recommendations_list = "\n".join([f"{rec[0]} (Similarity: {rec[1]:.2f})" for rec in recommendations])
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
