from data_preparation import load_data
from embedding_faiss import load_faiss_index, load_embeddings
from recommendation_system import recommend_books_with_faiss
import gradio as gr

# Load data and embeddings
data = load_data('preprocessed_books_data.csv')
data['categories_list'] = data['categories_list'].apply(eval).apply(set)  # Convert categories to sets
embeddings = load_embeddings('book_embeddings.npy')

# Load or build Faiss index
try:
    faiss_index = load_faiss_index("faiss_index.bin")
except FileNotFoundError:
    print("Faiss index not found. Please ensure embeddings and index are built.")

def recommend_ui(book_title):
    """
    Gradio wrapper for the recommendation system.
    """
    recommendations, book_name = recommend_books_with_faiss(
        book_title, data, faiss_index, embeddings, top_n=5
    )
    if len(recommendations)<2:
        return "Book not found in the dataset. Please try another title."
    return (
    f"Giving results for: {book_name}\n\n" +
    "\n".join([f"{i+1}. {rec[0]}" for i, rec in enumerate(recommendations)]))

# Initialize Gradio app
iface = gr.Interface(
    fn=recommend_ui,
    inputs=gr.Textbox(label="Enter Book Title"),
    outputs=gr.Textbox(label="Recommended Books"),
    title="Book Recommendation System",
    description="Enter a book title to get 5 similar recommendations based on content."
)

if __name__ == "__main__":
    iface.launch()
