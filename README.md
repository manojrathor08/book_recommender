# Book Recommendation System

**Live Demo:** [View the demo here](https://huggingface.co/spaces/rathormanoj08/book_recommender)

This repository contains a Book Recommendation System that uses fine-tuned Sentence-BERT (SBERT) embeddings to recommend books based on the similarity of their summaries and shared categories.

## Features
 - **`Preprocessing`**: Handles missing values, removes duplicates, and organizes categories for each book.
 - **`Embedding Generation`**: Fine-tunes Sentence-BERT (all-MiniLM-L6-v2) on book summaries for domain-specific embeddings.
 - **`Efficient Search`**: Utilizes Faiss for fast searches.
 - **`Similarity Scoring`**: Combines cosine similarity and category filtering to ensure highly relevant recommendations.
 - **`Scalable Design`**: Modularized code for data preprocessing, embedding generation, and recommendation.

## File Structure
 - **`app.py`**: The main script for running the recommendation system.
 - **`books_summary.csv`**: The dataset containing book summaries, categories, and names.
 - **`preprocessed_books_data.csv`**: Processed dataset after cleaning and feature engineering.
 - **`book_embeddings.npy`**: Precomputed embeddings of book summaries (after fine-tuning SBERT).
 - **`faiss_index.bin`**: Faiss index for efficient similarity search.
 - **`requirements.txt`**: List of dependencies required to run the project.

## How It Works
 - **`Preprocessing`**:
    - Handles missing or empty rows for book_name and summaries.
    - Removes duplicates and groups categories for each book.
    - Splits categories into lists for efficient category filtering.

 - **`Fine-Tuning`**:
    - Fine-tunes Sentence-BERT (all-MiniLM-L6-v2) using stratified book pairs created based on:
    - Semantic similarity: Calculated using SBERT.
    - Jaccard similarity: Based on category overlap.

 - **`Embedding Generation`**:
    - Generates embeddings using the fine-tuned model and saves them as book_embeddings.npy.

 - **`Faiss Indexing`**:
    - Builds a Faiss L2 Index for fast nearest-neighbor searches.
    - Saves the index as faiss_index.bin.

 - **`Recommendation`**:
     - Searches for similar books using Faiss.
     - Filters recommendations based on shared categories with the input book.
     - Returns the top 5 recommendations ranked by similarity.

## Usage
 - **`Clone this repository`**:
   - git clone <repository_url>
   - cd <repository_folder>
 - **`Install the required dependencies`**:
   - pip install -r requirements.txt
 - **`Fine-tune the model`**:
    - python main.py
 - **`Run the application`**:
   - python app.py

## Example Input/Output
##### Input:
- Book title: Siddhartha

##### Output:
- Recommendations:
    - Giving results for: siddhartha
    1. the universe has your back
    2. you’ll see it when you believe it
    3. trying not to try
    4. everyday zen
    5. see you at the top

## Requirements
- Python 3.7 or higher
- Libraries listed in requirements.txt

## Acknowledgments
### Pre-trained Sentence-BERT model used: 
- Pre-trained SBERT Model: all-MiniLM-L6-v2

### Libraries used:
- **pandas**: For data manipulation and cleaning.
- **numpy**: For numerical operations and handling embeddings.
- **sentence-transformers**: For loading pre-trained models and generating embeddings.
- **scikit-learn**: For cosine similarity calculation.
- **Faiss**: A library for efficient similarity searches.
- **Hugging Face Spaces**: For hosting and running the application in a web-based environment.
- **rapidfuzz**: For fuzzy string matching to find close matches to book titles.
- **Gradio**: For building a user-friendly web interface for the recommendation system.
