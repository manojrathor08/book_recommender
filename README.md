# Book Recommendation System

This repository contains a Book Recommendation System built using pre-trained Sentence-BERT embeddings. The application recommends books based on the similarity of their summaries and shared categories.

## Features
- Precomputed embeddings for faster recommendations.
- Category filtering to ensure relevant results.
- Cosine similarity for calculating book similarity.
- User-friendly modular code structure.

## File Structure
- **`app.py`**: The main Python script for running the recommendation system.
- **`books_summary.csv`**: Dataset containing book summaries, categories, and names.
- **`book_embeddings.npy`**: Precomputed Sentence-BERT embeddings for all book summaries.
- **`requirements.txt`**: List of dependencies required to run the project.

## How It Works
1. The dataset is preprocessed to handle missing values, duplicates, and group categories for each book.
2. Sentence-BERT (`all-MiniLM-L6-v2`) is used to compute embeddings for book summaries.
3. The system calculates the cosine similarity between the input book and other books in the dataset.
4. Recommendations are filtered based on shared categories and ranked by similarity scores.

## Usage
1. Clone this repository:
   - git clone <repository_url>
   - cd <repository_folder>
2. Install the required dependencies:
   - pip install -r requirements.txt
3. Run the application:
   - python app.py
4. Enter a book title to receive recommendations.

## Example Input/Output
#### Input:
- Book title: Siddhartha

#### Output:
- Recommendations:
- Book Name: The Alchemist
- Book Name: Life of Pi
- Book Name: The Power of Now

## Requirements
- Python 3.7 or higher
- Libraries listed in requirements.txt

## Acknowledgments
### Pre-trained Sentence-BERT model used: 
- all-MiniLM-L6-v2

### Libraries used:
- pandas: For data manipulation.
- numpy: For numerical operations.
- sentence-transformers: For loading pre-trained embeddings.
- scikit-learn: For cosine similarity calculation. 