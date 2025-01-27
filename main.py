import numpy as np
from data_preparation import load_data, preprocess_data, generate_pairs, stratify_data
from fine_tuning import fine_tune_sbert
from embedding_faiss import generate_embeddings, build_faiss_index
from sentence_transformers import SentenceTransformer,InputExample

model_name = 'all-MiniLM-L6-v2'
def main():
    ### Step 1: Load and Preprocess Data ###
    print("Loading and preprocessing data...")
    data = load_data('books_summary.csv')
    data = preprocess_data(data)
    data.to_csv('preprocessed_books_data.csv', index=False)  # Save preprocessed data

    ### Step 2: Prepare Data for Fine-Tuning ###
    print("Generating embeddings for fine-tuning preparation...")
    data = generate_embeddings(data, model_name='all-MiniLM-L6-v2')  # Pre-trained model
    
    # Generate pairs and stratify
    pairs_df = generate_pairs(data)
    stratified_pairs_df = stratify_data(pairs_df)
    train_examples = [
        InputExample(texts=[row['text1'], row['text2']], label=float(row['similarity']))
        for _, row in stratified_pairs_df.iterrows()
    ]

    ### Step 3: Fine-Tune SBERT Model ###
    print("Fine-tuning SBERT model...")
    fine_tune_sbert(train_examples, model_name='all-MiniLM-L6-v2', output_path='fine_tuned_sbert', epochs=10)

    ### Step 4: Generate Embeddings Using Fine-Tuned Model ###
    print("Generating embeddings using the fine-tuned model...")
    embeddings = generate_embeddings(data, model_name='fine_tuned_sbert')

    ### Step 5: Build Faiss Index ###
    print("Building Faiss index...")
    build_faiss_index(embeddings, index_path="faiss_index.bin")

    ### Step 6: Save Embeddings ###
    print("Saving embeddings...")
    
    np.save('book_embeddings.npy', embeddings)  # Save embeddings
    print("All steps completed successfully!")
if __name__ == "__main__":
    main()
