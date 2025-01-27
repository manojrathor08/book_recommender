from rapidfuzz import fuzz, process

def recommend_books_with_faiss(book_title, data, faiss_index, embeddings, top_n=5, min_similarity=60):
    """
    Recommend books similar to the input book using Faiss for nearest-neighbor search.
    """
    # Normalize book titles to lowercase
    data['book_name'] = data['book_name'].str.lower()
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

    # Find the index of the input book
    input_idx = data[data['book_name'] == book_title].index[0]
    input_categories = data.loc[input_idx, 'categories_list']
    input_embedding = embeddings[input_idx].reshape(1, -1)  # Reshape for Faiss compatibility

    
    # Use Faiss to find the nearest neighbors once
    distances, indices = faiss_index.search(input_embedding, len(embeddings))  # Search all embeddings
    indices = indices.flatten()
    distances = distances.flatten()
    
    # Define maximum attempts to avoid infinite loop
    max_attempts = len(indices)
    attempt_count = 0
    filtered_books = []

    # Exclude the input book itself
    indices = indices[1:]
    distances = distances[1:]

    # Convert distances to cosine similarity
    cosine_similarities = 1 - (distances / 2)

    # Start filtering by categories
    while len(filtered_books) < top_n and attempt_count < max_attempts:
        
        idx = indices[attempt_count]
        sim = cosine_similarities[attempt_count]
        
        if len(input_categories & data.loc[idx, 'categories_list']) > 0:  # Category overlap
            filtered_books.append((data.loc[idx, 'book_name'], sim))
        
        attempt_count += 1

    return filtered_books[:top_n], book_title