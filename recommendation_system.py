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