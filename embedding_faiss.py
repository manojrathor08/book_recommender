import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_embeddings(data, model_name='all-MiniLM-L6-v2'):
    """
    Generate embeddings for the dataset using a fine-tuned SBERT model.

    Args:
        data (pd.DataFrame): Dataset with summaries.
        model_path (str): Path to the fine-tuned SBERT model.

    Returns:
        np.ndarray: Generated embeddings.
    """
    model = SentenceTransformer(model_name)
    if model_name =='all-MiniLM-L6-v2':
      data['embeddings'] = data['summaries'].apply(lambda x: model.encode(x, convert_to_tensor=True))
      return data
    else:
      embeddings = model.encode(data['summaries'], batch_size=16, show_progress_bar=True)
      return embeddings


def build_faiss_index(embeddings, index_path="faiss_index.bin"):
    """
    Build a Faiss index for fast nearest-neighbor searches.

    Args:
        embeddings (np.ndarray): Array of embeddings.
        index_path (str): Path to save the Faiss index.

    Returns:
        faiss.Index: The built Faiss index.
    """
    embeddings = embeddings.astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"Faiss index saved to {index_path}.")
    return index
  
def load_embeddings(embedding_path):
    """
    Load embeddings from a file.
    """
    try:
        embeddings = np.load(embedding_path).astype('float32')
        return embeddings
    except Exception as e:
        raise FileNotFoundError(f"Error loading embeddings: {e}")

def load_faiss_index(index_path):
  """
  Load a prebuilt Faiss index from a file.
  """
  try:
      return faiss.read_index(index_path)
  except Exception as e:
      raise FileNotFoundError(f"Error loading Faiss index: {e}")

