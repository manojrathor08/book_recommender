from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

def fine_tune_sbert(train_examples, model_name='all-MiniLM-L6-v2', output_path='fine_tuned_sbert1', epochs=10):
    """
    Fine-tune the SBERT model on the provided training data.

    Args:
        train_examples (List[InputExample]): List of training examples.
        model_name (str): Name of the pre-trained SBERT model.
        output_path (str): Path to save the fine-tuned model.
        epochs (int): Number of training epochs.
    """
    model = SentenceTransformer(model_name)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=128)
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=int(0.01 * len(train_dataloader)),
        output_path=output_path
    )
    print(f"Model fine-tuned and saved to {output_path}.")
