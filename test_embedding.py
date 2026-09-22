from src.models import EmbeddingModel


model = EmbeddingModel()

texts = [
    "What is attention in transformers?",
    "Attention allows a model to focus on different words."
]

embeddings = model.encode(texts)

print("Shape:", embeddings.shape)

print("First embedding:")
print(embeddings[0][:10])