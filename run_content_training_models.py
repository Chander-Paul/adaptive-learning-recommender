from models.content.content_model import ContentModel
import pandas as pd

# Create model with bi-encoder (default)
model = ContentModel()
model.load_model()

# Load resources
resources_df = pd.read_csv("data/resources.csv")

# Generate embeddings
embeddings = model.generate_embeddings(resources_df)

# Save embeddings
model.save_embeddings(embeddings, model.embedding_file_name)

# Later: Load saved embeddings
model.load_embeddings(model.embedding_file_name)

