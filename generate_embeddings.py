import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Chargement
df = pd.read_csv("mail_data.csv")
messages = df['Message'].fillna('').astype(str).tolist()

# Modèle de transformation (384 dimensions)
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Encodage en cours... Patientez un instant.")
mail_embeddings = model.encode(messages, show_progress_bar=True)

# Sauvegarde binaire
np.save("mail_embeddings.npy", mail_embeddings)
print("Fichier 'mail_embeddings.npy' généré !")