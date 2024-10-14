import pandas as pd
import re
from sentence_transformers import SentenceTransformer

# Charger le dataset
df = pd.read_csv('gamesIGDB.csv')  # Remplacez par le nom de votre fichier

def nettoyer_texte(texte):
    if pd.isna(texte):
        return ''
    # Supprimer les caractères spéciaux et les chiffres
    texte = re.sub(r'[^a-zA-Z\s]', '', str(texte))
    
    # Convertir en minuscules
    texte = texte.lower()
    
    # Supprimer les espaces multiples
    texte = re.sub(r'\s+', ' ', texte).strip()
    
    return texte

# Afficher le nombre de lignes avant le nettoyage
print(f"Nombre de lignes avant le nettoyage : {len(df)}")

# Appliquer la fonction de nettoyage à la colonne 'summary'
df['summary_clean'] = df['summary'].apply(nettoyer_texte)

# Supprimer les lignes où 'summary_clean' est vide
df = df[df['summary_clean'] != '']

# Réinitialiser l'index
df = df.reset_index(drop=True)

# Afficher le nombre de lignes après le nettoyage
print(f"Nombre de lignes après le nettoyage : {len(df)}")

# Afficher quelques exemples pour vérification
print(df['summary_clean'].head())

# Chargement du modèle Sentence Transformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Chargement des données de jeux depuis cleangames.csv

# Fonction pour encoder un résumé
def encode_summary(summary):
    return model.encode(summary)

# Encodage des résumés
print("Encodage des résumés...")
df['encoded_summary'] = df['summary_clean'].apply(encode_summary)

# Conversion des encodages en chaînes de caractères pour le stockage CSV
df['encoded_summary'] = df['encoded_summary'].apply(lambda x: ','.join(map(str, x)))

# Sauvegarde du DataFrame enrichi dans un nouveau fichier CSV
print("Sauvegarde des données encodées...")
df.to_csv('encoded_games.csv', index=False)

print("Prétraitement terminé. Les données encodées sont sauvegardées dans 'encoded_games.csv'.")