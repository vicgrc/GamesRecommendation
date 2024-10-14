import pandas as pd
import numpy as np
from time import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import string

t = time()

df = pd.read_csv("data/csv_clean.csv")
df = df.drop('Unnamed: 0', axis=1)

#le playtime de minutes en heures
def min_heure(nom_dataset, colonne):
    nom_dataset[colonne] = nom_dataset[colonne] / 60
min_heure(df, "Average playtime forever")
min_heure(df, "Average playtime two weeks")
min_heure(df, "Median playtime forever")
min_heure(df, "Median playtime two weeks")


#On charge le df des users
df2 = pd.read_csv("data/steam-200k.csv", names=["UserID", "Game", "purchase/play", "Heure_jouee", "0"])

def supprimer_colonne(nom_dataset, colonne):
    nom_dataset = nom_dataset.drop(columns=colonne,inplace = True)

#On garde que les lignes "play" car purchase sert a rien
df2 = df2.drop(df2[df2["purchase/play"] == "purchase"].index)
supprimer_colonne(df2,'0')

#On enlève des caractères spéciaux pour préparer le merge sur le nom des jeux
def remplacement(nom_dataset, colonne, element):
    nom_dataset[colonne] = nom_dataset[colonne].str.replace(element,'')

remplacement(df, 'Name', ':')
remplacement(df, 'Name', '®')
remplacement(df, 'Name', '™')
#En particulier pour Resident Evil (~150 lignes en +)
remplacement(df2, 'Game', ' / Biohazard 6')
remplacement(df2, 'Game', ' / biohazard HD REMASTER')
remplacement(df2, 'Game', ' / Biohazard 5')
remplacement(df2, 'Game', ' / biohazard 4')
remplacement(df2, 'Game', ' / Biohazard 6')
remplacement(df2, 'Game', ' / Biohazard Revelations 2')
remplacement(df2, 'Game', ' / Biohazard Revelations')

#On merge les deux datasets par rapport au nom du jeu
df_clean = df2.merge(df, right_on="Name", left_on="Game")
supprimer_colonne(df_clean, "Name")
df_clean = df_clean.astype({'UserID' : 'int', 'Game' : 'string', 'purchase/play' : 'string', 'Heure_jouee' : 'float', 'Release date' : 'string', 
                            'About the game' : 'string', 'Reviews' : 'string', 'Notes' : 'string', 'Developers' : 'string', 'Publishers' : 'string',
                           'Categories' : 'string', 'Genres' : 'string', 'Tags' : 'string'})
#56789 lignes en commun



#On calcule un score de user en fonction de son temps de jeu par rapport au temps de jeu moyen de tous les users
heure = "Heure_jouee"
temps = "Average playtime forever"
condition = [
    df_clean['Heure_jouee'].astype('float')>= (df_clean["Median playtime forever"]),
   (df_clean['Heure_jouee'].astype('float')>=0.9*df_clean["Median playtime forever"])&(df_clean['Heure_jouee'].astype('float')<1*df_clean["Median playtime forever"]),
   (df_clean['Heure_jouee'].astype('float')>=0.8*df_clean["Median playtime forever"])&(df_clean['Heure_jouee'].astype('float')<0.9*df_clean["Median playtime forever"]),
   (df_clean['Heure_jouee'].astype('float')>=0.7*df_clean["Median playtime forever"])&(df_clean['Heure_jouee'].astype('float')<0.8*df_clean["Median playtime forever"]),
   (df_clean['Heure_jouee'].astype('float')>=0.6*df_clean["Median playtime forever"])&(df_clean['Heure_jouee'].astype('float')<0.7*df_clean["Median playtime forever"]),
   (df_clean['Heure_jouee'].astype('float')>=0.5*df_clean["Median playtime forever"])&(df_clean['Heure_jouee'].astype('float')<0.6*df_clean["Median playtime forever"]),
   (df_clean['Heure_jouee'].astype('float')>=0.4*df_clean["Median playtime forever"])&(df_clean['Heure_jouee'].astype('float')<0.5*df_clean["Median playtime forever"]),
   (df_clean['Heure_jouee'].astype('float')>=0.3*df_clean["Median playtime forever"])&(df_clean['Heure_jouee'].astype('float')<0.4*df_clean["Median playtime forever"]),
   (df_clean['Heure_jouee'].astype('float')>=0.2*df_clean["Median playtime forever"])&(df_clean['Heure_jouee'].astype('float')<0.3*df_clean["Median playtime forever"]),
   (df_clean['Heure_jouee'].astype('float')>=0.1*df_clean["Median playtime forever"])&(df_clean['Heure_jouee'].astype('float')<0.2*df_clean["Median playtime forever"]),
    df_clean['Heure_jouee'].astype('float')>=0
]

values = [5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0]
df_clean['Score'] = np.select(condition,values) 

#On attribue un tag True ou False pour savoir si le jeu est recommandable si le score >= 4
df_clean["Recommandable"] = df_clean['Score'].apply(lambda x: True if x >= 4 else False)

#"Normalisation" des UserID pour qu'ils soint plus petit en taille pour rentrer en mémoire dans le SVD
unique_names = df_clean['Game'].unique()
name_to_id = {name: i + 1 for i, name in enumerate(unique_names)}
df_clean['GameID'] = df_clean['Game'].map(name_to_id).astype(int)

df_clean['UserID'] = df_clean['UserID'].astype('Int64')
unique_ids = df_clean['UserID'].unique()
name_to_id = {name: i + 1 for i, name in enumerate(unique_ids)}
df_clean['UserID'] = df_clean['UserID'].map(name_to_id).astype(int)

# Ajouter le preprocess pour Word2Vec ici
# Lien du kaggle pour le preprocess : https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial#Training-the-model

df_clean.to_csv("data/Dataset.csv", index=False)

print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))