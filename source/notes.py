import numpy as np
import pandas as pd

game_data = pd.read_csv("../data/Dataset.csv", usecols=["GameID", "Game", "Categories", "About the game", "Genres", "Header image"])
game_data = game_data.rename(columns={"About the game": "About_the_game", "Header image": "Header_image"})
game_data = game_data.drop_duplicates()
game_data = game_data.reset_index(drop=True)

df = pd.read_csv("../data/Dataset.csv", usecols=["UserID", "Game", "Score", "GameID"])

ratings_mat = np.ndarray(shape=(np.max(df.GameID.values), np.max(df.UserID.values)))
ratings_mat[df.GameID.values-1, df.UserID.values-1] = df.Score.values

# normalisation de la matrice : on soustrait la moyenne
normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T
# ultérieure normalisation et transposition pour passer à la matrice A "classique"
A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, Vh = np.linalg.svd(A, full_matrices=True)

def clean_liste():
    liste = game_data["Genres"].dropna().unique().tolist()
    new_list = []
    for i in liste:
        if',' in i:
            split_items = i.split(',')
            new_list.extend(split_items)
        else:
            new_list.append(i)
    new_list = list(set(new_list))
    return new_list

def top_cosine_similarity(data, game, top_n=5):
    index = game_data[game_data.Game == game].GameID.values[0] # game id starts from 1 in the dataset
    print(index, game_data[game_data.Game == game].Game.values[0])
    game_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(game_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[1:top_n+1]

def print_similar_games(game_data, game, V):
    print(V.T[:, :25])
    sliced = V.T[:, :25] # utilisation seulement des K features latentes les plus importantes
    top_indexes = top_cosine_similarity(sliced, game, 5)
    jeu_reco = {"Jeux": [], "Description": [], "Genre": [], "Image": []}
    for id in top_indexes + 1:
        jeu_reco["Jeux"].append(game_data[game_data.GameID == id].Game.values[0])
        jeu_reco["Description"].append(game_data[game_data.GameID == id].About_the_game.values[0])
        jeu_reco["Genre"].append(game_data[game_data.GameID == id].Genres.values[0])
        jeu_reco["Image"].append(game_data[game_data.GameID == id].Header_image.values[0])
    return jeu_reco

def df_catego(genre):
    counts = df['GameID'].value_counts()
    tri = df['GameID'].isin(counts[counts >= 20].index)
    tri = df[tri]
    grouped = tri.groupby("GameID", as_index=False).mean("Score")
    df_catego = game_data.join(grouped, how='inner', lsuffix="ID")
    catego = df_catego[df_catego["Genres"] == genre].sort_values(by="Score", ascending=False).head()
    catego = catego.drop("GameIDID", axis=1)
    catego = pd.DataFrame({"Game": catego["Game"], "About_the_game": catego["About_the_game"], "Header_image": catego["Header_image"], "Categories": catego["Categories"], "Genres": catego["Genres"], "GameID": catego["GameID"], "Score": catego["Score"]})
    return catego

def run_machine_learning_model(query):
    global Vh
    liste_jeux = print_similar_games(game_data, query, Vh)
    result = pd.DataFrame({"Jeux": liste_jeux["Jeux"], "Description": liste_jeux["Description"], "Genres": liste_jeux["Genre"], "Image": liste_jeux["Image"]})
    return result