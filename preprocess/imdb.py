import requests
import pandas as pd

# Credentials pour l'API IGDB
IGDB_CLIENT_ID = "e3u537fwyhd8ph42f719r5pfrpkjhf"
IGDB_ACCESS_TOKEN = "diccwz1n0aion7y9fsf9vs9n11vndv"
IGDB_URL = "https://api.igdb.com/v4/covers"

headers = {
    "Client-ID": IGDB_CLIENT_ID,
    "Authorization": f"Bearer {IGDB_ACCESS_TOKEN}"
}

# Initialiser une liste pour stocker les jeux
all_games = []

# Définir une fonction pour récupérer les jeux
def fetch_games(offset=0):
    data = f"fields name, rating; sort rating desc; limit 500; offset {offset};"
    response = requests.post(IGDB_URL, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}")
        return []

# Boucle pour récupérer les jeux
offset = 0
while True:
    games = fetch_games(offset)
    if not games:
        break
    all_games.extend(games)
    if len(games) < 500:
        break
    offset += 500

# Convertir la liste des jeux en DataFrame
df = pd.DataFrame(all_games)

# Afficher les premières lignes du DataFrame
print(df.head())
