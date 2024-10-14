from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from notes import print_similar_games, run_machine_learning_model
import torch
import jwt
import pandas as pd
import numpy as np
import bcrypt

app = FastAPI()

# Configuration de l'authentification
SECRET_KEY = "votre_clé_secrète_ici"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Structure de données en mémoire pour stocker les utilisateurs, les amitiés, les favoris et les demandes d'amitié
users: Dict[str, Dict] = {}
friendships: Dict[str, List[str]] = {}
favorites: Dict[str, List[Dict]] = {}
friend_requests: Dict[str, List[str]] = {}

# Modèles de données
class UserCreate(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    username: str

class UsernameUpdate(BaseModel):
    new_username: str

class PasswordUpdate(BaseModel):
    current_password: str
    new_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class RecommendRequest(BaseModel):
    query: str

class Game(BaseModel):
    title: str
    description: str

class PromptRequest(BaseModel):
    prompt: str

# Configuration de la sécurité
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Définir le dispositif à utiliser (GPU si disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Chargement du modèle Sentence Transformer avec configuration explicite du tokenizer
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.model_max_length = 512
tokenizer.clean_up_tokenization_spaces = True

model = SentenceTransformer(model_name)
model.tokenizer = tokenizer
model.to(device)

# Chargement des données de jeux depuis encoded_games.csv avec les résumés préencodés
print("Chargement des données encodées...")
df = pd.read_csv('../data/encodedgames.csv')
df['encoded_summary'] = df['encoded_summary'].apply(lambda x: np.fromstring(x[1:-1], sep=', ', dtype=np.float32))

# Optimisation : création d'un seul numpy array avant la conversion en tensor
encoded_summaries = np.array(df['encoded_summary'].tolist(), dtype=np.float32)
games_embeddings = torch.from_numpy(encoded_summaries).to(device)

games = df[['name', 'summary', 'url']].to_dict('records')

# Fonctions utilitaires
def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password)

def get_password_hash(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def get_user(username: str):
    return users.get(username)

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user or not verify_password(password, user['hashed_password']):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

# Routes
@app.post("/register", response_model=UserOut)
def register(user: UserCreate):
    if user.username in users:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    users[user.username] = {"username": user.username, "hashed_password": hashed_password}
    friendships[user.username] = []
    favorites[user.username] = []
    friend_requests[user.username] = []
    return UserOut(username=user.username)

@app.put("/update_username")
async def update_username(username_update: UsernameUpdate, current_user: dict = Depends(get_current_user)):
    if username_update.new_username in users:
        raise HTTPException(status_code=400, detail="Ce nom d'utilisateur est déjà pris")
    
    # Mettre à jour le nom d'utilisateur dans toutes les structures de données
    old_username = current_user['username']
    users[username_update.new_username] = users.pop(old_username)
    users[username_update.new_username]['username'] = username_update.new_username
    
    friendships[username_update.new_username] = friendships.pop(old_username)
    favorites[username_update.new_username] = favorites.pop(old_username)
    friend_requests[username_update.new_username] = friend_requests.pop(old_username)
    
    # Mettre à jour les listes d'amis des autres utilisateurs
    for friends_list in friendships.values():
        if old_username in friends_list:
            friends_list[friends_list.index(old_username)] = username_update.new_username
    
    return {"message": "Nom d'utilisateur mis à jour avec succès"}

@app.put("/update_password")
async def update_password(password_update: PasswordUpdate, current_user: dict = Depends(get_current_user)):
    if not verify_password(password_update.current_password, current_user['hashed_password']):
        raise HTTPException(status_code=400, detail="Mot de passe actuel incorrect")
    
    new_hashed_password = get_password_hash(password_update.new_password)
    users[current_user['username']]['hashed_password'] = new_hashed_password
    
    return {"message": "Mot de passe mis à jour avec succès"}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user['username']}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/send_friend_request/{friend_username}")
async def send_friend_request(friend_username: str, current_user: dict = Depends(get_current_user)):
    if friend_username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    if friend_username in friendships[current_user['username']]:
        raise HTTPException(status_code=400, detail="Already friends")
    if current_user['username'] in friend_requests[friend_username]:
        raise HTTPException(status_code=400, detail="Friend request already sent")
    friend_requests[friend_username].append(current_user['username'])
    return {"message": f"Friend request sent to {friend_username}"}

@app.get("/friend_requests", response_model=List[str])
async def get_friend_requests(current_user: dict = Depends(get_current_user)):
    return friend_requests[current_user['username']]

@app.post("/accept_friend_request/{friend_username}")
async def accept_friend_request(friend_username: str, current_user: dict = Depends(get_current_user)):
    if friend_username not in friend_requests[current_user['username']]:
        raise HTTPException(status_code=404, detail="Friend request not found")
    friend_requests[current_user['username']].remove(friend_username)
    friendships[current_user['username']].append(friend_username)
    friendships[friend_username].append(current_user['username'])
    return {"message": f"Friend request from {friend_username} accepted"}

@app.post("/reject_friend_request/{friend_username}")
async def reject_friend_request(friend_username: str, current_user: dict = Depends(get_current_user)):
    if friend_username not in friend_requests[current_user['username']]:
        raise HTTPException(status_code=404, detail="Friend request not found")
    friend_requests[current_user['username']].remove(friend_username)
    return {"message": f"Friend request from {friend_username} rejected"}

@app.get("/friends", response_model=List[UserOut])
async def get_friends(current_user: dict = Depends(get_current_user)):
    return [UserOut(username=friend) for friend in friendships[current_user['username']]]

@app.get("/friend_favorites/{friend_username}")
async def get_friend_favorites(friend_username: str, current_user: dict = Depends(get_current_user)):
    if friend_username not in friendships[current_user['username']]:
        raise HTTPException(status_code=404, detail="Friend not found")
    return favorites.get(friend_username, [])

@app.delete("/remove_friend/{friend_username}")
async def remove_friend(friend_username: str, current_user: dict = Depends(get_current_user)):
    if friend_username not in friendships[current_user['username']]:
        raise HTTPException(status_code=404, detail="Friend not found")
    friendships[current_user['username']].remove(friend_username)
    friendships[friend_username].remove(current_user['username'])
    return {"message": f"Friend {friend_username} removed successfully"}

@app.post("/add_favorite")
async def add_favorite(game: Game, current_user: dict = Depends(get_current_user)):
    if game.dict() not in favorites[current_user['username']]:
        favorites[current_user['username']].append(game.dict())
        return {"message": f"Le jeu {game.title} a été ajouté à vos favoris"}
    else:
        return {"message": f"Le jeu {game.title} est déjà dans vos favoris"}

@app.delete("/remove_favorite/{game_title}")
async def remove_favorite(game_title: str, current_user: dict = Depends(get_current_user)):
    user_favorites = favorites[current_user['username']]
    for game in user_favorites:
        if game['title'] == game_title:
            user_favorites.remove(game)
            return {"message": f"Le jeu {game_title} a été supprimé de vos favoris"}
    raise HTTPException(status_code=404, detail="Game not found in favorites")

@app.get("/favorites", response_model=List[Game])
async def get_favorites(current_user: dict = Depends(get_current_user)):
    return favorites[current_user['username']]

@app.post("/recommend")
async def recommend_games(request: RecommendRequest, current_user: dict = Depends(get_current_user)):
    query_embedding = model.encode(request.query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, games_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(10, len(games)))
    
    recommended_games = [
        {
            "title": games[idx]['name'],
            "description": games[idx]['summary'],
            "url": str(games[idx]['url']),
            "score": score.item()
        }
        for score, idx in zip(top_results.values, top_results.indices)
    ]
    
    return {"recommended_games": recommended_games}

#recommandations par notes des users, comme sur le 1er projet
@app.post("/recommend_note")
async def recommend_note(request: RecommendRequest, current_user: dict = Depends(get_current_user)):
    result = run_machine_learning_model(request.query)
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)