import pandas as pd
import streamlit as st
from streamlit_searchbox import st_searchbox
import requests


API_URL = "http://localhost:8000"


def register(username, password):
    response = requests.post(f"{API_URL}/register", json={"username": username, "password": password})
    return response.json() if response.status_code == 200 else None

def login(username, password):
    response = requests.post(f"{API_URL}/token", data={"username": username, "password": password})
    return response.json() if response.status_code == 200 else None

def update_username(new_username, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.put(f"{API_URL}/update_username", json={"new_username": new_username}, headers=headers)
    return response.json() if response.status_code == 200 else None

def update_password(current_password, new_password, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.put(f"{API_URL}/update_password", json={"current_password": current_password, "new_password": new_password}, headers=headers)
    return response.json() if response.status_code == 200 else None

def send_friend_request(friend_username, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{API_URL}/send_friend_request/{friend_username}", headers=headers)
    return response.json() if response.status_code == 200 else None

def get_friend_requests(token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_URL}/friend_requests", headers=headers)
    return response.json() if response.status_code == 200 else None

def accept_friend_request(friend_username, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{API_URL}/accept_friend_request/{friend_username}", headers=headers)
    return response.json() if response.status_code == 200 else None

def reject_friend_request(friend_username, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{API_URL}/reject_friend_request/{friend_username}", headers=headers)
    return response.json() if response.status_code == 200 else None

def get_friends(token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_URL}/friends", headers=headers)
    return response.json() if response.status_code == 200 else None

def get_friend_favorites(friend_username, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_URL}/friend_favorites/{friend_username}", headers=headers)
    return response.json() if response.status_code == 200 else None

def remove_friend(friend_username, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.delete(f"{API_URL}/remove_friend/{friend_username}", headers=headers)
    return response.json() if response.status_code == 200 else None

def recommend_games(query, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{API_URL}/recommend", json={"query": query}, headers=headers)
    return response.json() if response.status_code == 200 else None

def recommend_note(query, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{API_URL}/recommend_note", json={"query": query}, headers=headers)
    return response.json() #if response.status_code == 200 else None

def add_favorite(game, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{API_URL}/add_favorite", json=game, headers=headers)
    return response.json() if response.status_code == 200 else None

def remove_favorite(game_title, token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.delete(f"{API_URL}/remove_favorite/{game_title}", headers=headers)
    return response.json() if response.status_code == 200 else None

def get_favorites(token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_URL}/favorites", headers=headers)
    return response.json() if response.status_code == 200 else None

st.set_page_config(page_title="Recommandation de Jeux IA", page_icon="🎮", layout="wide")

st.title("Trouvez votre prochain jeu préféré avec l'IA 🎮")

# Gestion de l'état de l'utilisateur
if 'token' not in st.session_state:
    st.session_state.token = None
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'recommended_games' not in st.session_state:
    st.session_state.recommended_games = None

# Interface de connexion/inscription
if not st.session_state.token:
    tab1, tab2 = st.tabs(["Connexion", "Inscription"])
    
    with tab1:
        st.header("Connexion")
        username = st.text_input("Nom d'utilisateur", key="login_username")
        password = st.text_input("Mot de passe", type="password", key="login_password")
        if st.button("Se connecter"):
            result = login(username, password)
            if result:
                st.session_state.token = result['access_token']
                st.session_state.favorites = get_favorites(st.session_state.token)
                st.success("Connexion réussie!")
                st.rerun()
            else:
                st.error("Échec de la connexion. Veuillez vérifier vos identifiants.")

    with tab2:
        st.header("Inscription")
        new_username = st.text_input("Nom d'utilisateur", key="register_username")
        new_password = st.text_input("Mot de passe", type="password", key="register_password")
        if st.button("S'inscrire"):
            result = register(new_username, new_password)
            if result:
                st.success("Inscription réussie!")
                # Connexion automatique après inscription
                login_result = login(new_username, new_password)
                if login_result:
                    st.session_state.token = login_result['access_token']
                    st.session_state.favorites = []
                    st.success("Vous êtes maintenant connecté!")
                    st.rerun()
                else:
                    st.error("Inscription réussie, mais erreur lors de la connexion automatique. Veuillez vous connecter manuellement.")
            else:
                st.error("Échec de l'inscription. Ce nom d'utilisateur est peut-être déjà pris.")

else:
    # Interface principale après connexion
    st.sidebar.success("Connecté avec succès!")
    if st.sidebar.button("Se déconnecter"):
        st.session_state.token = None
        st.session_state.favorites = []
        st.session_state.recommended_games = None
        st.rerun()

    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs(["Recommandations par langage naturel", "Recommendations par notes des users", "Profil", "Amis"])

    with tab1:
        # Recommandation de jeux
        query = st.text_input("Décrivez le jeu que vous recherchez")

        if st.button("Rechercher", key=1):
            if query:
                recommended_games = recommend_games(query, st.session_state.token)
                if recommended_games:
                    st.session_state.recommended_games = recommended_games['recommended_games']
                    st.rerun()
                else:
                    st.error("Une erreur s'est produite lors de la récupération des recommandations.")
            else:
                st.warning("Veuillez entrer une description pour obtenir des recommandations.")
        
        if st.session_state.recommended_games:
            st.subheader("Jeux recommandés:")
            for i, game in enumerate(st.session_state.recommended_games, 1):
                similarity_score = game['score'] * 100  # Convert to percentage
                with st.expander(f"{i}. {game['title']} (Similarité: {similarity_score:.2f}%)"):
                    st.image(game['url'], width=300)
                    st.write(game['description'])
                    st.progress(game['score'])
                    if st.button(f"Ajouter aux favoris", key="1"f"fav_{i}"):
                        result = add_favorite({"title": game['title'], "description": game['description']}, st.session_state.token)
                        if result:
                            st.success(result['message'])
                            st.session_state.favorites = get_favorites(st.session_state.token)
                        else:
                            st.error("Erreur lors de l'ajout aux favoris.")

    with tab2:
        game_data = pd.read_csv("../data/Dataset.csv", usecols=["Game"])
        game_data = game_data.drop_duplicates()
        game_data = game_data.reset_index(drop=True)

        def search(searchterm: str) -> list[tuple[str, any]]:
            liste = []
            for i in range(len(game_data[game_data["Game"].str.contains(searchterm, case=False)].values)):
                liste.append(str(game_data[game_data["Game"].str.contains(searchterm, case=False)].values[i])[2:-2])
            return liste

        query = st_searchbox(search)
        
        if st.button("Rechercher", key=2):
            result = recommend_note(query, st.session_state.token)
            st.subheader(f"Si vous avez aimé ce jeu vous allez aimer :sunglasses: :")
            url_dict = pd.DataFrame(result["result"])
            #On itere dans le df pour afficher les jeux
            for i, game in url_dict.iterrows():
                with st.expander(f"{i}. {url_dict['Jeux'][i]}"):
                    st.image(url_dict["Image"][i])
                    st.write(f"Genre : {url_dict['Genres'][i]}")
                    st.write(f"Description : {url_dict['Description'][i]}")
                    if st.button(f"Ajouter aux favoris", key="2"f"fav_{i}"):
                        result = add_favorite({"title": game['Jeux'], "description": game['Description']}, st.session_state.token)
                        if result:
                            st.success(result['message'])
                            st.session_state.favorites = get_favorites(st.session_state.token)
                        else:
                            st.error("Erreur lors de l'ajout aux favoris.")

    with tab3:
        # Profil utilisateur
        st.header("Votre profil")

        # Affichage des favoris
        with st.expander("Vos jeux favoris", expanded=True):
            st.session_state.favorites = get_favorites(st.session_state.token)
            if st.session_state.favorites:
                for i, game in enumerate(st.session_state.favorites, 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{i}. {game['title']}")
                        st.write(f"   Description: {game['description']}")
                    with col2:
                        if st.button("Supprimer", key=f"remove_{i}"):
                            result = remove_favorite(game['title'], st.session_state.token)
                            if result:
                                st.success(result['message'])
                                st.session_state.favorites = get_favorites(st.session_state.token)
                                st.rerun()
                            else:
                                st.error("Erreur lors de la suppression du favori.")
                    st.write("---")
            else:
                st.info("Vous n'avez pas encore de jeux favoris.")

        # Changer le nom d'utilisateur
        with st.expander("Changer le nom d'utilisateur"):
            new_username = st.text_input("Nouveau nom d'utilisateur")
            if st.button("Changer le nom d'utilisateur"):
                if new_username:
                    result = update_username(new_username, st.session_state.token)
                    if result:
                        st.success(result['message'])
                        st.warning("Veuillez vous reconnecter avec votre nouveau nom d'utilisateur.")
                        st.session_state.token = None
                        st.rerun()
                    else:
                        st.error("Erreur lors du changement de nom d'utilisateur.")
                else:
                    st.warning("Veuillez entrer un nouveau nom d'utilisateur.")

        # Changer le mot de passe
        with st.expander("Changer le mot de passe"):
            current_password = st.text_input("Mot de passe actuel", type="password")
            new_password = st.text_input("Nouveau mot de passe", type="password")
            confirm_new_password = st.text_input("Confirmer le nouveau mot de passe", type="password")

            if st.button("Changer le mot de passe"):
                if current_password and new_password and confirm_new_password:
                    if new_password == confirm_new_password:
                        result = update_password(current_password, new_password, st.session_state.token)
                        if result:
                            st.success(result['message'])
                            st.warning("Veuillez vous reconnecter avec votre nouveau mot de passe.")
                            st.session_state.token = None
                            st.rerun()
                        else:
                            st.error("Erreur lors du changement de mot de passe.")
                    else:
                        st.error("Les nouveaux mots de passe ne correspondent pas.")
                else:
                    st.warning("Veuillez remplir tous les champs pour changer le mot de passe.")

    with tab4:
        # Système d'amis
        st.header("Amis")
        # Envoi de demande d'ami
        col1, col2 = st.columns([1, 1])
        with col1:
            new_friend = st.text_input("Envoyer une demande d'ami", label_visibility="collapsed")
        with col2:
            if st.button("Envoyer la demande", use_container_width=True):
                if new_friend:
                    result = send_friend_request(new_friend, st.session_state.token)
                    if result:
                        st.success(result['message'])
                    else:
                        st.error("Impossible d'envoyer la demande d'ami.")
                else:
                    st.warning("Veuillez entrer un nom d'utilisateur.")

        # Affichage des demandes d'amitié
        friend_requests = get_friend_requests(st.session_state.token)
        if friend_requests:
            st.subheader("Demandes d'amitié en attente:")
            for request in friend_requests:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(request)
                with col2:
                    if st.button("Accepter", key=f"accept_{request}", use_container_width=True):
                        result = accept_friend_request(request, st.session_state.token)
                        if result:
                            st.success(result['message'])
                            st.rerun()
                        else:
                            st.error("Erreur lors de l'acceptation de la demande.")
                with col3:
                    if st.button("Refuser", key=f"reject_{request}", use_container_width=True):
                        result = reject_friend_request(request, st.session_state.token)
                        if result:
                            st.success(result['message'])
                            st.rerun()
                        else:
                            st.error("Erreur lors du refus de la demande.")

        # Liste d'amis
        friends = get_friends(st.session_state.token)
        if friends:
            st.subheader("Liste d'amis:")
            for friend in friends:
                with st.expander(f"{friend['username']}"):
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.write(f"Ami: {friend['username']}")
                    with col2:
                        if st.button("Retirer", key=f"remove_friend_{friend['username']}", use_container_width=True):
                            result = remove_friend(friend['username'], st.session_state.token)
                            if result:
                                st.success(result['message'])
                                st.rerun()
                            else:
                                st.error("Erreur lors de la suppression de l'ami.")
                    
                    # Affichage des jeux favoris de l'ami
                    friend_favorites = get_friend_favorites(friend['username'], st.session_state.token)
                    if friend_favorites:
                        st.write("Jeux favoris :")
                        for game in friend_favorites:
                            st.write(f"- {game['title']}")
                    else:
                        st.write("Cet ami n'a pas encore de jeux favoris.")
        else:
            st.info("Vous n'avez pas encore d'amis.")

    # Ajoutez ce style CSS personnalisé pour réduire la taille des boutons
    st.markdown("""
    <style>
        .stButton>button {
            padding: 0.2rem 0.5rem;
            font-size: 0.8rem;
        }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.header("À propos")
st.sidebar.info(
    "Cette application utilise un modèle d'intelligence artificielle avancé (Sentence Transformers) "
    "pour recommander des jeux en fonction de votre description. Le modèle comprend le contexte "
    "et la sémantique de votre requête pour fournir des recommandations plus précises."
)