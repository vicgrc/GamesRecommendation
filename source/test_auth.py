import pytest
from fastapi.testclient import TestClient
from backend import app  # Assurez-vous que c'est le bon chemin vers votre application FastAPI

client = TestClient(app)

def test_register():
    response = client.post("/register", json={"username": "testuser", "password": "testpassword"})
    assert response.status_code == 200
    assert "username" in response.json()

def test_login_valid():
    response = client.post("/token", data={"username": "testuser", "password": "testpassword"})
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_login_invalid():
    response = client.post("/token", data={"username": "testuser", "password": "wrongpassword"})
    assert response.status_code == 401

# def test_protected_route():
#     # D'abord, obtenez un token valide
#     login_response = client.post("/token", data={"username": "testuser", "password": "testpassword"})
#     token = login_response.json()["access_token"]
    
#     # Utilisez le token pour accéder à une route protégée
#     response = client.get("/protected_route", headers={"Authorization": f"Bearer {token}"})
#     assert response.status_code == 200

# def test_protected_route_invalid_token():
#     response = client.get("/protected_route", headers={"Authorization": "Bearer invalid_token"})
#     assert response.status_code == 401