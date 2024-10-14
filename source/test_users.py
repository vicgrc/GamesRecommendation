import pytest
from fastapi.testclient import TestClient
from backend import app

client = TestClient(app)

@pytest.fixture(scope="module")
def auth_token():
    response = client.post("/token", data={"username": "testuser", "password": "testpassword"})
    return response.json()["access_token"]

def test_update_username(auth_token):
    response = client.put("/update_username", 
                          json={"new_username": "newusername"},
                          headers={"Authorization": f"Bearer {auth_token}"})
    assert response.status_code == 200
    assert "message" in response.json()

def test_update_password(auth_token):
    response = client.put("/update_password", 
                          json={"current_password": "testpassword", "new_password": "newpassword"},
                          headers={"Authorization": f"Bearer {auth_token}"})
    assert response.status_code == 200
    assert "message" in response.json()

def test_get_user_info(auth_token):
    response = client.get("/user_info", headers={"Authorization": f"Bearer {auth_token}"})
    assert response.status_code == 200
    assert "username" in response.json()