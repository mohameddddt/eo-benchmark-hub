"""Phase 0 exit-criterion tests: register, login, create API token.

Runs against SQLite in-memory so no services are needed: pytest backend/tests
"""
import os

os.environ["DATABASE_URL"] = "sqlite://"
os.environ["AUTO_CREATE_TABLES"] = "false"

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app import database  # noqa: E402
from app.database import Base, get_db  # noqa: E402
from app.main import app  # noqa: E402

engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
TestingSession = sessionmaker(bind=engine)
database.engine = engine
Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

USER = {"email": "ada@example.com", "password": "correct-horse", "name": "Ada"}


def test_register_login_token_flow():
    r = client.post("/auth/register", json=USER)
    assert r.status_code == 201, r.text

    r = client.post("/auth/register", json=USER)
    assert r.status_code == 409  # duplicate email rejected

    r = client.post("/auth/login", json={"email": USER["email"], "password": "wrong"})
    assert r.status_code == 401

    r = client.post("/auth/login", json={"email": USER["email"], "password": USER["password"]})
    assert r.status_code == 200
    jwt_token = r.json()["access_token"]
    auth = {"Authorization": f"Bearer {jwt_token}"}

    r = client.get("/auth/me", headers=auth)
    assert r.status_code == 200 and r.json()["email"] == USER["email"]

    r = client.post("/auth/tokens", json={"name": "colab"}, headers=auth)
    assert r.status_code == 201
    api_token = r.json()["token"]
    assert api_token.startswith("eob_")

    # the API token itself authenticates
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {api_token}"})
    assert r.status_code == 200 and r.json()["email"] == USER["email"]


def test_unauthenticated_rejected():
    assert client.get("/auth/me").status_code == 401
