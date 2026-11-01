"""End-to-end API behaviour: auth, the submission pipeline, publishing and the
leaderboard rules (collapsing, ranking, CI tiers), quotas and visibility."""
from conftest import predictions, signup, submit

CARD = {"architecture": "ResNet-18", "family": "CNN", "num_params": 11_000_000}


def publish(client, headers, sid):
    assert client.put(f"/api/submissions/{sid}/model-card", json=CARD, headers=headers).status_code == 200
    r = client.post(f"/api/submissions/{sid}/publish", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def test_register_login_token_flow(client):
    user = {"email": "ada@example.com", "password": "correct-horse", "name": "Ada"}
    assert client.post("/api/auth/register", json=user).status_code == 201
    assert client.post("/api/auth/register", json=user).status_code == 409
    assert client.post("/api/auth/login", json={**user, "password": "nope-nope"}).status_code == 401
    auth = {"Authorization": "Bearer " + client.post("/api/auth/login", json=user).json()["access_token"]}
    assert client.get("/api/auth/me", headers=auth).json()["email"] == user["email"]
    token = client.post("/api/auth/tokens", json={"name": "kaggle"}, headers=auth).json()["token"]
    assert token.startswith("eob_")
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    assert client.patch("/api/auth/me", json={"affiliation": "ESA"}, headers=auth).json()["affiliation"] == "ESA"
    assert client.get("/api/auth/me").status_code == 401


def test_submission_scored_with_confidence_interval(client, benchmark):
    h = signup(client)
    r = submit(client, h, benchmark, predictions(3), name="ResNet-18")
    assert r.status_code == 201, r.text
    sub = r.json()
    assert sub["status"] == "COMPLETED", sub["error_report"]
    acc = next(m for m in sub["results"] if m["metric_name"] == "accuracy")
    assert acc["value"] == 0.9
    assert acc["ci_low"] <= 0.9 <= acc["ci_high"]
    assert sub["details"]["confusion_matrix"]["labels"] == ["Forest", "River", "Highway"]


def test_invalid_file_rejected_and_free(client, benchmark):
    h = signup(client)
    for _ in range(4):  # quota is 3; rejected uploads must not consume it
        r = submit(client, h, benchmark, b"sample_id,predicted_label\nghost,Lava\n")
        assert r.json()["status"] == "REJECTED"
    assert submit(client, h, benchmark, predictions(0)).json()["status"] == "COMPLETED"
    submit(client, h, benchmark, predictions(1))
    submit(client, h, benchmark, predictions(2))
    assert submit(client, h, benchmark, predictions(0)).status_code == 429


def test_wrong_extension_and_missing_ground_truth(client, benchmark):
    h = signup(client)
    assert submit(client, h, benchmark, b"x", filename="p.zip").status_code == 422
    from app.database import SessionLocal
    from app.models import Benchmark

    db = SessionLocal()
    db.add(Benchmark(slug="empty", name="Empty", task_type="scene_classification", classes=["a"]))
    db.commit()
    db.close()
    assert submit(client, h, "empty", predictions(0)).status_code == 409


def test_publish_requires_card_and_leaderboard_ranks(client, benchmark):
    ada, bob = signup(client), signup(client, "bob@example.com", "Bob")
    a1 = submit(client, ada, benchmark, predictions(6), name="Net-A").json()["id"]
    a2 = submit(client, ada, benchmark, predictions(3), name="Net-A").json()["id"]   # same model, better
    a3 = submit(client, ada, benchmark, predictions(9), name="Net-B").json()["id"]   # distinct model
    b1 = submit(client, bob, benchmark, predictions(0), name="Perfect").json()["id"]

    assert client.post(f"/api/submissions/{a1}/publish", headers=ada).status_code == 409  # no card
    for sid, h in ((a1, ada), (a2, ada), (a3, ada), (b1, bob)):
        publish(client, h, sid)

    board = client.get(f"/api/benchmarks/{benchmark}/leaderboard").json()
    assert [e["model_name"] for e in board] == ["Perfect", "Net-A", "Net-B"]  # Net-A collapsed to best
    assert board[1]["submission_id"] == a2 and board[1]["score"] == 0.9
    assert [e["rank"] for e in board] == [1, 2, 3]
    assert board[0]["tier"] == 1 and board[0]["card"]["family"] == "CNN"
    assert len(client.get(f"/api/benchmarks/{benchmark}/leaderboard?all=true").json()) == 4

    detail = client.get(f"/api/submissions/{a2}").json()          # public once published
    assert detail["rank"] == 2 and detail["n_ranked"] == 4

    bm = client.get(f"/api/benchmarks/{benchmark}").json()
    assert bm["ready"] and bm["n_entries"] == 3 and bm["top"]["model_name"] == "Perfect"
    assert any(m["name"] == "ece" for m in bm["metrics"])

    stats = client.get("/api/stats").json()
    assert stats["models"] == 3 and stats["researchers"] == 2 and stats["leaders"][0]["score"] == 1.0
    profile = client.get(f"/api/users/{board[0]['user']['id']}").json()
    assert profile["entries"][0]["rank"] == 1


def test_tiers_group_statistically_tied_entries(client, benchmark):
    h = signup(client)
    ids = [submit(client, h, benchmark, predictions(n), name=f"m{n}").json()["id"] for n in (0, 1, 15)]
    for sid in ids:
        publish(client, h, sid)
    board = client.get(f"/api/benchmarks/{benchmark}/leaderboard").json()
    # 100% vs 96.7% on 30 samples: CIs overlap -> same tier; 50% is clearly worse
    assert [e["tier"] for e in board] == [1, 1, 2]


def test_private_submissions_are_private(client, benchmark):
    ada, bob = signup(client), signup(client, "bob@example.com", "Bob")
    sid = submit(client, ada, benchmark, predictions(0)).json()["id"]
    assert client.get(f"/api/submissions/{sid}", headers=bob).status_code == 404
    assert client.get(f"/api/submissions/{sid}").status_code == 404
    assert client.get(f"/api/submissions/{sid}", headers=ada).status_code == 200
    assert client.delete(f"/api/submissions/{sid}", headers=bob).status_code == 404
    assert client.delete(f"/api/submissions/{sid}", headers=ada).status_code == 204
    assert client.get("/api/users/me/submissions", headers=ada).json() == []


def test_spa_fallback_and_api_404(client):
    from conftest import TMP

    (TMP / "dist").mkdir(exist_ok=True)
    (TMP / "dist" / "index.html").write_text("<div id=root></div>")
    assert "root" in client.get("/benchmarks/eurosat").text
    assert client.get("/api/nope").status_code == 404
