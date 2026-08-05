import io

import pytest
from fastapi.testclient import TestClient

from api.main import app
from modules.feature_extraction import FEATURE_NAMES, extract_features_from_df


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_status_lists_all_three(client):
    r = client.get("/models/status")
    assert r.status_code == 200
    body = r.json()
    assert {m["model"] for m in body["models"]} == {"MLR", "Fuzzy Logic", "CNN"}


def test_openapi_schema_is_served(client):
    assert client.get("/openapi.json").status_code == 200


def test_predict_happy_path(client, vibration_csv):
    r = client.post("/predict", files={"file": ("part.csv", vibration_csv, "text/csv")})
    assert r.status_code == 200

    body = r.json()
    assert len(body["features"]) == len(FEATURE_NAMES)
    assert len(body["channels_found"]) == 3
    assert body["latency_ms"] > 0
    assert body["category"] in {"Smooth", "Average", "Rough"}

    ran = [p for p in body["predictions"] if p["available"]]
    assert ran, "expected at least one model to produce a prediction"
    assert all(p["ra"] is not None for p in ran)


def test_predict_rejects_file_without_channels(client):
    bad = io.BytesIO(b"alpha,beta\n1,2\n3,4\n")
    r = client.post("/predict", files={"file": ("bad.csv", bad, "text/csv")})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_input"


def test_predict_rejects_unsupported_extension(client):
    r = client.post(
        "/predict", files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    )
    assert r.status_code == 400
    assert ".txt" in r.json()["detail"]


def test_predict_rejects_unparseable_csv(client):
    junk = io.BytesIO(b"\x00\x01\x02 not a csv at all")
    r = client.post("/predict", files={"file": ("broken.csv", junk, "text/csv")})
    assert r.status_code == 400


def test_predict_from_features_matches_file_upload(client, vibration_df, vibration_csv):
    features = extract_features_from_df(vibration_df)
    payload = {"features": {k: float(v) for k, v in features.iloc[0].items()}}

    from_json = client.post("/predict/features", json=payload).json()
    from_file = client.post(
        "/predict", files={"file": ("part.csv", vibration_csv, "text/csv")}
    ).json()

    json_ra = {p["model"]: p["ra"] for p in from_json["predictions"]}
    file_ra = {p["model"]: p["ra"] for p in from_file["predictions"]}
    assert json_ra == file_ra


def test_predict_from_features_rejects_incomplete_payload(client):
    r = client.post("/predict/features", json={"features": {"Channel 1 RMS": 1.0}})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_input"


def test_error_responses_never_leak_tracebacks(client):
    r = client.post("/predict", files={"file": ("x.txt", io.BytesIO(b"a"), "text/plain")})
    assert "Traceback" not in r.text
    assert set(r.json()) == {"error", "detail"}
