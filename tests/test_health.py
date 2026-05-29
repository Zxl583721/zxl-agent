def test_health_shape(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"ok", "degraded"}
    assert set(data["services"]) >= {"api", "mysql", "redis", "chroma", "celery"}
