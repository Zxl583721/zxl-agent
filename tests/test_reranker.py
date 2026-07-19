import pytest

from src.reranker import BGEReranker, LightweightReranker, build_reranker


def test_lightweight_is_the_default_reranker():
    selection = build_reranker()

    assert selection.requested_provider == "lightweight"
    assert selection.active_provider == "lightweight"
    assert isinstance(selection.reranker, LightweightReranker)
    assert selection.setup_error == ""


def test_bge_falls_back_to_lightweight_with_actionable_error(monkeypatch, tmp_path):
    def unavailable(*args, **kwargs):
        raise RuntimeError("未安装 FlagEmbedding")

    monkeypatch.setattr("src.reranker.BGEReranker", unavailable)
    selection = build_reranker("bge", model_path=tmp_path / "missing-model")

    assert selection.requested_provider == "bge"
    assert selection.active_provider == "lightweight"
    assert isinstance(selection.reranker, LightweightReranker)
    assert "已回退到 lightweight" in selection.setup_error
    assert "BGE_RERANKER_MODEL_PATH" in selection.setup_error


def test_bge_selection_keeps_bge_when_model_load_succeeds(monkeypatch, tmp_path):
    class FakeBGE:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr("src.reranker.BGEReranker", FakeBGE)
    selection = build_reranker(
        "bge",
        model_path=tmp_path / "model",
        use_fp16=True,
        batch_size=4,
    )

    assert selection.active_provider == "bge"
    assert isinstance(selection.reranker, FakeBGE)
    assert selection.reranker.kwargs == {"model_path": tmp_path / "model", "use_fp16": True, "batch_size": 4}


def test_invalid_provider_fails_with_config_hint():
    with pytest.raises(ValueError, match="RERANKER_PROVIDER"):
        build_reranker("unknown")


def test_bge_reranker_preserves_retrieval_metadata(monkeypatch, tmp_path):
    class FakeFlagReranker:
        def __init__(self, model_path, use_fp16):
            self.model_path = model_path
            self.use_fp16 = use_fp16

        def compute_score(self, pairs, batch_size):
            assert batch_size == 2
            return [0.1, 0.9]

    class FakeFlagEmbedding:
        FlagReranker = FakeFlagReranker

    monkeypatch.setitem(__import__("sys").modules, "FlagEmbedding", FakeFlagEmbedding)
    model_path = tmp_path / "model"
    model_path.mkdir()
    reranker = BGEReranker(model_path, batch_size=2)
    chunks = [
        {"text": "低相关", "score": 0.8, "metadata": {"id": "a"}},
        {"text": "高相关", "score": 0.7, "metadata": {"id": "b"}},
    ]

    result = reranker.rerank("问题", chunks, top_k=2)

    assert [chunk["metadata"]["id"] for chunk in result] == ["b", "a"]
    assert result[0]["bge_rerank_score"] == 0.9
    assert result[0]["score"] == 0.7
