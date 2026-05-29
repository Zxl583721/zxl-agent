import importlib


def test_deepseek_provider_selects_deepseek_chat_model(monkeypatch):
    import config
    import src.model_provider as model_provider

    monkeypatch.setattr(config, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(config, "LLM_MODEL", "deepseek-chat")
    importlib.reload(model_provider)

    model = model_provider.get_chat_model()
    assert model._llm_type == "deepseek-chat"
    assert model.model_name == "deepseek-chat"
