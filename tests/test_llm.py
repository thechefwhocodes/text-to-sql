from src.llm import LLM
from src.models import GPT_5_4, GPT_OSS_120B, get_model


def test_client_for_reuses_client_for_the_same_model_config():
    llm = LLM()
    config = get_model(GPT_OSS_120B)

    first = llm._client_for(config)
    second = llm._client_for(config)

    assert first is second


def test_client_for_creates_separate_clients_per_provider():
    llm = LLM()

    fireworks_client = llm._client_for(get_model(GPT_OSS_120B))
    openai_client = llm._client_for(get_model(GPT_5_4))

    assert fireworks_client is not openai_client
