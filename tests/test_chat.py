import pytest

from langchain_core.messages import HumanMessage
from chat import get_chat_model


@pytest.fixture
def fake_chat_model():
    chat_model = get_chat_model()
    yield chat_model
    get_chat_model.cache_clear()


@pytest.mark.parametrize("answer", ["What gives your life meaning?", "42"])
def test_valid_post_returns_answer_from_llm(answer, client, fake_chat_model):
    fake_chat_model.responses.append(answer)

    question = "What is the meaning of life?"
    response = client.post(
        "/api/chat/",
        {"question": question},
        content_type="application/json",
    )
    assert response.status_code == 200

    result = response.json()
    expected = {"answer": answer}

    assert result == expected


@pytest.mark.parametrize("question", ["42", "What do you think?"])
def test_messages_sent_to_llm(client, fake_chat_model, question):
    fake_chat_model.responses.append("42")
    response = client.post(
        "/api/chat/",
        {"question": question},
        content_type="application/json",
    )
    assert response.status_code == 200

    expected = [HumanMessage(content=question)]
    result = fake_chat_model.messages

    assert result == expected
