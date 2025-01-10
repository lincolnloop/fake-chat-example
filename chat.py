import os
from functools import lru_cache

from django.conf import settings
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from nanodjango import Django
from pydantic.fields import Field

app = Django(
    CHAT_MODEL=os.environ.get("CHAT_MODEL", "ollama"),
)


class FakeChatModel(FakeListChatModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    responses: list[str] = Field(default_factory=list)

    def _call(self, messages: list[BaseMessage], *args, **kwargs) -> str:
        self.messages.extend(messages)
        return super()._call(messages, *args, **kwargs)


@lru_cache
def get_chat_model():
    setting = settings.CHAT_MODEL
    if setting == "fake":
        return FakeChatModel()
    if setting == "ollama":
        return ChatOllama(
            model="llama3.2",
            temperature=0,
        )
    # ... extend / modify to use LLM of choice.

    msg = f"CHAT_MODEL={setting} not implemented."
    raise NotImplementedError(msg)


class RequestData(app.ninja.Schema):
    question: str


@app.api.post("/chat/")
def chat(request, data: RequestData):
    chat_model = get_chat_model()
    ai_message = chat_model.invoke(data.question)
    answer = ai_message.content
    return {"answer": answer}
