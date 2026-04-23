"""최종 응답 생성(Generator).

system prompt + 검색 메타 + 후보 식당 리스트를 LLM에 넘겨
최종 답변 문장을 생성한다. 세션별 대화 히스토리도 유지.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import SETTINGS
from .prompts import load_system_prompt
from .retriever import simple_retrieve_restaurants


_SESSION_MESSAGES: dict[str, list[BaseMessage]] = {}


def clear_session(session_id: str) -> None:
    _SESSION_MESSAGES.pop(session_id, None)


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=SETTINGS.llm_model,
        temperature=0,
        api_key=SETTINGS.openai_api_key,
    )


def generate_response(
    question: str,
    restaurant_list: list[dict[str, Any]],
    route: str,
    session_id: str = "default",
    route_payload: dict[str, Any] | None = None,
    connector_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_rules = load_system_prompt()
    history = _SESSION_MESSAGES.setdefault(session_id, [])

    route_payload = route_payload or {}
    connector_meta = connector_meta or {}

    retrieved_restaurants = simple_retrieve_restaurants(
        query=question,
        docs=restaurant_list,
        k=SETTINGS.top_k,
    )

    system_prompt = f"""
{prompt_rules}

[Search Route]
{route}

[Search Payload]
{json.dumps(route_payload, ensure_ascii=False, indent=2)}

[Connector Meta]
{json.dumps(connector_meta, ensure_ascii=False, indent=2)}
""".strip()

    rag_input = {
        "question": question,
        "restaurant_list": retrieved_restaurants,
    }

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        *history,
        HumanMessage(content=json.dumps(rag_input, ensure_ascii=False, indent=2)),
    ]

    llm = get_llm()
    result = llm.invoke(messages)
    response = result.content if hasattr(result, "content") else str(result)

    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=response))

    return {
        "answer": response,
        "used_restaurant_list": retrieved_restaurants,
    }