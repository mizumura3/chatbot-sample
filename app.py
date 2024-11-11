from typing import TypedDict, Annotated, List
from uuid import uuid4

import chainlit as cl
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import add_messages, StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# AWS Bedrock setup
model = ChatBedrockConverse(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    temperature=0.7,
    max_tokens=None,
)

# システムメッセージを定義
SYSTEM_MESSAGE = SystemMessage(content="""
あなたは優秀な AI エージェントです。キテレツ大百科のコロ助になりきってください。
""")

class State(TypedDict):
    """The state of the workflow."""
    messages: Annotated[List[BaseMessage], add_messages]

async def ai_assistant(state: State) -> State:
    messages = state["messages"]
    prompt = ChatPromptTemplate.from_messages([
        SYSTEM_MESSAGE,
        MessagesPlaceholder(variable_name="messages"),
    ])
    chain: Runnable = prompt | model | StrOutputParser()
    response = await chain.ainvoke({"messages": messages})

    return State(
        messages=messages + [AIMessage(content=response)],
    )

def create_workflow() -> CompiledStateGraph:
    workflow = StateGraph(State)
    workflow.add_node("ai_assistant", ai_assistant)
    workflow.add_edge("ai_assistant", END)
    workflow.set_entry_point("ai_assistant")

    # 会話内容をメモリに保存する設定
    checkpointer = MemorySaver()
    cl.user_session.set("checkpointer", checkpointer)
    thread_id = uuid4().hex
    # config = RunnableConfig(configurable={"thread_id": thread_id},)
    config = {"configurable": {"thread_id": thread_id}}
    cl.user_session.set("config", config)

    app = workflow.compile(checkpointer=checkpointer)
    print(f"Workflow graph: {app.get_graph().draw_mermaid()}")

    return app

@cl.on_chat_start
async def on_chat_start():
    app = create_workflow()
    cl.user_session.set("app", app)
    cl.user_session.set("inputs", {"messages": []})


@cl.on_message
async def on_message(message: cl.Message):
    app: Runnable = cl.user_session.get("app")
    inputs = cl.user_session.get("inputs")
    inputs["messages"].append(HumanMessage(content=message.content))
    config = cl.user_session.get("config")

    ui_message = None

    async for output in app.astream_events(inputs, config=config, version="v1"):
        print(output)
        text = await extract_text_from_output(output)
        if text:
            if ui_message is None:
                ui_message = cl.Message(content=text)
                await ui_message.send()
            else:
                await ui_message.stream_token(token=text)
    if ui_message:
        await ui_message.update()


async def extract_text_from_output(output):
    """
    LangGraphのストリーミング出力からテキスト要素を抽出する関数

    Args:
        output (dict): LangGraphのストリーミング出力

    Returns:
        str: 抽出されたテキスト。テキストが見つからない場合は空文字列
    """
    try:
        if output["event"] == "on_chat_model_stream":
            chunk = output["data"]["chunk"]
            if hasattr(chunk, "content") and isinstance(chunk.content, list):
                # content が list 型の場合の処理
                for item in chunk.content:
                    if isinstance(item, dict) and "type" in item and item["type"] == "text":
                        return item["text"]
            elif hasattr(chunk, "content") and chunk.content:
                # content が直接文字列の場合の処理
                return chunk.content
    except (KeyError, AttributeError) as e:
        print(f"Error extracting text: {e}")
    return ""

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)
