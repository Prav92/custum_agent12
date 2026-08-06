import os
import asyncio
from dotenv import load_dotenv
from agent import get_agent_app
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

load_dotenv()
DB_URI = os.getenv("POSTGRES_DB_URL")

async def test_stream():
    async with AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=5,
        kwargs={"autocommit": True, "prepare_threshold": 0}
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        agent_app = get_agent_app(checkpointer=checkpointer)
        
        config = {"recursion_limit": 50, "configurable": {"thread_id": "test_streaming_session_new7"}}
        
        print("Starting stream_events...")
        # We will query Tavily to trigger a tool call to see if we capture it
        inputs = {"messages": [HumanMessage(content="What is the current weather in Paris?")]}
        
        async for event in agent_app.astream_events(inputs, config=config, version="v2"):
            kind = event.get("event")
            name = event.get("name")
            print(f"Event: {kind}, Name: {name}")
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    print(f"Token: {repr(content)}")
            elif kind == "on_tool_start":
                print(f"Tool Start: {name} with inputs {event['data'].get('input')}")
            elif kind == "on_tool_end":
                print(f"Tool End: {name}")

if __name__ == "__main__":
    asyncio.run(test_stream())
