import os
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from agent import get_agent_app

load_dotenv()
DB_URI = os.getenv("POSTGRES_DB_URL")

def run_agent():
    print("\n" + "="*30)
    print("   GEMINI MULTIMODAL AGENT   ")
    print("="*30)
    print("Type your request naturally.")
    print("Example: 'What is the latest news on AI?'")
    print("Example: 'Extract the total cost from 4223811716.pdf and format as JSON'")
    print("Type 'exit' or 'quit' to stop.\n")
    
    with ConnectionPool(conninfo=DB_URI, max_size=20, kwargs={"autocommit": True, "prepare_threshold": 0}) as pool:
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        app = get_agent_app(checkpointer=checkpointer)

        while True:
            user_input = input("User: ")
            if not user_input.strip(): continue
            if user_input.lower() in ['exit', 'quit']: break

            config = {"recursion_limit": 50, "configurable": {"thread_id": "cli_session"}}
            
            try:
                for output in app.stream({"messages": [("user", user_input)]}, config=config):
                    for node_name, node_output in output.items():
                        if node_name == "agent":
                            last_msg = node_output["messages"][-1]
                            if last_msg.tool_calls:
                                for tc in last_msg.tool_calls:
                                    print(f"  [Thinking] Calling tool: {tc['name']}...")
                        elif node_name == "tools":
                            print("  [System] Tools executed.")
                        elif node_name == "data_saver":
                            final_msg = node_output["messages"][-1]
                            print(f"\n[Final Response]:\n{final_msg.content}\n")
            except Exception as e:
                print(f"  [Error]: {str(e)}")

if __name__ == "__main__":
    run_agent()