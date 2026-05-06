import os
from dotenv import load_dotenv
from agent import app
from langchain_core.messages import HumanMessage

load_dotenv()

def test_agent():
    print("Testing agent with gemini-1.5-flash...")
    inputs = {"messages": [HumanMessage(content="Hello, what can you do?")]}
    config = {"recursion_limit": 10}
    
    try:
        for output in app.stream(inputs, config=config):
            for node_name, node_output in output.items():
                print(f"Node: {node_name}")
                if "messages" in node_output:
                    print(f"Response: {node_output['messages'][-1].content}")
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")

if __name__ == "__main__":
    test_agent()
