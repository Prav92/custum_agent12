from agent import app

def run_agent():
    print("\n" + "="*30)
    print("   GEMINI MULTIMODAL AGENT   ")
    print("="*30)
    print("Type your request naturally.")
    print("Example: 'What is the latest news on AI?'")
    print("Example: 'Extract the total cost from 4223811716.pdf and format as JSON'")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        user_input = input("User: ")
        if not user_input.strip(): continue
        if user_input.lower() in ['exit', 'quit']: break

        # Run through LangGraph
        # The agent now autonomously decides which tools to call.
        config = {"recursion_limit": 50}
        
        try:
            for output in app.stream({"messages": [("user", user_input)]}, config=config):
                print(f"Current output {output}")
                for node_name, node_output in output.items():
                    print(f"Current node {node_name}", node_output)
                    if node_name == "agent":
                        last_msg = node_output["messages"][-1]
                        print(f"\nLast message {last_msg}")
                        if last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                print(f"  [Thinking] Calling tool: {tc['name']}...")
                    elif node_name == "tools":
                        print("  [System] Tools executed.")
                    elif node_name == "data_saver":
                        final_content = node_output["messages"][-1].content
                        print(f"\n[Final Response]:\n{final_content}\n")
        except Exception as e:
            print(f"  [Error]: {str(e)}")

if __name__ == "__main__":
    run_agent()