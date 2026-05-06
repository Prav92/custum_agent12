from agent import app

def run_agent():
    print("\n" + "="*30)
    print("   GEMINI MULTIMODAL AGENT   ")
    print("="*30)
    print("Type your request naturally.")
    print("Example: 'What is the latest news on AI?'")
    print("Example: 'Extract the total cost from 4223811716.pdf and format as JSON'")
    print("Type 'exit' or 'quit' to stop.\n")
    
    messages = []
    while True:
        user_input = input("User: ")
        if not user_input.strip(): continue
        if user_input.lower() in ['exit', 'quit']: break

        messages.append(("user", user_input))
        config = {"recursion_limit": 50, "configurable": {"thread_id": "cli_session"}}
        
        try:
            # We pass the full history to the agent
            for output in app.stream({"messages": messages}, config=config):
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
                        messages.append(final_msg) # Update history with agent response
                        print(f"\n[Final Response]:\n{final_msg.content}\n")
        except Exception as e:
            print(f"  [Error]: {str(e)}")

if __name__ == "__main__":
    run_agent()