import os
import time
from typing import Annotated, TypedDict
import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from google import genai

load_dotenv()

# --- STEP 1: Define the Tools ---

@tool
def web_search(query: str):
    """Finds up-to-date information on the web. Always include the current year/date in your query if looking for current events."""
    search = TavilySearch(max_results=3, topic="news")
    return search.invoke({"query": query})

@tool
def extract_from_document(file_path: str, prompt: str):
    """Uploads a local PDF/Doc and extracts specific information from it."""
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Upload to Gemini's temporary file storage
    print(f"Uploading {file_path}...")
    uploaded_file = client.files.upload(file=file_path)
    
    # Wait for processing
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)
        
    try:
        # Use Gemini SDK directly for extraction to handle multimodal reliably
        print(f"Extracting information from {file_path}...")
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[prompt, uploaded_file]
        )
        result = response.text
    finally:
        # Cleanup: Delete the file from Gemini storage
        client.files.delete(name=uploaded_file.name)
        print(f"Deleted remote file {uploaded_file.name}")
        
    return result

tools = [web_search, extract_from_document]
tool_node = ToolNode(tools)

# --- STEP 2: Define the Agent State ---

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- STEP 3: Logic Nodes ---

# Set up the model with tools
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=os.getenv("GOOGLE_API_KEY")
).bind_tools(tools)

def get_system_prompt():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    return SystemMessage(content=(
        f"You are a multimodal expert. The current date is {current_date}. "
        "You can search the web and extract data from documents. "
        "If you need to read a file, use the extract_from_document tool. "
        "If you need current info, use web_search. "
        "Always return your final answer as clear text, preferably formatted as JSON if requested."
    ))

def call_model(state: AgentState):
    # Include system message in the call
    messages = [get_system_prompt()] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

def data_saver(state: AgentState):
    """Node that represents the final state before completion."""
    # This node is tracked by main.py to print the final result.
    return state

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "data_saver"

# --- STEP 4: Build the Graph ---

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("data_saver", data_saver)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
workflow.add_edge("data_saver", END)

app = workflow.compile()