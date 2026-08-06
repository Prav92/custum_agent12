import os
import time
from typing import Annotated, TypedDict
import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from google import genai
import google.api_core.exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

MODEL_NAME = "gemini-flash-lite-latest"

# Define a common retry strategy for 429 errors
retry_strategy = retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(google.api_core.exceptions.ResourceExhausted),
    reraise=True
)

# --- STEP 1: Define the Tools ---

@tool
def web_search(query: str):
    """Finds up-to-date information on the web. Always include the current year/date in your query if looking for current events."""
    search = TavilySearch(max_results=3, topic="news")
    return search.invoke({"query": query})

@tool
def extract_from_document(file_path: str, prompt: str):
    """Uploads a local PDF/Doc and extracts specific information or answers questions about it.
    Use this tool for any questions related to a document's content.
    The file_path should be the path to a file in the 'uploads' directory or as provided by the user."""
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
        print(f"Extracting information from {file_path} using {MODEL_NAME}...")
        
        @retry_strategy
        def generate_with_retry():
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt, uploaded_file]
            )
            
        response = generate_with_retry()
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
    model=MODEL_NAME,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    streaming=True
).bind_tools(tools)

def get_system_prompt():
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    return SystemMessage(content=(
        f"You are a multimodal expert. The current date is {current_date}. "
        "You can search the web and extract data from documents. "
        "If you need to read a file, use the extract_from_document tool. "
        "If you need current info, use web_search. "
        "Always return your final answer as clear text, preferably formatted as JSON if requested. "
        "IMPORTANT: You MUST maintain context of previously mentioned files. If a user asks a follow-up question "
        "about a document you previously processed, reuse the file path from the conversation history."
    ))

async def call_model(state: AgentState, config: RunnableConfig):
    # Include system message in the call
    messages = [get_system_prompt()] + state["messages"]
    try:
        final_message = None
        async for chunk in llm.astream(messages, config=config):
            if final_message is None:
                final_message = chunk
            else:
                final_message += chunk
        return {"messages": [final_message]}
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        raise

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

def get_agent_app(checkpointer=None):
    return workflow.compile(checkpointer=checkpointer)

app = get_agent_app()