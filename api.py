import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List
from dotenv import load_dotenv

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agent import get_agent_app

load_dotenv()

DB_URI = os.getenv("POSTGRES_DB_URL", "postgresql://postgres:postgres@localhost:5432/novaagentdb")

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0}
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        app.state.agent_app = get_agent_app(checkpointer=checkpointer)
        yield

app = FastAPI(title="Gemini Multimodal Agent API", lifespan=lifespan)
# Setup CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
        
    session_id = request.session_id
    config = {"recursion_limit": 50, "configurable": {"thread_id": session_id}}
    final_response = ""
    agent_app = app.state.agent_app
    
    try:
        # Run the full history through the LangGraph agent
        async for output in agent_app.astream({"messages": [("user", request.message)]}, config=config):
            for node_name, node_output in output.items():
                if node_name == "data_saver":
                    final_msg = node_output["messages"][-1]
                    
                    content = final_msg.content
                    if isinstance(content, list):
                        # Extract text from list of dicts
                        text_parts = [part["text"] for part in content if isinstance(part, dict) and "text" in part]
                        final_response = " ".join(text_parts)
                    else:
                        final_response = str(content)
                    
        if not final_response:
            raise HTTPException(status_code=500, detail="Agent did not produce a final response.")
            
        return ChatResponse(response=final_response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"file_path": file_path, "message": "File uploaded successfully. You can now reference this path in your chat messages."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

class Message(BaseModel):
    role: str
    content: Any

class HistoryResponse(BaseModel):
    session_id: str
    messages: List[Message]

@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    agent_app = app.state.agent_app
    
    try:
        state = await agent_app.aget_state(config)
        messages = state.values.get("messages", [])
        
        formatted_messages = []
        for msg in messages:
            msg_type = getattr(msg, "type", None)
            if msg_type == "human":
                role = "user"
            elif msg_type == "ai":
                role = "assistant"
            elif msg_type == "system":
                role = "system"
            elif msg_type == "tool":
                role = "tool"
            else:
                role = "unknown"
                
            formatted_messages.append(Message(
                role=role,
                content=msg.content
            ))
            
        return HistoryResponse(session_id=session_id, messages=formatted_messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
