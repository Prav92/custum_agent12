import os
import shutil
import datetime
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, List
from dotenv import load_dotenv

from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agent import get_agent_app
from db.session import get_db
from db.models import User, ChatHistory
from auth.router import router as auth_router
from auth.utils import hash_password
from auth.dependencies import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

load_dotenv()

DB_URI = os.getenv("POSTGRES_DB_URL")

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
        app.state.db_pool = pool
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

# Include Authentication Router
app.include_router(auth_router)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/health")

def health():
    return {"status": "ok"}

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # Optional: auto-generated if not provided

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Use provided session_id or generate a new one
    session_id = request.session_id or str(uuid.uuid4())

    # Check if this session already exists and belongs to this user
    result = await db.execute(
        select(ChatHistory).where(ChatHistory.session_id == session_id)
    )
    chat_history = result.scalars().first()

    if chat_history:
        # Session exists — verify ownership
        if chat_history.user_id != current_user.id:
            raise HTTPException(status_code=403, detail="This session does not belong to you.")
    else:
        # Create a new ChatHistory record for this user
        title = request.message[:100].strip()  # Use first 100 chars of message as title
        chat_history = ChatHistory(
            user_id=current_user.id,
            session_id=session_id,
            title=title,
        )
        db.add(chat_history)
        await db.commit()
        await db.refresh(chat_history)

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

        # Update the chat history timestamp
        chat_history.updated_at = datetime.datetime.now(datetime.timezone.utc)
        await db.commit()

        return ChatResponse(response=final_response, session_id=session_id)
        
    except HTTPException:
        raise
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

class ChatHistoryItem(BaseModel):
    id: uuid.UUID
    session_id: str
    title: str | None = None
    created_at: datetime.datetime
    updated_at: datetime.datetime

    class Config:
        from_attributes = True

class ChatHistoryListResponse(BaseModel):
    user_id: uuid.UUID
    histories: List[ChatHistoryItem]

class HistoryResponse(BaseModel):
    session_id: str
    messages: List[Message]

@app.get("/history", response_model=ChatHistoryListResponse)
async def list_history(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all chat sessions for the authenticated user."""
    try:
        result = await db.execute(
            select(ChatHistory)
            .where(ChatHistory.user_id == current_user.id)
            .order_by(ChatHistory.updated_at.desc())
        )
        histories = result.scalars().all()

        # Resolve any placeholder titles ("Session <uuid>" or "Untitled") dynamically
        agent_app = app.state.agent_app
        updated_any = False
        for chat_history in histories:
            if not chat_history.title or chat_history.title.startswith("Session ") or chat_history.title == "Untitled":
                config = {"configurable": {"thread_id": chat_history.session_id}}
                try:
                    state = await agent_app.aget_state(config)
                    messages = state.values.get("messages", [])
                    for msg in messages:
                        msg_type = getattr(msg, "type", None)
                        if msg_type == "human" and msg.content:
                            content = msg.content
                            if isinstance(content, list):
                                text_parts = [part["text"] for part in content if isinstance(part, dict) and "text" in part]
                                content = " ".join(text_parts)
                            title = str(content)[:100].strip()
                            if title:
                                chat_history.title = title
                                updated_any = True
                                break
                except Exception:
                    pass

        if updated_any:
            await db.commit()
            # Re-fetch to return the updated titles
            result = await db.execute(
                select(ChatHistory)
                .where(ChatHistory.user_id == current_user.id)
                .order_by(ChatHistory.updated_at.desc())
            )
            histories = result.scalars().all()

        return ChatHistoryListResponse(
            user_id=current_user.id,
            histories=histories
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch histories: {str(e)}")

@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get chat messages for a specific session. Verifies ownership."""
    # Verify that this session belongs to the current user
    result = await db.execute(
        select(ChatHistory).where(
            ChatHistory.session_id == session_id,
            ChatHistory.user_id == current_user.id
        )
    )
    chat_history = result.scalars().first()
    if not chat_history:
        raise HTTPException(status_code=404, detail="Chat session not found or does not belong to you.")

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

class ChatHistoryUpdateRequest(BaseModel):
    title: str

@app.patch("/history/{session_id}", response_model=ChatHistoryItem)
async def update_history(
    session_id: str,
    request: ChatHistoryUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Rename a chat session title."""
    result = await db.execute(
        select(ChatHistory).where(
            ChatHistory.session_id == session_id,
            ChatHistory.user_id == current_user.id
        )
    )
    chat_history = result.scalars().first()
    if not chat_history:
        raise HTTPException(status_code=404, detail="Chat session not found or does not belong to you.")

    chat_history.title = request.title
    chat_history.updated_at = datetime.datetime.now(datetime.timezone.utc)
    await db.commit()
    await db.refresh(chat_history)
    return chat_history

@app.delete("/history/{session_id}")
async def delete_history(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a chat session."""
    result = await db.execute(
        select(ChatHistory).where(
            ChatHistory.session_id == session_id,
            ChatHistory.user_id == current_user.id
        )
    )
    chat_history = result.scalars().first()
    if not chat_history:
        raise HTTPException(status_code=404, detail="Chat session not found or does not belong to you.")

    await db.delete(chat_history)
    await db.commit()
    return {"status": "success", "message": f"Session '{session_id}' deleted."}

class UserCreate(BaseModel):
    name: str | None = None
    email: str
    age: int | None = None

class UserResponse(BaseModel):
    id: uuid.UUID
    name: str | None = None
    email: str
    age: int | None = None
    created_at: datetime.datetime

    class Config:
        from_attributes = True

@app.post("/users", response_model=UserResponse)
async def create_user(user: UserCreate, db: AsyncSession = Depends(get_db)):
    try:
        # Check if email is already registered
        result = await db.execute(select(User).where(User.email == user.email))
        existing_user = result.scalars().first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create user with a dummy/default password hash for legacy compatibility
        dummy_pwd_hash = hash_password("legacy_user_no_password_12345")
        db_user = User(
            name=user.name,
            email=user.email,
            age=user.age,
            password_hash=dummy_pwd_hash
        )
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        return db_user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/users", response_model=List[UserResponse])
async def list_users(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(select(User).order_by(User.created_at.asc()))
        return result.scalars().all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str, db: AsyncSession = Depends(get_db)):
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format (UUID expected)")
    
    try:
        result = await db.execute(select(User).where(User.id == user_uuid))
        user = result.scalars().first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
