import os
import shutil
import datetime
import uuid
import json
import asyncio
import sqlalchemy as sa
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, List
from dotenv import load_dotenv
from google import genai

from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agent import get_agent_app
from db.session import get_db, async_session_factory
from db.models import User, ChatHistory, ChatMessage
from auth.router import router as auth_router

from auth.utils import hash_password
from auth.dependencies import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

load_dotenv()

DB_URI = os.getenv("POSTGRES_DB_URL")

async def sync_existing_histories_to_messages(app: FastAPI):
    """Background task to sync existing history from LangGraph checkpointer to chat_messages table."""
    print("Starting historical chat messages indexing/synchronization task...")
    try:
        agent_app = app.state.agent_app
        
        async with async_session_factory() as db:
            result = await db.execute(select(ChatHistory))
            histories = result.scalars().all()
            
            for hist in histories:
                msg_check = await db.execute(
                    select(sa.func.count(ChatMessage.id)).where(ChatMessage.chat_history_id == hist.id)
                )
                count = msg_check.scalar() or 0
                if count > 0:
                    continue
                
                config = {"configurable": {"thread_id": hist.session_id}}
                try:
                    state = await agent_app.aget_state(config)
                    messages = state.values.get("messages", [])
                    if not messages:
                        continue
                    
                    print(f"Indexing {len(messages)} messages for historical chat session {hist.session_id}...")
                    
                    for msg in messages:
                        msg_type = getattr(msg, "type", None)
                        if msg_type == "human":
                            role = "user"
                        elif msg_type == "ai":
                            role = "assistant"
                        else:
                            continue
                            
                        content = msg.content
                        if isinstance(content, list):
                            text_parts = [part["text"] for part in content if isinstance(part, dict) and "text" in part]
                            content = " ".join(text_parts)
                        else:
                            content = str(content)
                            
                        chat_msg = ChatMessage(
                            chat_history_id=hist.id,
                            role=role,
                            content=content,
                            created_at=hist.created_at
                        )
                        db.add(chat_msg)
                        
                    await db.commit()
                except Exception as e:
                    print(f"Failed to sync session {hist.session_id}: {e}")
                    
        print("Historical chat messages synchronization task completed.")
    except Exception as e:
        print(f"Error in chat history sync task: {e}")

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
        
        # Run the historical messages sync task in the background
        asyncio.create_task(sync_existing_histories_to_messages(app))
        
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

# Mount uploads directory
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/health")
def health():
    return {"status": "ok"}

async def generate_title_from_llm(user_msg: str, assistant_msg: str) -> str:
    """Generate a short, concise, and clean title (3-5 words) using Gemini."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        
        client = genai.Client(api_key=api_key)
        prompt = (
            "You are a helpful assistant. Generate a short, concise, and engaging title (3 to 6 words) "
            "for a chat session based on the following conversation start. Do not use quotes, asterisks, "
            "or prefix it with 'Title:'. Keep it clean and direct.\n\n"
            f"User: {user_msg}\n"
            f"Assistant: {assistant_msg}"
        )
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=prompt
            )
        )
        if response and response.text:
            return response.text.strip().replace('"', '').replace("'", "")
    except Exception as e:
        print(f"Error generating title: {e}")
    return None

async def save_message_and_update_title(
    db: AsyncSession,
    chat_history: ChatHistory,
    role: str,
    content: str,
    user_message: str = None
):
    """Save message to database and update title if it's the first assistant message."""
    msg = ChatMessage(
        chat_history_id=chat_history.id,
        role=role,
        content=content
    )
    db.add(msg)
    await db.commit()

    if role == "assistant" and user_message:
        stmt = select(sa.func.count(ChatMessage.id)).where(ChatMessage.chat_history_id == chat_history.id)
        count_res = await db.execute(stmt)
        count = count_res.scalar() or 0
        
        if count <= 2:
            new_title = await generate_title_from_llm(user_message, content)
            if new_title:
                chat_history.title = new_title
                await db.commit()


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

    # Save user message to database
    await save_message_and_update_title(db, chat_history, "user", request.message)

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

        # Save assistant message to database and potentially update chat title
        await save_message_and_update_title(db, chat_history, "assistant", final_response, request.message)

        # Update the chat history timestamp
        chat_history.updated_at = datetime.datetime.now(datetime.timezone.utc)
        await db.commit()

        return ChatResponse(response=final_response, session_id=session_id)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream_endpoint(
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
        title = request.message[:100].strip()
        chat_history = ChatHistory(
            user_id=current_user.id,
            session_id=session_id,
            title=title,
        )
        db.add(chat_history)
        await db.commit()
        await db.refresh(chat_history)

    # Save user message to database
    await save_message_and_update_title(db, chat_history, "user", request.message)

    async def event_generator():
        config = {"recursion_limit": 50, "configurable": {"thread_id": session_id}}
        agent_app = app.state.agent_app
        accumulated_response = ""
        
        try:
            # Using astream_events to get real-time stream of LLM tokens and tool executions
            async for event in agent_app.astream_events({"messages": [("user", request.message)]}, config=config, version="v2"):
                kind = event.get("event")
                name = event.get("name")
                
                # Check for LLM token chunks
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        if isinstance(content, list):
                            text_parts = [part["text"] for part in content if isinstance(part, dict) and "text" in part]
                            content_str = " ".join(text_parts)
                        elif isinstance(content, dict) and "text" in content:
                            content_str = content["text"]
                        else:
                            content_str = str(content)
                        
                        if content_str:
                            accumulated_response += content_str
                            yield f"event: token\ndata: {json.dumps({'text': content_str})}\n\n"
                
                # Check for tool starts
                elif kind == "on_tool_start":
                    tool_input = event["data"].get("input")
                    yield f"event: tool_start\ndata: {json.dumps({'name': name, 'input': tool_input})}\n\n"
                    
                # Check for tool ends
                elif kind == "on_tool_end":
                    yield f"event: tool_end\ndata: {json.dumps({'name': name})}\n\n"
            
            # Streaming completed successfully!
            # Save assistant message to the database and potentially update chat title
            if accumulated_response:
                # We need a local DB session because generator runs outside the request dependency scope after the function returns
                async with async_session_factory() as local_db:
                    # Re-fetch chat history in local session
                    hist_res = await local_db.execute(select(ChatHistory).where(ChatHistory.id == chat_history.id))
                    local_chat_history = hist_res.scalars().first()
                    if local_chat_history:
                        await save_message_and_update_title(
                            local_db,
                            local_chat_history,
                            "assistant",
                            accumulated_response,
                            request.message
                        )
                        local_chat_history.updated_at = datetime.datetime.now(datetime.timezone.utc)
                        await local_db.commit()
                        title = local_chat_history.title
                    else:
                        title = chat_history.title
            else:
                title = chat_history.title
                
            yield f"event: done\ndata: {json.dumps({'session_id': session_id, 'title': title})}\n\n"
            
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"file_path": file_path, "message": "File uploaded successfully. You can now reference this path in your chat messages."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

class SearchResultItem(BaseModel):
    session_id: str
    title: str | None
    message_id: uuid.UUID
    role: str
    content: str
    created_at: datetime.datetime

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]

@app.get("/chat/search", response_model=SearchResponse)
async def search_chat(
    q: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if not q.strip():
        return SearchResponse(query=q, results=[])

    try:
        stmt = (
            select(
                ChatHistory.session_id,
                ChatHistory.title,
                ChatMessage.id.label("message_id"),
                ChatMessage.role,
                ChatMessage.content,
                ChatMessage.created_at
            )
            .join(ChatHistory, ChatMessage.chat_history_id == ChatHistory.id)
            .where(ChatHistory.user_id == current_user.id)
            .where(ChatMessage.content.ilike(f"%{q}%"))
            .order_by(ChatMessage.created_at.desc())
        )
        
        result = await db.execute(stmt)
        rows = result.all()
        
        results_list = []
        for row in rows:
            results_list.append(SearchResultItem(
                session_id=row.session_id,
                title=row.title,
                message_id=row.message_id,
                role=row.role,
                content=row.content,
                created_at=row.created_at
            ))
            
        return SearchResponse(query=q, results=results_list)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

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
