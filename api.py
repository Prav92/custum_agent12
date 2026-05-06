import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the compiled LangGraph app from agent.py
from agent import app as agent_app

app = FastAPI(title="Gemini Multimodal Agent API")

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

# In-memory session store
sessions = {}

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
    if session_id not in sessions:
        sessions[session_id] = []
        
    # Add user message to history
    sessions[session_id].append(("user", request.message))
    
    config = {"recursion_limit": 50, "configurable": {"thread_id": session_id}}
    final_response = ""
    
    try:
        # Run the full history through the LangGraph agent
        for output in agent_app.stream({"messages": sessions[session_id]}, config=config):
            for node_name, node_output in output.items():
                if node_name == "data_saver":
                    final_msg = node_output["messages"][-1]
                    sessions[session_id].append(final_msg) # Save agent response to history
                    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
