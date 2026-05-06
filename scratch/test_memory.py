import requests
import os
import time

BASE_URL = "http://localhost:8000"

def test_session():
    session_id = f"test_session_{int(time.time())}"
    file_path = "uploads/Praveen Kumar - Full Stack Engineer.pdf"
    
    print(f"--- Starting test session: {session_id} ---")
    
    # 1. Ask a question about the document
    print("\nQuestion 1: What is the main skill listed in this document?")
    payload = {
        "message": f"Please look at {file_path} and tell me the main skills listed.",
        "session_id": session_id
    }
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    print(f"Response 1: {response.json().get('response')}")
    
    # 2. Ask a follow-up question without mentioning the file path
    print("\nQuestion 2: Does the document mention experience with Python?")
    payload = {
        "message": "Does the document mention experience with Python? Answer based on the same file.",
        "session_id": session_id
    }
    response = requests.post(f"{BASE_URL}/chat", json=payload)
    print(f"Response 2: {response.json().get('response')}")

if __name__ == "__main__":
    # Ensure api.py is running in another terminal or start it here if possible
    # For now, I'll just check if I can run it.
    try:
        test_session()
    except Exception as e:
        print(f"Error: {e}. Make sure the API server is running at {BASE_URL}")
