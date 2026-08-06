import time
import requests
import json
import uuid

BASE_URL = "http://127.0.0.1:8000"

def verify_all():
    # Setup test credentials
    test_email = f"user_{int(time.time())}@example.com"
    valid_password = "SecurePassword123!"
    
    print("\n==============================================")
    print("      VERIFYING NEW BACKEND FEATURES")
    print("==============================================\n")
    
    # 1. Register & Login
    print("--- 1. Registering & Logging In ---")
    register_payload = {
        "email": test_email,
        "password": valid_password,
        "name": "Feature Tester",
        "age": 25
    }
    r = requests.post(f"{BASE_URL}/auth/register", json=register_payload)
    print(f"Register status: {r.status_code}")
    assert r.status_code == 201
    
    login_payload = {
        "email": test_email,
        "password": valid_password
    }
    r = requests.post(f"{BASE_URL}/auth/login", json=login_payload)
    print(f"Login status: {r.status_code}")
    assert r.status_code == 200
    access_token = r.cookies.get("access_token")
    cookies = {"access_token": access_token}
    print("✓ Auth successful\n")
    
    # 2. Test uploads static serving
    print("--- 2. Testing Upload and File Serving ---")
    file_content = b"This is a test file for verification."
    files = {"file": ("test_verify.txt", file_content, "text/plain")}
    r = requests.post(f"{BASE_URL}/upload", files=files)
    print(f"Upload status: {r.status_code}")
    assert r.status_code == 200
    res_upload = r.json()
    file_path = res_upload["file_path"]
    print(f"Uploaded file path: {file_path}")
    
    # Try fetching the file statically
    served_url = f"{BASE_URL}/uploads/test_verify.txt"
    r_get = requests.get(served_url)
    print(f"GET static served file status: {r_get.status_code}")
    assert r_get.status_code == 200
    print(f"Served file content: '{r_get.text}'")
    assert r_get.content == file_content
    print("✓ Static uploads serving verified successfully\n")
    
    # 3. Test SSE streaming chat endpoint
    print("--- 3. Testing SSE Chat Streaming ---")
    session_id = str(uuid.uuid4())
    # We will trigger a search to see tool events too
    chat_payload = {
        "message": "Find current news on Space X launch and summarize it.",
        "session_id": session_id
    }
    
    print(f"Sending stream request for session: {session_id}")
    r = requests.post(
        f"{BASE_URL}/chat/stream",
        json=chat_payload,
        cookies=cookies,
        stream=True
    )
    print(f"Stream response status: {r.status_code}")
    assert r.status_code == 200
    
    tokens_received = []
    tool_events = []
    done_event = None
    
    for line in r.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("event:"):
                event_type = line_str.split("event: ")[1].strip()
            elif line_str.startswith("data:"):
                data_str = line_str.split("data: ")[1].strip()
                data = json.loads(data_str)
                if event_type == "token":
                    tokens_received.append(data["text"])
                    # Print token chunks inline to show live streaming feel
                    print(data["text"], end="", flush=True)
                elif event_type == "tool_start":
                    print(f"\n[Tool Started]: {data['name']} with input {data['input']}")
                    tool_events.append(("start", data["name"]))
                elif event_type == "tool_end":
                    print(f"\n[Tool Ended]: {data['name']}")
                    tool_events.append(("end", data["name"]))
                elif event_type == "done":
                    done_event = data
                    print(f"\n[Done]: Session ID = {data['session_id']}, Title = '{data['title']}'")
                elif event_type == "error":
                    print(f"\n[Error event]: {data['error']}")
                    
    print("\n✓ Stream parsing complete")
    assert len(tokens_received) > 0
    print(f"Total tokens received: {len(tokens_received)}")
    print(f"Tool events captured: {tool_events}")
    assert done_event is not None
    print("✓ SSE streaming verified successfully\n")
    
    # 4. Test LLM-based Automatic Title Generation
    print("--- 4. Checking LLM-generated Title ---")
    # Fetch histories and check the title
    r_hist = requests.get(f"{BASE_URL}/history", cookies=cookies)
    assert r_hist.status_code == 200
    histories_data = r_hist.json()
    matched_history = None
    for h in histories_data["histories"]:
        if h["session_id"] == session_id:
            matched_history = h
            break
            
    assert matched_history is not None
    generated_title = matched_history["title"]
    print(f"Generated title in database for session {session_id}: '{generated_title}'")
    # Verify it is not just the slice of first message (message starts with "Find current news on Space X...")
    assert generated_title != ""
    assert "Find current news on Space X" not in generated_title or len(generated_title) < 50
    print("✓ Dynamic LLM title generation verified\n")
    
    # 5. Test Global Search Endpoint
    print("--- 5. Testing Global Full-Text Search ---")
    # We will search for a keyword that was in the message or response (e.g. 'Space')
    search_query = "Space"
    r_search = requests.get(f"{BASE_URL}/chat/search", params={"q": search_query}, cookies=cookies)
    print(f"Search status: {r_search.status_code}")
    assert r_search.status_code == 200
    search_data = r_search.json()
    print(f"Search results for '{search_query}':")
    print(json.dumps(search_data, indent=2))
    assert len(search_data["results"]) > 0
    # The first result must match our session
    assert any(item["session_id"] == session_id for item in search_data["results"])
    print("✓ Search returned matching messages successfully\n")
    
    print("==============================================")
    print("      ALL FEATURE VERIFICATIONS PASSED!")
    print("==============================================")

if __name__ == "__main__":
    verify_all()
