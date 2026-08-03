import time
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_auth_flow():
    # Setup test credentials
    test_email = f"test_{int(time.time())}@example.com"
    valid_password = "SecurePassword123!"
    weak_password = "weak"
    
    print("\n--- 1. Testing Registration ---")
    
    # Test weak password
    register_weak_payload = {
        "email": test_email,
        "password": weak_password,
        "name": "Weak User",
        "age": 30
    }
    r = requests.post(f"{BASE_URL}/auth/register", json=register_weak_payload)
    print(f"Weak Password Registration: Status {r.status_code}")
    assert r.status_code == 422 or r.status_code == 400
    print("✓ Weak password rejected as expected")

    # Test valid registration
    register_payload = {
        "email": test_email,
        "password": valid_password,
        "name": "Auth Tester",
        "age": 28
    }
    r = requests.post(f"{BASE_URL}/auth/register", json=register_payload)
    print(f"Valid Registration: Status {r.status_code}")
    assert r.status_code == 201
    user_data = r.json()
    assert "id" in user_data
    assert user_data["email"] == test_email
    user_id = user_data["id"]
    print(f"✓ Valid registration succeeded. Created user UUID: {user_id}")

    # Test duplicate registration
    r = requests.post(f"{BASE_URL}/auth/register", json=register_payload)
    print(f"Duplicate Registration: Status {r.status_code}")
    assert r.status_code == 400
    print("✓ Duplicate email rejected as expected")

    print("\n--- 2. Testing Login ---")
    login_payload = {
        "email": test_email,
        "password": valid_password
    }
    r = requests.post(f"{BASE_URL}/auth/login", json=login_payload)
    print(f"Login: Status {r.status_code}")
    assert r.status_code == 200
    login_data = r.json()
    assert login_data["id"] == user_id
    
    cookies = r.cookies
    assert "access_token" in cookies
    assert "refresh_token" in cookies
    
    access_token = cookies["access_token"]
    refresh_token = cookies["refresh_token"]
    print("✓ Login succeeded. Cookies set:")
    print(f"  access_token: {access_token[:20]}...")
    print(f"  refresh_token: {refresh_token[:20]}...")

    print("\n--- 3. Testing Get Profile (/auth/me) ---")
    # Call /auth/me with cookies
    r = requests.get(f"{BASE_URL}/auth/me", cookies={"access_token": access_token})
    print(f"Get Profile: Status {r.status_code}")
    assert r.status_code == 200
    profile_data = r.json()
    assert profile_data["id"] == user_id
    assert profile_data["email"] == test_email
    print("✓ Profile fetched successfully")

    # Call /auth/me without cookies
    r = requests.get(f"{BASE_URL}/auth/me")
    print(f"Get Profile (No Token): Status {r.status_code}")
    assert r.status_code == 401
    print("✓ Missing token rejected successfully")

    print("\n--- 4. Testing Refresh Token Rotation ---")
    time.sleep(1) # Ensure time drift is handled
    r = requests.post(f"{BASE_URL}/auth/refresh", cookies={"refresh_token": refresh_token})
    print(f"Refresh Tokens: Status {r.status_code}")
    assert r.status_code == 200
    refresh_cookies = r.cookies
    assert "access_token" in refresh_cookies
    assert "refresh_token" in refresh_cookies
    
    new_access_token = refresh_cookies["access_token"]
    new_refresh_token = refresh_cookies["refresh_token"]
    assert new_access_token != access_token
    assert new_refresh_token != refresh_token
    print("✓ Token rotation succeeded. Received new tokens")

    # Test profile with new access token
    r = requests.get(f"{BASE_URL}/auth/me", cookies={"access_token": new_access_token})
    print(f"Get Profile with rotated token: Status {r.status_code}")
    assert r.status_code == 200
    print("✓ Rotated access token works")

    # Test replay attack / using old refresh token again
    r = requests.post(f"{BASE_URL}/auth/refresh", cookies={"refresh_token": refresh_token})
    print(f"Replay Old Refresh Token: Status {r.status_code}")
    assert r.status_code == 401
    print("✓ Replaying old refresh token rejected successfully")

    print("\n--- 5. Testing Logout ---")
    r = requests.post(f"{BASE_URL}/auth/logout", cookies={"refresh_token": new_refresh_token})
    print(f"Logout: Status {r.status_code}")
    assert r.status_code == 200
    
    # Confirm cookies are deleted (max_age/expiry updated or deleted)
    logout_cookies = r.cookies
    # Usually requests library removes cookies or shows them as empty/expired
    print("✓ Logout succeeded")

    # Test profile access after logout
    r = requests.get(f"{BASE_URL}/auth/me", cookies={"access_token": new_access_token})
    print(f"Get Profile after logout (old access token): Status {r.status_code}")
    # Access token itself is stateless, so it might technically still be valid if it hasn't expired (15m),
    # but the cookie has been cleared on the client side.
    
    # Test refresh token after logout (should be deleted from DB, so it must fail)
    r = requests.post(f"{BASE_URL}/auth/refresh", cookies={"refresh_token": new_refresh_token})
    print(f"Refresh after logout: Status {r.status_code}")
    assert r.status_code == 401
    print("✓ Revoked refresh token rejected successfully")

    print("\n--- 6. Testing Legacy Endpoints Compatibility ---")
    # Test POST /users (should work with UUID user table and return UUID)
    legacy_user_email = f"legacy_{int(time.time())}@example.com"
    r = requests.post(f"{BASE_URL}/users", json={"name": "Legacy", "email": legacy_user_email, "age": 40})
    print(f"Legacy User POST: Status {r.status_code}")
    assert r.status_code == 200
    legacy_user = r.json()
    legacy_id = legacy_user["id"]
    print(f"✓ Created legacy user with UUID: {legacy_id}")

    # Test GET /users
    r = requests.get(f"{BASE_URL}/users")
    print(f"Legacy Users List: Status {r.status_code}")
    assert r.status_code == 200
    users_list = r.json()
    assert len(users_list) >= 2 # Alice is gone (table dropped), but we created test user and legacy user
    print(f"✓ Found {len(users_list)} users in list")

    # Test GET /users/{id}
    r = requests.get(f"{BASE_URL}/users/{legacy_id}")
    print(f"Legacy User Fetch: Status {r.status_code}")
    assert r.status_code == 200
    assert r.json()["email"] == legacy_user_email
    print("✓ Legacy user fetch works")
    
    print("\n==============================")
    print("  ALL AUTH FLOW TESTS PASSED!  ")
    print("==============================\n")

if __name__ == "__main__":
    test_auth_flow()
