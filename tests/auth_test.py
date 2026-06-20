import os
from dotenv import load_dotenv
load_dotenv()

# auth_test.py - Test authentication step by step
from huggingface_hub import login, HfApi
import requests

# Step 1: Test token validity
TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = "Maikobi/RARE_baseline_model"

print("ðŸ” Testing Hugging Face Authentication...")
print(f"Token: {TOKEN[:10]}...")
print(f"Model: {MODEL_ID}")
print("-" * 50)

# Test 1: Login
try:
    login(token=TOKEN)
    print("âœ… Login successful")
except Exception as e:
    print(f"âŒ Login failed: {e}")
    exit()

# Test 2: Check token permissions
try:
    api = HfApi(token=TOKEN)
    user_info = api.whoami()
    print(f"âœ… Token valid - User: {user_info['name']}")
except Exception as e:
    print(f"âŒ Token validation failed: {e}")

# Test 3: Check model existence and access
try:
    # Try to get model info
    model_info = api.model_info(MODEL_ID)
    print(f"âœ… Model exists: {model_info.modelId}")
    print(f"   Private: {model_info.private}")
    print(f"   Author: {model_info.author}")
except Exception as e:
    print(f"âŒ Model access failed: {e}")
    if "401" in str(e):
        print("ðŸ” This means the model is private and you don't have access")
        print("ðŸ”§ Solutions:")
        print("   1. Make the model public")
        print("   2. Check if you're the owner")
        print("   3. Regenerate your token with proper permissions")

# Test 4: Direct HTTP test
print("\nðŸŒ Testing direct HTTP access...")
headers = {"Authorization": f"Bearer {TOKEN}"}
url = f"https://huggingface.co/{MODEL_ID}/resolve/main/config.json"

response = requests.get(url, headers=headers)
print(f"HTTP Status: {response.status_code}")

if response.status_code == 200:
    print("âœ… Direct HTTP access works")
elif response.status_code == 401:
    print("âŒ Direct HTTP access also fails - Authentication issue")
    print("ðŸ”§ This confirms the token/permissions problem")
elif response.status_code == 404:
    print("âŒ Model not found")
else:
    print(f"âŒ Other error: {response.status_code}")

print("\n" + "="*50)
print("SUMMARY:")
print("If you see âŒ for model access, the issue is:")
print("1. Model is private and token doesn't have access")
print("2. Token doesn't have correct permissions")
print("3. You're not the model owner")
print("\nTo fix:")
print("1. Go to https://huggingface.co/Maikobi/RARE_baseline_model")
print("2. Check if you can see it (are you logged in as the right user?)")
print("3. If private, make it public OR regenerate token with full permissions")

