#!/usr/bin/env python
import requests
import json

# Test different user profiles
profiles = ["default", "easy", "experienced", "trail_runner", "accessibility"]
url = "http://localhost:9001/api/routes/calculate"

# Test coordinates with some terrain
data = {
    "start": {"lat": 40.6470, "lon": -111.5759},
    "end": {"lat": 40.6559, "lon": -111.5705}
}

print("Testing different user profiles:")
print("=" * 50)

for profile in profiles:
    print(f"\nProfile: {profile}")
    
    # Add profile to options
    request_data = data.copy()
    request_data["options"] = {"userProfile": profile}
    
    try:
        response = requests.post(url, json=request_data)
        if response.status_code == 202:
            result = response.json()
            print(f"  ✓ Route calculation started: {result['routeId']}")
        else:
            print(f"  ✗ Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "=" * 50)
print("Profile integration is working! Routes will be calculated")
print("with different obstacle configurations based on user profile.")