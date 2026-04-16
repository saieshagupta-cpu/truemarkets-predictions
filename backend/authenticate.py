"""
Interactive OTP auth for TrueMarkets REST API.
Run locally once; writes tokens to backend/tm_tokens.json.
The backend will pick those up on startup.

Usage:
    cd backend && venv/bin/python authenticate.py
"""

import json
import os
import sys
import time

from curl_cffi import requests

API_HOST = "https://api.truemarkets.co"
API_VERSION = "2026-01-26"
USER_AGENT = "tm/0.0.11"

TOKEN_FILE = os.path.join(os.path.dirname(__file__), "tm_tokens.json")


def main() -> int:
    session = requests.Session(impersonate="chrome")
    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    })

    email = input("Email: ").strip()

    r1 = session.post(f"{API_HOST}/v1/auth/email/otc", json={"email": email}, timeout=30)
    print(f"request OTC status: {r1.status_code}")
    if r1.status_code >= 400:
        print(r1.text[:500])
        return 1

    code = input("Enter verification code: ").strip()

    r2 = session.post(
        f"{API_HOST}/v1/auth/email/otc/verify",
        json={"email": email, "code": code},
        timeout=30,
    )
    print(f"verify status: {r2.status_code}")
    if r2.status_code >= 400:
        print(r2.text[:500])
        return 1

    tokens = r2.json()
    tokens["email"] = email
    tokens["saved_at"] = int(time.time())

    with open(TOKEN_FILE, "w") as f:
        json.dump(tokens, f, indent=2)

    print(f"\n✓ Saved tokens → {TOKEN_FILE}")
    print(f"  Keys: {list(tokens.keys())}")

    # Sanity check: fetch profile
    at = tokens.get("access_token")
    if at:
        r3 = session.get(
            f"{API_HOST}/v1/defi/core/profile/me",
            params={"version": API_VERSION},
            headers={"Authorization": f"Bearer {at}"},
            timeout=30,
        )
        print(f"\nprofile check: {r3.status_code}")
        if r3.status_code == 200:
            print(f"  {r3.text[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
