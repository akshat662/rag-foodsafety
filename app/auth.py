"""API key authentication via the X-API-Key header.

The key is read from the APP_API_KEY environment variable at call time
(never hardcoded), matching src/config.py's convention for every other
provider key. Comparison uses secrets.compare_digest to avoid a timing
side-channel on key comparison.
"""

import os
import secrets

from fastapi import Header, HTTPException, status


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("APP_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="APP_API_KEY is not configured on the server",
        )
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key")
