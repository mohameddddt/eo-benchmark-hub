import hashlib
import secrets
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from app.config import settings

TOKEN_PREFIX = "eob_"


def hash_password(password: str) -> str:
    # bcrypt only looks at the first 72 bytes; pre-hash so long passphrases count fully
    digest = hashlib.sha256(password.encode()).hexdigest().encode()
    return bcrypt.hashpw(digest, bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    digest = hashlib.sha256(plain.encode()).hexdigest().encode()
    try:
        return bcrypt.checkpw(digest, hashed.encode())
    except ValueError:
        return False


def create_access_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> str | None:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])["sub"]
    except jwt.PyJWTError:
        return None


def generate_api_token() -> tuple[str, str, str]:
    """Returns (plaintext_token, display_prefix, sha256_hash). Plaintext is shown once."""
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    return token, token[:12], hashlib.sha256(token.encode()).hexdigest()


def hash_api_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
