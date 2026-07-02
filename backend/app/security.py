import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from passlib.context import CryptContext

from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

TOKEN_PREFIX = "eob_"


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes),
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
