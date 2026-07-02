from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app import models
from app.database import get_db
from app.security import TOKEN_PREFIX, decode_access_token, hash_api_token

bearer = HTTPBearer(auto_error=False)


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> models.User:
    """Accepts either a JWT session token or a long-lived API token (eob_...)."""
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    token = creds.credentials

    if token.startswith(TOKEN_PREFIX):
        row = (
            db.query(models.ApiToken)
            .filter(models.ApiToken.token_hash == hash_api_token(token))
            .first()
        )
        if row is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API token")
        row.last_used_at = datetime.now(timezone.utc)
        db.commit()
        return row.user

    user_id = decode_access_token(token)
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired session")
    user = db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    return user


def require_admin(user: models.User = Depends(get_current_user)) -> models.User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
    return user
