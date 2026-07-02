from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import models, schemas
from app.database import get_db
from app.deps import get_current_user
from app.security import create_access_token, generate_api_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=schemas.UserOut, status_code=201)
def register(body: schemas.RegisterIn, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.email == body.email.lower()).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")
    user = models.User(
        email=body.email.lower(),
        password_hash=hash_password(body.password),
        name=body.name,
        affiliation=body.affiliation,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=schemas.TokenOut)
def login(body: schemas.LoginIn, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == body.email.lower()).first()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Email or password is incorrect")
    return schemas.TokenOut(access_token=create_access_token(user.id))


@router.get("/me", response_model=schemas.UserOut)
def me(user: models.User = Depends(get_current_user)):
    return user


@router.post("/tokens", response_model=schemas.ApiTokenCreatedOut, status_code=201)
def create_token(
    body: schemas.ApiTokenCreateIn,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    plaintext, prefix, token_hash = generate_api_token()
    row = models.ApiToken(user_id=user.id, name=body.name, prefix=prefix, token_hash=token_hash)
    db.add(row)
    db.commit()
    db.refresh(row)
    return schemas.ApiTokenCreatedOut(id=row.id, name=row.name, token=plaintext, prefix=prefix)


@router.get("/tokens", response_model=list[schemas.ApiTokenOut])
def list_tokens(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(models.ApiToken).filter(models.ApiToken.user_id == user.id).all()


@router.delete("/tokens/{token_id}", status_code=204)
def revoke_token(
    token_id: str,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    row = db.get(models.ApiToken, token_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Token not found")
    db.delete(row)
    db.commit()
