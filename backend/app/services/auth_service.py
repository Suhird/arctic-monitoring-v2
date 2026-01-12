"""
Authentication service for user management
"""
from sqlalchemy.orm import Session
from ..models.user import User, UserRole
from ..schemas.user import UserCreate
from ..utils.auth import hash_password, verify_password, create_access_token
from datetime import timedelta
from uuid import uuid4
from ..config import settings


def create_user(db: Session, user_data: UserCreate) -> User:
    """Create a new user"""
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise ValueError("User with this email already exists")

    # Create user
    hashed_pw = hash_password(user_data.password)
    user = User(
        email=user_data.email,
        hashed_password=hashed_pw,
        full_name=user_data.full_name,
        role=UserRole.FREE,
        api_key=str(uuid4())
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    """Authenticate a user"""
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None

    if not verify_password(password, user.hashed_password):
        return None

    return user


def login_user(db: Session, email: str, password: str) -> dict:
    """Login user and return access token"""
    user = authenticate_user(db, email, password)
    if not user:
        raise ValueError("Invalid credentials")

    # Generate token
    access_token = create_access_token(
        user.id,
        timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": user
    }
