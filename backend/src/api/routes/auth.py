import os
import uuid
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import bcrypt
import jwt
from cryptography.fernet import Fernet
from src.db.user_queries import (
    create_user, get_user_by_email, 
    get_user_by_id, update_gemini_key
)
from src.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()

JWT_SECRET = os.getenv("JWT_SECRET", "change-this-secret")
JWT_EXPIRY_DAYS = 30
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")  

def get_cipher():
    if not ENCRYPTION_KEY:
        raise HTTPException(status_code=500, detail="Encryption key not configured")
    return Fernet(ENCRYPTION_KEY.encode())

def create_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.utcnow() + timedelta(days=JWT_EXPIRY_DAYS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    try:
        payload = jwt.decode(
            credentials.credentials, 
            JWT_SECRET, 
            algorithms=["HS256"]
        )
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
class RegisterRequest(BaseModel):
    email: str
    password: str
    invite_code: str

class LoginRequest(BaseModel):
    email: str
    password: str

class ApiKeyRequest(BaseModel):
    gemini_api_key: str

INVITE_CODE = os.getenv("DAX_INVITE_CODE")

@router.post("/register")
def register(request:RegisterRequest):
    if request.invite_code != INVITE_CODE:
        raise HTTPException(status_code=403, detail="Invalid invite code")
    
    existing = get_user_by_email(request.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    password_hash = bcrypt.hashpw(
        request.password.encode(), 
        bcrypt.gensalt()
    ).decode()
    user_id = str(uuid.uuid4())
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    create_user(user_id, request.email, password_hash, created_at)
    token = create_token(user_id)
    logger.info(f"User registered | email={request.email}")
    return {"token": token, "user_id": user_id, "email": request.email}


@router.post('/login')
def login(request:LoginRequest):
    user = get_user_by_email(request.email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    if not user[4]:  
        raise HTTPException(status_code=403, detail="Account disabled")
    
    if not bcrypt.checkpw(request.password.encode(), user[2].encode()):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    token = create_token(user[0])
    logger.info(f"User logged in | email={request.email}")
    return {"token": token, "user_id": user[0], "email": user[1]}


@router.get("/me")
def get_me(user_id: str = Depends(verify_token)):
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    has_api_key = bool(user[3])  
    return {
        "user_id": user[0],
        "email": user[1],
        "has_gemini_key": has_api_key,
        "created_at": user[5]
    }


@router.post("/api-key")
def save_api_key(
    request: ApiKeyRequest,
    user_id: str = Depends(verify_token)
):
    cipher = get_cipher()
    encrypted = cipher.encrypt(request.gemini_api_key.encode()).decode()
    update_gemini_key(user_id, encrypted)
    logger.info(f"Gemini API key saved | user_id={user_id}")
    return {"message": "API key saved"}

@router.get("/api-key/status")
def get_api_key_status(user_id: str = Depends(verify_token)):
    user = get_user_by_id(user_id)
    return {"has_gemini_key": bool(user[3])}