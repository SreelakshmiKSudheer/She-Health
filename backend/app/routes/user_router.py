from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from app.db.database import MongoDB
from app.schemas.user import UserCreate, UserResponse, UserUpdate
from app.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["Users"])

async def get_user_service():
    return UserService(MongoDB.db)

@router.post("/", response_model=UserResponse)
async def register_user(user: UserCreate, service: UserService = Depends(get_user_service)):
    # 1. Check if UUID already exists in MongoDB
    existing = await service.get_user_by_id(user.user_id)
    if existing:
        raise HTTPException(status_code=400, detail="User UUID already registered")
    
    # 2. Create profile with timestamps
    return await service.create_user_profile(user)

@router.get("/{user_id}", response_model=UserResponse)
async def fetch_user(user_id: str, service: UserService = Depends(get_user_service)):
    user = await service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.get("/", response_model=List[UserResponse])
async def list_users(service: UserService = Depends(get_user_service)):
    return await service.list_all_users()

@router.put("/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, update_data: UserUpdate, service: UserService = Depends(get_user_service)):
    # Convert Pydantic model to dict, removing unset values
    data = update_data.dict(exclude_unset=True)
    success = await service.update_user_profile(user_id, data)
    
    if not success:
        raise HTTPException(status_code=404, detail="User not found or no changes made")
    
    # Return the updated document
    return await service.get_user_by_id(user_id)

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str, service: UserService = Depends(get_user_service)):
    success = await service.delete_user_profile(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return None