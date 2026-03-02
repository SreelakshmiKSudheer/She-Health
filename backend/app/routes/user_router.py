from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from app.db.database import MongoDB
from app.schemas.user import UserCreate, UserResponse, UserUpdate
from app.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["Users"])

# This ensures the service always has the latest MongoDB connection
async def get_user_service():
    if MongoDB.db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return UserService(MongoDB.db)

@router.post("/", response_model=UserResponse)
async def register_user(user: UserCreate, service: UserService = Depends(get_user_service)):
    existing = await service.get_user_by_id(user.user_id)
    if existing:
        raise HTTPException(status_code=400, detail="User UUID already registered")
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
    data = update_data.model_dump(exclude_unset=True) # model_dump is the Pydantic v2 version of .dict()
    success = await service.update_user_profile(user_id, data)
    
    if not success:
        raise HTTPException(status_code=404, detail="User not found or no changes made")
    
    return await service.get_user_by_id(user_id)

@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: str, service: UserService = Depends(get_user_service)):
    success = await service.delete_user_profile(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return None