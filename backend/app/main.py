from fastapi import FastAPI
from app.config.database import connect_to_mongo, close_mongo_connection
# from app.routers import test_router

app = FastAPI()

@app.on_event("startup")
async def startup():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown():
    await close_mongo_connection()

# app.include_router(test_router.router)

@app.get("/")
async def root():
    return "Hello world"