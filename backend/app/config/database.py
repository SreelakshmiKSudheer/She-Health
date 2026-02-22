from pymongo import AsyncMongoClient
import pymongo
import os

# MongoDB connection
conn: AsyncMongoClient = None
db = None

async def connect_to_mongo():
    global conn, db
    client = AsyncMongoClient(os.environ["MONGODB_URL"],server_api=pymongo.server_api.ServerApi(version="1", strict=True,deprecation_errors=True))
    db = client.get_database(os.environ["DB_NAME"])
