# utils/db.py
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from datetime import datetime
from bson import ObjectId

load_dotenv()

# Same MongoDB URI jo Node.js me hai
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/medical_auth")
client = MongoClient(MONGO_URI)
db = client.get_default_database()

# Collections
chat_collection = db["chatmessages"]  # Same collection name as Node.js

# Create indexes
chat_collection.create_index([("userId", 1), ("createdAt", -1)])


class ChatMessageDB:
    """Helper class for chat operations"""
    
    @staticmethod
    def save_message(user_id, sender, text, audio_data=None, session_id=None):
        """Save message to MongoDB"""
        try:
            message = {
                "userId": ObjectId(user_id),  # Convert string to ObjectId
                "sender": sender,
                "text": text,
                "audioData": audio_data,
                "sessionId": session_id or str(datetime.now().timestamp()),
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow()
            }
            result = chat_collection.insert_one(message)
            return str(result.inserted_id)
        except Exception as e:
            print(f"Error saving message: {e}")
            return None