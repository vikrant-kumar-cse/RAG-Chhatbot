# middleware/auth.py
from functools import wraps
from flask import request, jsonify
import jwt
import os
from dotenv import load_dotenv

load_dotenv()

# Same secret key jo Node.js me use kar rahe ho
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-must-match-nodejs")

def authenticate_user(f):
    """Middleware to authenticate JWT token from Node.js"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        # Get token from header
        if "Authorization" in request.headers:
            auth_header = request.headers["Authorization"]
            try:
                token = auth_header.split(" ")[1]  # Bearer <token>
            except IndexError:
                return jsonify({"success": False, "error": "Invalid token format"}), 401
        
        if not token:
            return jsonify({"success": False, "error": "Token is missing"}), 401
        
        try:
            # Decode JWT token (same secret as Node.js)
            decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            
            # Extract user info from token
            request.user_id = decoded.get("userId") or decoded.get("_id")
            
            if not request.user_id:
                return jsonify({"success": False, "error": "Invalid token"}), 401
                
        except jwt.ExpiredSignatureError:
            return jsonify({"success": False, "error": "Token has expired"}), 401
        except jwt.InvalidTokenError as e:
            return jsonify({"success": False, "error": f"Invalid token: {str(e)}"}), 401
        
        return f(*args, **kwargs)
    
    return decorated_function