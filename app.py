from flask import Flask, render_template, jsonify, request, session
from flask_cors import CORS
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.voice_handler import voice_handler
from dotenv import load_dotenv
from src.prompt import *
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import os
import requests
import secrets
import jwt
import base64

app = Flask(__name__)
CORS(app, supports_credentials=True, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

load_dotenv()

# Configuration
NODE_API_URL = os.environ.get('NODE_API_URL', 'http://localhost:8080/api')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
ACCESS_TOKEN_SECRET = os.environ.get('ACCESS_TOKEN_SECRET')

# Twilio WhatsApp Configuration
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_WHATSAPP_NUMBER = os.environ.get('TWILIO_WHATSAPP_NUMBER')

# Initialize Twilio Client
twilio_client = None
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Pinecone and OpenAI setup
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "medicalchatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
chatModel = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


def extract_user_id_from_token():
    """Extract user ID from Authorization header"""
    auth_header = request.headers.get('Authorization')
    
    if not auth_header:
        return "anonymous"
    
    try:
        token = auth_header.replace('Bearer ', '').strip()
        
        if not token or not isinstance(token, str) or not ACCESS_TOKEN_SECRET:
            return "anonymous"
        
        decoded = jwt.decode(
            token, 
            ACCESS_TOKEN_SECRET, 
            algorithms=["HS256"],
            options={"verify_signature": True}
        )
        
        user_id = decoded.get('id')
        return str(user_id) if user_id else "anonymous"
        
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError, Exception):
        return "anonymous"


def save_message_to_node(sender, text, audio_data=None, session_id=None, user_id="anonymous", token=None):
    """Save message via Node.js API"""
    try:
        url = f"{NODE_API_URL}/chat/save-message-public"
        
        payload = {
            "sender": sender,
            "text": text,
            "audioData": audio_data,
            "sessionId": session_id or "default-session",
        }
        
        headers = {'Content-Type': 'application/json'}
        
        if token:
            token_str = str(token) if not isinstance(token, str) else token
            headers['Authorization'] = f'Bearer {token_str}'
        
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        
        if response.status_code in [200, 201]:
            return response.json()
        return None
            
    except Exception:
        return None


def get_chat_history_from_node(session_id=None, limit=100, token=None):
    """Get chat history from Node.js API"""
    try:
        url = f"{NODE_API_URL}/chat/chat-history-public"
        
        params = {
            "limit": limit,
            "sessionId": session_id or "default-session"
        }
        
        headers = {}
        if token:
            token_str = str(token) if not isinstance(token, str) else token
            headers['Authorization'] = f'Bearer {token_str}'
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return response.json().get('messages', [])
        return []
            
    except Exception:
        return []


def get_ai_response(user_message):
    """Get AI response from RAG chain"""
    try:
        response = rag_chain.invoke({"input": user_message})
        return str(response["answer"])
    except Exception as e:
        print(f"Error in AI response: {str(e)}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again."


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    """WhatsApp webhook - Handles both text and voice messages"""
    try:
        print("\n" + "="*60)
        print("📱 WhatsApp Webhook Triggered!")
        print("="*60)
        
        # Get incoming message details
        incoming_msg = request.values.get('Body', '').strip()
        from_number = request.values.get('From', '')
        media_url = request.values.get('MediaUrl0', '')  # Voice/audio URL
        media_type = request.values.get('MediaContentType0', '')
        num_media = request.values.get('NumMedia', '0')
        
        print(f"📞 From Number: {from_number}")
        print(f"💬 Text Message: {incoming_msg}")
        print(f"🎵 Media URL: {media_url}")
        print(f"📎 Media Type: {media_type}")
        print(f"📊 Num Media: {num_media}")
        
        # Extract user phone number for session
        user_phone = from_number.replace('whatsapp:', '')
        session_id = f"whatsapp_{user_phone}"
        print(f"🎫 Session ID: {session_id}")
        
        # Handle Voice Message
        if media_url and media_type and 'audio' in media_type.lower():
            print("🎤 Voice message detected!")
            print(f"💾 Downloading audio from: {media_url}")
            
            try:
                # Download audio file from Twilio with authentication
                auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
                audio_response = requests.get(media_url, auth=auth, timeout=30)
                
                if audio_response.status_code != 200:
                    raise Exception(f"Failed to download audio: {audio_response.status_code}")
                
                # Convert to base64
                audio_base64 = base64.b64encode(audio_response.content).decode('utf-8')
                print(f"✅ Audio downloaded ({len(audio_response.content)} bytes) and converted to base64")
                
                # Transcribe audio to text
                print("🎯 Transcribing audio...")
                user_result = voice_handler.speech_to_text(audio_base64)
                
                if not user_result or not isinstance(user_result, dict) or not user_result.get('text'):
                    raise Exception("Failed to transcribe audio")
                
                user_text = user_result.get('text')
                print(f"📝 Transcribed Text: {user_text}")
                
                # Save user voice message
                print(f"💾 Saving user voice message to database...")
                save_message_to_node(
                    sender="user",
                    text=user_text,
                    audio_data=audio_base64,
                    session_id=session_id,
                    user_id=user_phone
                )
                
                # Get AI response
                print(f"🤖 Generating AI response...")
                answer = get_ai_response(user_text)
                print(f"✅ AI Response: {answer[:100]}...")
                
                # Save bot response
                print(f"💾 Saving bot response to database...")
                save_message_to_node(
                    sender="bot",
                    text=answer,
                    session_id=session_id,
                    user_id=user_phone
                )
                
                # Send text response back to WhatsApp
                resp = MessagingResponse()
                msg = resp.message()
                msg.body(f"🎤 You said: *{user_text}*\n\n{answer}")
                
                print(f"📤 Sending response back to WhatsApp")
                print("="*60 + "\n")
                
                return str(resp), 200, {'Content-Type': 'text/xml'}
                
            except Exception as voice_error:
                print(f"❌ Voice processing error: {str(voice_error)}")
                import traceback
                traceback.print_exc()
                
                resp = MessagingResponse()
                resp.message("Sorry, I couldn't process your voice message. Please try sending a text message instead or try recording again.")
                return str(resp), 200, {'Content-Type': 'text/xml'}
        
        # Handle Text Message
        elif incoming_msg:
            print("📝 Text message detected!")
            print(f"💾 Saving user message to database...")
            
            # Save user message
            save_message_to_node(
                sender="user",
                text=incoming_msg,
                session_id=session_id,
                user_id=user_phone
            )
            
            # Get AI response
            print(f"🤖 Generating AI response...")
            answer = get_ai_response(incoming_msg)
            print(f"✅ AI Response: {answer[:100]}...")
            
            # Save bot response
            print(f"💾 Saving bot response to database...")
            save_message_to_node(
                sender="bot",
                text=answer,
                session_id=session_id,
                user_id=user_phone
            )
            
            # Send response back to WhatsApp
            resp = MessagingResponse()
            msg = resp.message()
            msg.body(answer)
            
            print(f"📤 Sending response back to WhatsApp")
            print("="*60 + "\n")
            
            return str(resp), 200, {'Content-Type': 'text/xml'}
        
        # Empty message (ignore)
        else:
            print("⚠️ Empty message - ignoring")
            return str(MessagingResponse()), 200, {'Content-Type': 'text/xml'}
        
    except Exception as e:
        print(f"\n❌ ERROR in WhatsApp webhook:")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n")
        
        resp = MessagingResponse()
        resp.message("Sorry, something went wrong. Please try again later.")
        return str(resp), 200, {'Content-Type': 'text/xml'}


@app.route("/whatsapp/send", methods=["POST"])
def send_whatsapp_message():
    """Send proactive WhatsApp message"""
    if not twilio_client:
        return jsonify({"error": "Twilio not configured"}), 500
    
    try:
        data = request.get_json()
        to_number = data.get('to')
        message = data.get('message')
        
        if not to_number or not message:
            return jsonify({"error": "Missing 'to' or 'message'"}), 400
        
        # Ensure proper WhatsApp format
        if not to_number.startswith('whatsapp:'):
            to_number = f'whatsapp:{to_number}'
        
        # Send message via Twilio
        twilio_message = twilio_client.messages.create(
            body=message,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=to_number
        )
        
        return jsonify({
            "success": True,
            "message_sid": twilio_message.sid,
            "status": twilio_message.status
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/whatsapp/status", methods=["GET"])
def whatsapp_status():
    """Check WhatsApp integration status"""
    return jsonify({
        "configured": twilio_client is not None,
        "account_sid": TWILIO_ACCOUNT_SID[:10] + "..." if TWILIO_ACCOUNT_SID else None,
        "whatsapp_number": TWILIO_WHATSAPP_NUMBER
    })


@app.route("/get", methods=["GET", "POST"])
def chat():
    """Handle text chat"""
    user_id = extract_user_id_from_token()
    auth_header = request.headers.get('Authorization', '')
    token = auth_header.replace('Bearer ', '').strip() if auth_header else None
    
    if request.method == "POST":
        if request.is_json:
            data = request.get_json()
            msg = data.get("msg")
            session_id = data.get("session_id")
        else:
            msg = request.form.get("msg")
            session_id = request.form.get("session_id")
    else:
        msg = request.args.get("msg")
        session_id = request.args.get("session_id")
    
    if not msg:
        return jsonify({"error": "No message provided"}), 400
    
    if not session.get('chat_session_id'):
        session['chat_session_id'] = session_id or str(secrets.token_hex(8))
    
    save_message_to_node(
        sender="user",
        text=msg,
        session_id=session.get('chat_session_id'),
        user_id=user_id,
        token=token
    )
    
    answer = get_ai_response(msg)
    
    save_message_to_node(
        sender="bot",
        text=answer,
        session_id=session.get('chat_session_id'),
        user_id=user_id,
        token=token
    )
    
    return jsonify({
        "answer": answer,
        "session_id": session.get('chat_session_id')
    })


@app.route("/voice-chat", methods=["POST"])
def voice_chat():
    """Handle voice chat"""
    user_id = extract_user_id_from_token()
    auth_header = request.headers.get('Authorization', '')
    token = auth_header.replace('Bearer ', '').strip() if auth_header else None
    
    try:
        data = request.get_json()
        audio_base64 = data.get('audio')
        session_id = data.get('session_id')
        
        if not audio_base64:
            return jsonify({"error": "No audio data provided"}), 400
        
        if not session.get('chat_session_id'):
            session['chat_session_id'] = session_id or str(secrets.token_hex(8))
        
        user_result = voice_handler.speech_to_text(audio_base64)
        
        if not user_result or not isinstance(user_result, dict) or not user_result.get('text'):
            return jsonify({"error": "Failed to transcribe audio"}), 400

        user_text = user_result.get('text')
        
        save_message_to_node(
            sender="user",
            text=user_text,
            audio_data=audio_base64,
            session_id=session.get('chat_session_id'),
            user_id=user_id,
            token=token
        )
        
        answer_text = get_ai_response(user_text)
        answer_audio = voice_handler.text_to_speech(answer_text)
        
        save_message_to_node(
            sender="bot",
            text=answer_text,
            audio_data=answer_audio,
            session_id=session.get('chat_session_id'),
            user_id=user_id,
            token=token
        )
        
        return jsonify({
            "text": answer_text,
            "audio": answer_audio,
            "user_text": user_text,
            "session_id": session.get('chat_session_id')
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/history", methods=["GET"])
def get_chat_history():
    """Get chat history from Node.js backend"""
    auth_header = request.headers.get('Authorization', '')
    token = auth_header.replace('Bearer ', '').strip() if auth_header else None
    session_id = request.args.get('session_id') or session.get('chat_session_id', 'default-session')
    limit = request.args.get('limit', 100)
    
    messages = get_chat_history_from_node(session_id=session_id, limit=limit, token=token)
    
    return jsonify({
        "success": True,
        "messages": messages,
        "session_id": session_id
    })


@app.route("/speech-to-text", methods=["POST"])
def speech_to_text_endpoint():
    """Transcribe audio to text only"""
    try:
        data = request.get_json()
        audio_base64 = data.get('audio')
        
        if not audio_base64:
            return jsonify({"error": "No audio data provided"}), 400

        user_result = voice_handler.speech_to_text(audio_base64)
        
        if not user_result or not isinstance(user_result, dict) or not user_result.get('text'):
            return jsonify({"error": "Failed to transcribe audio"}), 500

        return jsonify({
            "text": user_result.get('text'),
            "confidence": user_result.get('confidence')
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/text-to-speech", methods=["POST"])
def text_to_speech_endpoint():
    """Convert text response to speech"""
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        audio_base64 = voice_handler.text_to_speech(text)
        
        if not audio_base64:
            return jsonify({"error": "Failed to convert text to speech"}), 500
        
        return jsonify({"audio": audio_base64})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "success": False,
        "error": "Internal Server Error",
        "message": "An unexpected error occurred"
    }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Medical Chatbot Server Starting...")
    print("="*60)
    print(f"✅ Flask Server: Running on http://0.0.0.0:5000")
    print(f"✅ Web Chat: /get endpoint")
    print(f"✅ Voice Chat: /voice-chat endpoint")
    print(f"✅ WhatsApp Bot: /whatsapp endpoint (Text + Voice)")
    
    if twilio_client:
        print(f"✅ Twilio: Configured")
        print(f"   📱 WhatsApp Number: {TWILIO_WHATSAPP_NUMBER}")
        print(f"   🎤 Voice Message Support: Enabled")
    else:
        print(f"⚠️  Twilio: Not configured (set env variables)")
    
    print("="*60 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True)