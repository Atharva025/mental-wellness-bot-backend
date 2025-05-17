import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import logging
from flask import Flask, request, jsonify
from datetime import datetime
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
# Configure CORS to allow requests from your frontend explicitly
CORS(app, resources={r"/api/*": {"origins": ["https://mental-wellness-ten.vercel.app", "*"]}})

# Model paths - adjust based on your project structure
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model')

# Global variables for model and tokenizer
model = None
tokenizer = None

# Replace the load_mental_health_model function with this:
def load_mental_health_model():
    """Load the fine-tuned model and tokenizer from Hugging Face Hub"""
    global model, tokenizer
    
    try:
        # Your Hugging Face model ID 
        model_id = "Atharva025/mental-wellness-chatbot"  # Replace with your username
        
        logger.info(f"Loading model and tokenizer from Hugging Face Hub: {model_id}")
        
        # Set your token if your model is private
        hf_token = os.environ.get('HF_TOKEN')
        
        # Load tokenizer with special tokens
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        
        # Load the model with memory optimization
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, 
            token=hf_token,
            low_cpu_mem_usage=True  # Good for deployment environments with limited RAM
        )
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        logger.info(f"Model loaded successfully from Hugging Face. Using device: {device}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model from Hugging Face: {str(e)}")
        return None, None

def detect_mood(text):
    """Detect mood from text to provide context"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['anxious', 'anxiety', 'worried', 'panic', 'nervous']):
        return "anxious"
    elif any(word in text_lower for word in ['sad', 'depress', 'hopeless', 'unmotivated']):
        return "depressed"
    elif any(word in text_lower for word in ['stress', 'overwhelm', 'pressure', 'burnt out']):
        return "stressed"
    
    return None

def detect_crisis(text):
    """Detect if the message contains crisis indicators"""
    CRISIS_KEYWORDS = [
        "suicide", "kill myself", "end my life", "take my own life",
        "don't want to live", "want to die", "harming myself", "self harm",
        "hurting myself", "end it all"
    ]
    
    text_lower = text.lower()
    for keyword in CRISIS_KEYWORDS:
        if keyword in text_lower:
            return True
    return False

def prepare_input(message, history="", mood=None, is_crisis=False):
    """Prepare the input text for the model with appropriate special tokens"""
    
    # Add mood context if provided
    mood_context = f"Current mood: {mood}. " if mood else ""
    
    # Add appropriate special token based on content
    if is_crisis:
        special_token = "[CRISIS] "
        if "suicid" in message.lower() or "kill myself" in message.lower():
            special_token = "[SUICIDAL] "
    elif "anxious" in message.lower() or "anxiety" in message.lower() or "worried" in message.lower():
        special_token = "[ANXIETY] "
    elif "depress" in message.lower() or "sad" in message.lower() or "hopeless" in message.lower():
        special_token = "[DEPRESSION] "
    elif "stress" in message.lower() or "overwhelm" in message.lower():
        special_token = "[STRESS] "
    else:
        special_token = "[SUPPORT] "
    
    # Combine context, history, and message
    input_text = f"{special_token}{mood_context}{history}\nUser: {message}"
    
    return input_text

def get_model_response(message, history="", max_length=150):
    """Get a response from the mental health model"""
    global model, tokenizer
    
    # If model not loaded, try to load it
    if model is None or tokenizer is None:
        model, tokenizer = load_mental_health_model()
    
    if not model or not tokenizer:
        return "I'm sorry, I couldn't load my knowledge base. Could you try again later?"
    
    try:
        # Detect mood and crisis indicators
        mood = detect_mood(message)
        is_crisis = detect_crisis(message)
        
        # Prepare input with special tokens and context
        input_text = prepare_input(message, history, mood, is_crisis)
        
        # Tokenize the input
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate a response
        generation_config = {
            "max_length": max_length,
            "num_beams": 4,
            "no_repeat_ngram_size": 3,
            "early_stopping": True
        }
        
        with torch.no_grad():
            output = model.generate(**inputs, **generation_config)
        
        # Decode the response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Clean up the response
        response = response.strip()
        
        # Add resource prefix for crisis responses if not already present
        if is_crisis and not response.startswith("[RESOURCE]"):
            response = f"[RESOURCE] {response}"
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I'm having trouble processing your message right now. Could you try rephrasing it?"

# Add manual CORS headers to all responses as a backup
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Add health check endpoint
@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Health check endpoint"""
    # Handle preflight request
    if request.method == 'OPTIONS':
        return '', 204
    
    # Check if model is loaded
    global model, tokenizer
    if model is None or tokenizer is None:
        # Try to load it
        model, tokenizer = load_mental_health_model()
    
    model_loaded = model is not None and tokenizer is not None
    
    return jsonify({
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    })

# Chat API endpoint with OPTIONS method support
@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat_endpoint():
    """Chat API endpoint"""
    # Handle preflight request
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.json
    if not data or 'message' not in data:
        return jsonify({
            "error": "No message provided",
            "response": "Please provide a message."
        }), 400
    
    message = data.get('message', '')
    history = data.get('history', '')
    
    # Get response from model
    response = get_model_response(message, history)
    
    return jsonify({
        "response": response,
        "timestamp": datetime.now().isoformat()
    })

# At the end of the file, modify to this:
if __name__ == "__main__":
    # Preload the model before starting the server
    print("Pre-loading the model...")
    model, tokenizer = load_mental_health_model()
    print(f"Model loaded: {model is not None}, Tokenizer loaded: {tokenizer is not None}")
    
    # Get port from environment variable or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    