import os
import logging
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Secure CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000", 
            "https://your-frontend-domain.com",
            "https://travelbudget.yoursite.com"
        ]
    }
})

# Add proxy support for secure deployment
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('travel_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Secure API key retrieval
def get_gemini_api_key():
    api_key ="AIzaSyCCnAwdisdNl6H6fBNPGdTvtv6ohLcN7sg"
    if not api_key:
        raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY environment variable.")
    return api_key

class GeminiChatbot:
    def __init__(self, api_key):
        try:
            genai.configure(api_key=api_key)
            
            self.model = genai.GenerativeModel(
                model_name="gemini-pro",
                generation_config={
                    "temperature": 0.7,
                    "top_p": 1,
                    "top_k": 1,
                    "max_output_tokens": 2048,
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
            )
            
            self.conversation_history = [
                {"role": "user", "parts": ["You are a helpful AI travel assistant. Provide friendly, informative, and concise responses about travel destinations, trip planning, and travel tips."]},
                {"role": "model", "parts": ["Hello! I'm your AI Travel Assistant. I'm excited to help you plan your next adventure, answer travel questions, and provide recommendations."]}
            ]
        except Exception as e:
            logger.error(f"Chatbot initialization failed: {e}")
            raise

    def generate_response(self, user_message):
        try:
            # Validate input
            if not user_message or len(user_message) > 1000:
                raise ValueError("Invalid message length")

            self.conversation_history.append({"role": "user", "parts": [user_message]})
            
            # Limit conversation history
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            response = self.model.generate_content(self.conversation_history)
            
            # Sanitize response
            sanitized_response = response.text[:2000] if response.text else "I couldn't generate a response."
            
            self.conversation_history.append({"role": "model", "parts": [sanitized_response]})
            
            return sanitized_response
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return "I'm sorry, but I'm unable to process your request right now."

class TravelBudgetCalculator:
    def __init__(self):
        self.destination_costs = {
            'europe': {'accommodation': 100, 'food': 50, 'transportation': 30, 'activities': 40},
            'asia': {'accommodation': 40, 'food': 20, 'transportation': 15, 'activities': 25},
            'north_america': {'accommodation': 150, 'food': 60, 'transportation': 40, 'activities': 50},
            'south_america': {'accommodation': 50, 'food': 25, 'transportation': 20, 'activities': 30},
        }

    def estimate_budget(self, destination, duration):
        try:
            # Input validation
            if not destination or not isinstance(duration, (int, float)) or duration <= 0:
                raise ValueError("Invalid destination or duration")

            destination = destination.lower().strip()
            region = next((key for key in self.destination_costs if key in destination), 'europe')
            costs = self.destination_costs.get(region)
            
            budget_variance = np.random.uniform(0.9, 1.1)
            estimated_budget = {
                category: round(value * duration * budget_variance, 2) 
                for category, value in costs.items()
            }
            estimated_budget['total'] = round(sum(estimated_budget.values()), 2)
            estimated_budget['daily_average'] = round(estimated_budget['total'] / duration, 2)
            
            return estimated_budget
        except Exception as e:
            logger.error(f"Budget estimation error for {destination}: {e}")
            raise ValueError(f"Could not estimate budget for {destination}")

# Application Setup
try:
    gemini_api_key = get_gemini_api_key()
    chatbot = GeminiChatbot(gemini_api_key)
    budget_calculator = TravelBudgetCalculator()
except Exception as e:
    logger.critical(f"Critical initialization error: {e}")
    chatbot = None
    budget_calculator = None

# Routes
@app.route('/api/chat', methods=['POST'])
def chat():
    if not chatbot:
        return jsonify({"error": "Chatbot service unavailable", "status": "error"}), 503
    
    data = request.json
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({"error": "No message provided", "status": "error"}), 400
    
    try:
        response = chatbot.generate_response(user_message)
        return jsonify({"message": response, "status": "success"})
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return jsonify({"error": "Internal server error", "status": "error"}), 500

@app.route('/api/estimate_budget', methods=['POST'])
def estimate_trip_budget():
    if not budget_calculator:
        return jsonify({"error": "Budget calculator service unavailable", "status": "error"}), 503
    
    try:
        data = request.json
        destination = data.get('destination', '').strip()
        duration = data.get('duration', 7)
        
        # Additional input validation
        if not destination:
            return jsonify({'error': 'Destination is required', 'status': 'error'}), 400
        
        try:
            duration = float(duration)
            if duration <= 0:
                return jsonify({'error': 'Duration must be greater than 0', 'status': 'error'}), 400
        except ValueError:
            return jsonify({'error': 'Duration must be a valid number', 'status': 'error'}), 400
        
        budget_estimate = budget_calculator.estimate_budget(destination, duration)
        return jsonify(budget_estimate), 200
    
    except ValueError as ve:
        logger.error(f"Budget estimation error: {ve}")
        return jsonify({'error': str(ve), 'status': 'error'}), 400
    except Exception as e:
        logger.error(f"Unexpected budget estimation error: {e}")
        return jsonify({'error': 'Internal server error', 'status': 'error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "services": {
            "chatbot": "running" if chatbot else "offline",
            "budget_calculator": "running" if budget_calculator else "offline"
        }
    })

if __name__ == '__main__':
    # Ensure environment variables are set
    required_env_vars = ['GEMINI_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.critical(f"Missing environment variables: {missing_vars}")
        raise SystemExit(1)
    
    try:
        logger.info('Starting Flask server...')
        app.run(host='0.0.0.0', port=int(os.getenv('PORT', 3000)), debug=False)
    except Exception as e:
        logger.critical(f"Server startup failed: {e}")
        raise SystemExit(1)