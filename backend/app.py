import os
import logging
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure Gemini AI Chatbot
class GeminiChatbot:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
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
            {"role": "model", "parts": ["Hello! I'm your AI Travel Assistant. I'm excited to help you plan your next adventure, answer travel questions, and provide recommendations. What destination or travel topic would you like to explore today?"]},
        ]

    def generate_response(self, user_message):
        try:
            self.conversation_history.append({"role": "user", "parts": [user_message]})
            response = self.model.generate_content(self.conversation_history)
            self.conversation_history.append({"role": "model", "parts": [response.text]})
            
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            return response.text
        except Exception as e:
            return f"An error occurred: {str(e)}"

# Initialize chatbot
try:
    gemini_api_key = "AIzaSyCCnAwdisdNl6H6fBNPGdTvtv6ohLcN7sg"
    if not gemini_api_key:
        raise ValueError("No API key found. Please set GEMINI_API_KEY in your environment variables.")
    chatbot = GeminiChatbot(gemini_api_key)
except Exception as e:
    logger.error(f"Chatbot Initialization Error: {e}")
    chatbot = None

# Travel Budget Calculator
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
            destination = destination.lower().strip()
            region = next((key for key in self.destination_costs if key in destination), None)
            costs = self.destination_costs.get(region, {'accommodation': 80, 'food': 40, 'transportation': 25, 'activities': 35})
            
            budget_variance = np.random.uniform(0.9, 1.1)
            estimated_budget = {category: round(value * duration * budget_variance, 2) for category, value in costs.items()}
            estimated_budget['total'] = round(sum(estimated_budget.values()), 2)
            estimated_budget['daily_average'] = round(estimated_budget['total'] / duration, 2)
            
            return estimated_budget
        except Exception as e:
            logger.error(f"Budget estimation error: {e}")
            raise

budget_calculator = TravelBudgetCalculator()

# Routes
@app.route('/')
def serve_chat():
    return send_from_directory('../frontend', 'chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    if not chatbot:
        return jsonify({"error": "Chatbot not initialized", "status": "error"}), 500
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({"error": "No message provided", "status": "error"}), 400
    try:
        response = chatbot.generate_response(user_message)
        return jsonify({"message": response, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route('/estimate_budget', methods=['POST'])
def estimate_trip_budget():
    try:
        data = request.json
        destination = data.get('destination', '')
        duration = data.get('duration', 7)
        if not destination:
            return jsonify({'error': 'Destination is required'}), 400
        try:
            duration = float(duration)
            if duration <= 0:
                return jsonify({'error': 'Duration must be greater than 0'}), 400
        except ValueError:
            return jsonify({'error': 'Duration must be a valid number'}), 400
        budget_estimate = budget_calculator.estimate_budget(destination, duration)
        return jsonify(budget_estimate), 200
    except Exception as e:
        logger.error(f"Error processing budget estimation: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Travel Chatbot Backend is running"})

@app.route('/generate', methods=['POST'])
def generate_response():
    try:
        data = request.get_json()
        prompt = data.get("prompt")

        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        # Ensure response has text
        generated_text = response.text if response else "I couldn't generate a response."

        return jsonify({"response": generated_text})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        logger.info('Starting Flask server...')
        app.run(debug=True, host='0.0.0.0', port=3000)
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
