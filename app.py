import os
import logging
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

class GeminiChatbot:
    def __init__(self, api_key):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-pro",
                generation_config={
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "top_k": 40,
                    "max_output_tokens": 1024,
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
            )
            # Enhanced system prompt for structured responses
            self.conversation_history = [
                {"role": "user", "parts": ["""
                You are a travel expert assistant. Always provide responses in this exact format:
                
                [Main Answer - 1 sentence summary]
                
                • [Key point 1]
                • [Key point 2] 
                • [Key point 3]
                
                *Important Note*: [Any critical information]
                
                Rules:
                - Use ONLY bullet points (•) after the first sentence
                - Keep each bullet point to 1 line
                - Never write paragraphs
                - Highlight important info with *
                - Use simple language
                """]
                },
                {"role": "model", "parts": ["I'm your travel assistant. I'll provide clear, bullet-point answers to all your travel questions. Where would you like to go?"]},
            ]
            logger.info("Gemini chatbot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            raise

    def generate_response(self, user_message):
        try:
            if not user_message or not isinstance(user_message, str):
                raise ValueError("Invalid message format")
                
            self.conversation_history.append({"role": "user", "parts": [user_message]})
            response = self.model.generate_content(self.conversation_history)
            
            # Strictly format the response
            formatted_response = self._format_response(response.text)
            
            self.conversation_history.append({"role": "model", "parts": [formatted_response]})
            
            # Maintain conversation history
            if len(self.conversation_history) > 8:
                self.conversation_history = [self.conversation_history[0], self.conversation_history[1]] + self.conversation_history[-6:]
                
            return formatted_response
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return "• Sorry, I couldn't process that\n• Please ask another travel question"

    def _format_response(self, text):
        """Force response into strict bullet point format"""
        # First clean the text
        text = text.strip()
        
        # Replace any existing bullets with standard format
        text = text.replace('- ', '• ').replace('* ', '• ')
        
        # Split into lines and process
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        formatted_lines = []
        
        # Process first line as summary
        if lines:
            formatted_lines.append(lines[0])
        
        # Process remaining lines as bullet points
        for line in lines[1:]:
            if not line.startswith('•'):
                line = '• ' + line
            formatted_lines.append(line)
        
        # Ensure no paragraph breaks
        formatted_text = '\n'.join(formatted_lines)
        formatted_text = formatted_text.replace('\n\n', '\n')
        
        return formatted_text

# Initialize chatbot
try:
    gemini_api_key = "AIzaSyCCnAwdisdNl6H6fBNPGdTvtv6ohLcN7sg"
    if not gemini_api_key:
        raise ValueError("Missing Gemini API key")
    chatbot = GeminiChatbot(gemini_api_key)
    logger.info("Chatbot initialized successfully")
except Exception as e:
    logger.critical(f"Chatbot initialization failed: {str(e)}")
    chatbot = None

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        response = jsonify({"status": "preflight"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response

    if not chatbot:
        return jsonify({"error": "Chat service unavailable"}), 503

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' field"}), 400

        user_message = data['message']
        if not isinstance(user_message, str) or not user_message.strip():
            return jsonify({"error": "Invalid message format"}), 400

        bot_response = chatbot.generate_response(user_message)
        return jsonify({
            "message": bot_response,
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=True)