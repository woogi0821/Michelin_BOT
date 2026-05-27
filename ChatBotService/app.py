from flask import Flask, request, jsonify
from chatbot_engine import GeminiChatbot

app = Flask(__name__)
chatbot = GeminiChatbot()

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json()

    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = data.get('message')
    history = data.get('history', [])

    bot_reply = chatbot.get_response(message, history)

    return jsonify({"answer": bot_reply})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)