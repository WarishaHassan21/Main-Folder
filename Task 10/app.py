from flask import Flask, render_template, request, jsonify
import json
import random

app = Flask(__name__)

# Load intents
with open("intents.json") as file:
    intents = json.load(file)

# Simple keyword-based matcher
def get_response(user_input):
    user_input = user_input.lower()
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if pattern in user_input:
                return random.choice(intent['responses'])
    return "I'm sorry, I didn't understand that. Can you please rephrase?"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def chat():
    user_msg = request.args.get('msg')
    response = get_response(user_msg)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
