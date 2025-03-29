from flask import Flask, render_template, jsonify
import requests

app = Flask(__name__)

GNEWS_API_KEY = "afe15d33d1eb41c537edd12ff5baf1a1"
GNEWS_API_URL = "https://gnews.io/api/v4/top-headlines"
COUNTRY = "us"  # Change this to fetch news from a specific country

@app.route("/", methods=["GET"])
def index():
    try:
        params = {"token": GNEWS_API_KEY, "country": COUNTRY, "lang": "en", "max": 10}
        response = requests.get(GNEWS_API_URL, params=params)
        
        if response.status_code == 200:
            news_data = response.json()
            return render_template("index.html", news=news_data.get("articles", []))
        else:
            return jsonify({"error": "Failed to fetch data from GNews API", "status_code": response.status_code}), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)