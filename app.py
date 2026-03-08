from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import math
import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
try:
    model = joblib.load('random_forest_model.joblib')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")

def entropy(s):
    if not s:
        return 0
    prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
    return - sum([p * math.log(p) / math.log(2.0) for p in prob])

def extract_features(url):
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            url_to_parse = 'http://' + url
            parsed_url = urlparse(url_to_parse)
        domain = parsed_url.netloc
    except:
        domain = ""

    features = {
        'url_length': len(url),
        'domain_length': len(domain),
        'num_dots': url.count('.'),
        'num_hyphens': url.count('-'),
        'num_underscores': url.count('_'),
        'num_slashes': url.count('/'),
        'num_questionmarks': url.count('?'),
        'num_equals': url.count('='),
        'num_ampersands': url.count('&'),
        'num_at_symbols': url.count('@'),
        'num_digits': sum(c.isdigit() for c in url),
        'entropy_domain': entropy(domain),
        'has_http': 1 if 'http://' in url.lower() else 0,
        'has_https': 1 if 'https://' in url.lower() else 0,
        'has_www': 1 if 'www.' in url.lower() else 0
    }
    return pd.DataFrame([features])

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    url = data.get('url')
    
    if not url:
        logger.warning("Prediction requested without URL")
        return jsonify({'error': 'No URL provided'}), 400

    logger.info(f"Processing prediction for URL: {url}")
    
    try:
        # Feature extraction
        features_df = extract_features(url)
        
        # Model prediction
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        
        # 0: benign, 1: malicious. Risk score is the probability of class 1
        risk_score = float(probabilities[1])
        result = 'malicious' if prediction == 1 else 'benign'
        
        response = {
            'url': url,
            'prediction': result,
            'risk_score': round(risk_score, 4)
        }
        
        logger.info(f"Result for {url}: {result} (Score: {risk_score})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
