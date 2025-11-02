from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import json
import os
import requests
import sys

app = Flask(__name__)
CORS(app)

# Configuration paths
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")
THRESHOLDS_PATH = os.path.join(MODEL_DIR, "best_thresholds.npy")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# GitHub release URL for large model file
MODEL_URL = "https://github.com/ChiNguyen-git/mindbridge-model-api/releases/download/v1.0/best_model.pt"

# Device configuration
device = torch.device('cpu')

# Ensure model folder exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Simplified 3-tier configuration
CONFIG = {
    "dropout": 0.3,
    "model_name": "distilbert-base-uncased",
    "thresholds": {
        "depression": 0.27,
        "ptsd": 0.29
    },
    "severity_levels": {
        "depression": {"low": [0, 0.27], "moderate": [0.27, 0.93], "severe": [0.93, 1.0]},
        "ptsd": {"low": [0, 0.29], "moderate": [0.29, 0.94], "severe": [0.94, 1.0]}
    }
}

# Load config.json if exists (only update dropout and model_name)
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, 'r') as f:
            loaded_config = json.load(f)
            CONFIG['dropout'] = loaded_config.get('dropout', 0.3)
            CONFIG['model_name'] = loaded_config.get('model_name', 'distilbert-base-uncased')
        print("✅ Config partially loaded from file")
    except Exception as e:
        print(f"⚠️ Using default config: {e}")

# Load thresholds from .npy file if exists
if os.path.exists(THRESHOLDS_PATH):
    try:
        thresholds = np.load(THRESHOLDS_PATH, allow_pickle=True)
        if isinstance(thresholds, np.ndarray) and len(thresholds) >= 2:
            CONFIG['thresholds']['depression'] = float(thresholds[0])
            CONFIG['thresholds']['ptsd'] = float(thresholds[1])
            print(f"✅ Loaded thresholds from file - Depression: {CONFIG['thresholds']['depression']:.3f}, PTSD: {CONFIG['thresholds']['ptsd']:.3f}")
    except Exception as e:
        print(f"⚠️ Using default thresholds: {e}")
else:
    print(f"ℹ️ Using hardcoded thresholds - Depression: {CONFIG['thresholds']['depression']}, PTSD: {CONFIG['thresholds']['ptsd']}")

# Download model if missing
if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading model file from GitHub Release...")
    print(f"   URL: {MODEL_URL}")
    try:
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(MODEL_PATH, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"   Progress: {percent:.1f}%", end='\r')
            print("\n✅ Model downloaded successfully!")
        else:
            print(f"❌ Failed to download model: HTTP {response.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        sys.exit(1)
else:
    print("✅ Model file found locally")

# Define Model Architecture
class DepressionPTSDModel(nn.Module):
    def __init__(self, dropout=0.3):
        super(DepressionPTSDModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 2)  # 2 outputs: depression, ptsd
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return self.sigmoid(logits)

# Load Model
print("📦 Loading model architecture...")
try:
    model = DepressionPTSDModel(dropout=CONFIG.get('dropout', 0.3))
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("✅ Model loaded and ready!")
except Exception as e:
    print(f"❌ CRITICAL: Failed to load model: {e}")
    sys.exit(1)

# Load tokenizer
print("📦 Loading tokenizer...")
try:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    print("✅ Tokenizer loaded!")
except Exception as e:
    print(f"❌ CRITICAL: Failed to load tokenizer: {e}")
    sys.exit(1)

# 3-Tier Level Functions
def get_depression_level(prob):
    if prob < 0.27:
        return 'low', 0
    elif prob < 0.93:
        return 'moderate', 1
    else:
        return 'severe', 2

def get_ptsd_level(prob):
    if prob < 0.29:
        return 'low', 0
    elif prob < 0.94:
        return 'moderate', 1
    else:
        return 'severe', 2

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'DistilBERT Depression/PTSD Detector',
        'thresholds': CONFIG['thresholds'],
        'severity_levels': CONFIG['severity_levels'],
        'model_loaded': True
    })

# Simplified analyze endpoint
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        text = data.get('transcript', '')
        if not text or not text.strip():
            return jsonify({'error': 'No text provided'}), 400

        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            probs = outputs.cpu().numpy()[0]

        depression_prob = float(probs[0])
        ptsd_prob = float(probs[1])
        depression_level, _ = get_depression_level(depression_prob)
        ptsd_level, _ = get_ptsd_level(ptsd_prob)

        response = {
            'depression_probability': depression_prob,
            'depression_level': depression_level,
            'ptsd_probability': ptsd_prob,
            'ptsd_level': ptsd_level
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error in analyze: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'type': 'analysis_error'}), 500

# Simplified test endpoint
@app.route('/test', methods=['GET'])
def test():
    test_cases = [
        "I feel great and happy today",
        "I'm a bit stressed about work", 
        "I feel very sad and hopeless, I can't sleep"
    ]
    results = []
    for text in test_cases:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            probs = outputs.cpu().numpy()[0]

        dep_prob = float(probs[0])
        ptsd_prob = float(probs[1])
        results.append({
            'text': text[:50],
            'depression_probability': dep_prob,
            'depression_level': get_depression_level(dep_prob)[0],
            'ptsd_probability': ptsd_prob,
            'ptsd_level': get_ptsd_level(ptsd_prob)[0]
        })
    return jsonify({'test_results': results})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("="*50)
    print(f"🚀 Starting MindBridge Model API (3-Tier System)")
    print(f"📍 Port: {port}")
    print(f"✅ Model: Loaded and ready")
    print("="*50)
    app.run(host='0.0.0.0', port=port, debug=False)
