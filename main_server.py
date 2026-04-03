# ==============================================================================
# Module: Main API Gateway & Web Server (Web Speech API Upgraded)
# Description: Handles JSON Text payloads. Audio transcription is now 
# handled natively by the client browser for zero-latency processing.
# ==============================================================================

from flask import Flask, render_template, request, jsonify
from core_ai.sarcasm_engine import SarcasmIntelligence

app = Flask(__name__)

print("\n[BOOT] Initializing Sarcasm Engine...")
ai_brain = SarcasmIntelligence()
print("[BOOT] Engine Ready.\n")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles ALL input (typed text or browser-transcribed voice)."""
    data = request.json
    user_text = data.get('text', '')
    
    if not user_text:
        return jsonify({"error": "No text provided"}), 400
        
    print(f"[DATA RECEIVED] Analyzing: '{user_text}'")
    
    # Pass text to the AI Brain
    result = ai_brain.analyze_text(user_text)
    return jsonify(result)

if __name__ == '__main__':
    print("[SYSTEM] Multimodal Sarcasm Recognizer live on http://127.0.0.1:8000")
    app.run(debug=True, port=8000)