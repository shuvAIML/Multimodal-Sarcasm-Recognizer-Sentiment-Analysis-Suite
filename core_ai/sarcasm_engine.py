# ==============================================================================
# MULTIMODAL SARCASM RECOGNIZER & SENTIMENT ANALYSIS SUITE
# Module: Core AI Brain (Multilingual, ABSA, Maximum-Coverage)
# ==============================================================================

import torch
import warnings
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
print("[SYSTEM] Booting Maximum-Coverage Sarcasm Engine...")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Load SpaCy for Aspect-Based Sentiment Analysis (ABSA)
try:
    nlp = spacy.load("en_core_web_sm")
    print("[SYSTEM] SpaCy NLP pipeline loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load SpaCy model. Did you run 'python -m spacy download en_core_web_sm'? Error: {e}")

class SarcasmIntelligence:
    def __init__(self):
        # 1. MULTILINGUAL POSITIVE DICTIONARY (English, Hindi, Bengali, Spanish)
        self.pos_words = [
            # English
            "great", "brilliant", "genius", "love", "amazing", "perfect", "good", 
            "fantastic", "wow", "favorite", "adore", "best", "happy", "awesome",
            # Hindi
            "badiya", "accha", "mast", "zabardast", "pyar", "shandar",
            # Bengali
            "bhalo", "darun", "khub bhalo", "bhalobashi", "chomotkar",
            # Spanish
            "excelente", "bueno", "genial", "amor", "perfecto"
        ]
        
        # 2. MULTILINGUAL NEGATIVE DICTIONARY (Tech errors, damage, annoyances)
        self.neg_words = [
            # English
            "crack", "broke", "error", "crash", "delay", "fail", "stuck", "miss", 
            "late", "damage", "deadline", "traffic", "tax", "monday", "melt", 
            "bug", "ruin", "waste", "useless", "drop", "awful", "terrible", "trash",
            # Hindi
            "kharab", "bekar", "ghatiya", "toot", "bakwas", "raddi",
            # Bengali
            "kharap", "baje", "fokira", "nosto", "bhangcha",
            # Spanish
            "malo", "terrible", "basura", "roto", "error"
        ]

    def extract_aspects(self, text):
        """
        Aspect-Based Sentiment Analysis (ABSA)
        Uses SpaCy to extract the main nouns/subjects being discussed.
        """
        doc = nlp(text)
        aspects = []
        for chunk in doc.noun_chunks:
            # Filter out basic pronouns to get actual business aspects (e.g., "the battery", "my flight")
            if chunk.root.pos_ == "NOUN":
                aspects.append(chunk.text)
        return list(set(aspects)) # Return unique aspects

    def analyze_text(self, text):
        # A. GET THE AI's LITERAL READING
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label_idx = torch.argmax(scores).item()
        
        labels = ["Negative", "Neutral", "Positive"]
        base_sentiment = labels[label_idx]
        confidence = torch.max(scores).item() * 100
        
        # B. THE MASTER CONTRAST LOGIC (Multilingual)
        text_lower = text.lower()
        is_sarcastic = False
        
        has_pos = any(p in text_lower for p in self.pos_words)
        has_neg = any(n in text_lower for n in self.neg_words)

        # Contrast Rules
        if has_pos and has_neg:
            is_sarcastic = True
        elif base_sentiment == "Negative" and has_pos:
            is_sarcastic = True
        elif base_sentiment == "Positive" and has_neg:
            is_sarcastic = True
        elif "..." in text and has_pos:
            is_sarcastic = True

        # C. ASPECT EXTRACTION (ABSA)
        aspects = self.extract_aspects(text)

        # D. FINAL VERDICT
        final_sentiment = "Sarcastic / Mocking" if is_sarcastic else base_sentiment

        return {
            "input_text": text,
            "base_sentiment": base_sentiment,
            "final_sentiment": final_sentiment,
            "is_sarcastic": is_sarcastic,
            "aspects_detected": aspects if aspects else ["None Detected"],
            "confidence_score": f"{confidence:.2f}%"
        }