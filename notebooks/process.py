import joblib
import numpy as np
import re
import os
import json
# Load model and encoders
bundle = joblib.load("Models/Gradient Boosting_bundle.pkl")  # Replace with your actual filename if different
loaded_model = bundle["model"]
career_encoder = bundle["career_encoder"]
education_encoder = bundle["education_encoder"]

# Define options
career_options = {
    "1": "Construction",
    "2": "Media",
    "3": "Engineering",
    "4": "Education",
    "5": "Healthcare"
}

education_options = {
    "1": "High School",
    "2": "Associate's",
    "3": "Bachelor's",
    "4": "Master's",
    "5": "PhD"
}

# Helpers
def _normalize_label(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def safe_encode(encoder, label, fallback_classes=None):
    if label in encoder.classes_:
        return int(encoder.transform([label])[0])
    norm_map = { _normalize_label(c): c for c in encoder.classes_ }
    nl = _normalize_label(label)
    if nl in norm_map:
        return int(encoder.transform([norm_map[nl]])[0])
    for c in encoder.classes_:
        if nl in _normalize_label(c) or _normalize_label(c) in nl:
            return int(encoder.transform([c])[0])
    fallback = fallback_classes.mode().iloc[0] if fallback_classes is not None else encoder.classes_[0]
    print(f"⚠️ Warning: '{label}' not recognized. Falling back to '{fallback}'.")
    return int(encoder.transform([fallback])[0])

# Get user input
print("Select a career:")
for key, value in career_options.items():
    print(f"{key}: {value}")
career_choice = input("Enter the number for your career: ").strip()
career = career_options.get(career_choice, "Construction")

print("\nSelect an education level:")
for key, value in education_options.items():
    print(f"{key}: {value}")
education_choice = input("Enter the number for your education level: ").strip()
education = education_options.get(education_choice, "High School")

# Encode input
career_encoded = safe_encode(career_encoder, career)
education_encoded = safe_encode(education_encoder, education)

# Predict
input_data = np.array([[career_encoded, education_encoded]])
predicted_income = loaded_model.predict(input_data)[0]
print(f"\n💼 Predicted Monthly Income for {career} with {education}: ${predicted_income:.2f}")

# Format output
output = {
    "career": career,
    "education": education,
    "career_encoded": career_encoded,
    "education_encoded": education_encoded,
    "predicted_monthly_income": round(float(predicted_income), 2)
}

# ✅ Save to JSON file in 'json' folder (relative to script; falls back to CWD if __file__ isn't set)

try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.getcwd()

json_dir = os.path.join(base_dir, "json")
os.makedirs(json_dir, exist_ok=True)
json_path = os.path.join(json_dir, "prediction_output.json")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print(f"\n📁 Prediction saved to {json_path}")

