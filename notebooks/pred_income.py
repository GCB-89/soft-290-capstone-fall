import joblib
import pandas as pd
import json

# Load model and encoder
model = joblib.load('models/income_model.pkl')
encoder = joblib.load('models/encoder.pkl')

# Education cost lookup dictionary
education_costs = {
    "High School": 0,
    "Associate Degree": 30000,
    "Bachelor's Degree": 85000,
    "Master's Degree": 120000,
    "PhD": 180000
}

def get_education_cost(education_level):
    return education_costs.get(education_level, None)

def predict_income(career_field, career_level, education_level):
    # Create a DataFrame from user input
    input_df = pd.DataFrame([{
        'CareerField': career_field,
        'CareerLevel': career_level,
        'EducationLevel': education_level
    }])

    # Encode the input
    encoded_input = encoder.transform(input_df)
    encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out())

    # Predict income
    predicted_income = model.predict(encoded_df)[0]

    # Lookup education cost
    education_cost = get_education_cost(education_level)

    # Format as JSON
    result = {
        "careerField": career_field,
        "careerLevel": career_level,
        "educationLevel": education_level,
        "predictedMonthlyIncome": round(predicted_income, 2),
        "estimatedEducationCost": education_cost
    }

    return json.dumps(result, indent=4)

# Example usage
if __name__ == "__main__":
    print(predict_income("Construction", "Mid", "Associate Degree"))
