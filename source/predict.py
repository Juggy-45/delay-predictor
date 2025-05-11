import joblib
import pandas as pd

# Step 1: Load trained model and encoders
model = joblib.load("../model/delay_model.pkl")
label_encoders = joblib.load("../model/label_encoders.pkl")

# Step 2: Create a prediction function
def predict_delay(input_data: dict):
    """
    input_data: dictionary with keys:
        'Project_Type', 'County', 'Planned_Duration',
        'Actual_Progress (%)', 'RFIs', 'Rain_Days'
    """
    df = pd.DataFrame([input_data])

    for col in ['Project_Type', 'County']:
        if col in df:
            le = label_encoders[col]
            df[col] = le.transform(df[col])


    prediction = model.predict(df)[0]
    prediction_proba = model.predict_proba(df)[0]


    result = "Delayed" if prediction == 1 else "On Track"
    confidence = round(prediction_proba[prediction] * 100, 2)

    return {
        "result": result,
        "confidence": f"{confidence}%",
        "raw_prediction": prediction
    }

if __name__ == "__main__":
    sample_input = {
        "Project_Type": "Residential",
        "County": "Dublin",
        "Planned_Duration": 180,
        "Actual_Progress (%)": 65,
        "RFIs": 18,
        "Rain_Days": 14
    }

    output = predict_delay(sample_input)
    print("Prediction:", output)
