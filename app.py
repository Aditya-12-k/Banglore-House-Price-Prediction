# ------------------------------------------------
# Import Required Libraries
# ------------------------------------------------
from flask import Flask, render_template, request
import pickle
import pandas as pd

# ------------------------------------------------
# Create Flask Application
# ------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------
# Load Trained Pipeline Model
# (Pipeline includes preprocessing + Ridge model)
# ------------------------------------------------
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "RidgeModel.pkl")

model = pickle.load(open(model_path, "rb"))
# ------------------------------------------------
# Load Dataset to Extract Locations for Dropdown
# ------------------------------------------------
data = pd.read_csv("Bengaluru_House_Data.csv")

# Clean location column (handle missing values)
data['location'] = data['location'].fillna('other')

# Get sorted unique location list
locations = sorted(data['location'].unique())


# ------------------------------------------------
# Home Route
# Sends location list to HTML
# ------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", locations=locations)


# ------------------------------------------------
# Prediction Route
# Handles form submission
# ------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ----------------------------------------
        # Get values from form
        # ----------------------------------------
        location = request.form["location"]
        total_sqft = float(request.form["total_sqft"])
        bath = float(request.form["bath"])
        bhk = int(request.form["bhk"])

        # ----------------------------------------
        # Convert input into DataFrame
        # IMPORTANT:
        # Column names must match training data
        # ----------------------------------------
        input_df = pd.DataFrame(
            [[location, total_sqft, bath, bhk]],
            columns=['location', 'total_sqft', 'bath', 'bhk']
        )

        # ----------------------------------------
        # Make prediction
        # ----------------------------------------
        prediction = model.predict(input_df)[0]

        result = f"Estimated Price: ₹ {round(prediction, 2)} Lakhs"

    except Exception as e:
        # If any error occurs, show it on UI
        result = f"Error: {str(e)}"

    # ----------------------------------------
    # Return result to HTML
    # ----------------------------------------
    return render_template(
        "index.html",
        locations=locations,
        prediction_text=result
    )


# ------------------------------------------------
# Run Flask App
# ------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)