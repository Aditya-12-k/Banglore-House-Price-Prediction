# ------------------------------------------------
# Import Required Libraries
# ------------------------------------------------
from flask import Flask, render_template, request
import pickle
import pandas as pd
import os
import json

# ------------------------------------------------
# Create Flask Application
# ------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------
# Load Trained Pipeline Model
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "RidgeModel.pkl")

model = pickle.load(open(model_path, "rb"))

# ------------------------------------------------
# Load Column Names (for Dropdown Locations)
# IMPORTANT: Create columns.json during training
# ------------------------------------------------
columns_path = os.path.join(BASE_DIR, "columns.json")

with open(columns_path, "r") as f:
    data_columns = json.load(f)["data_columns"]

# First columns are numeric → location columns start after them
locations = data_columns[3:]   # ['location_Whitefield', etc.]

# Clean location names (remove prefix)
locations = [loc.replace("location_", "") for loc in locations]


# ------------------------------------------------
# Home Route
# ------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", locations=locations)


# ------------------------------------------------
# Prediction Route
# ------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ----------------------------------------
        # Get form values
        # ----------------------------------------
        location = request.form["location"]
        total_sqft = float(request.form["total_sqft"])
        bath = float(request.form["bath"])
        bhk = int(request.form["bhk"])

        # ----------------------------------------
        # Create DataFrame (must match training format)
        # ----------------------------------------
        input_df = pd.DataFrame(
            [[location, total_sqft, bath, bhk]],
            columns=['location', 'total_sqft', 'bath', 'bhk']
        )

        # ----------------------------------------
        # Predict
        # ----------------------------------------
        prediction = model.predict(input_df)[0]

        result = f"Estimated Price: ₹ {round(prediction, 2)} Lakhs"

    except Exception as e:
        result = f"Error: {str(e)}"

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