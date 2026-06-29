# 🏠 Bangalore House Price Prediction

A Machine Learning web application that predicts house prices in Bangalore based on property features such as location, total square feet, number of bedrooms (BHK), and bathrooms. The application is built using **Python**, **Flask**, and **Scikit-learn**.

---

# 📌 Project Overview

Real estate prices vary depending on several factors such as location, size, and amenities. This project uses machine learning to estimate the price of a house in Bangalore based on user-provided inputs.

The trained model is integrated into a Flask web application, allowing users to get instant house price predictions through a simple web interface.

---

# 🚀 Features

- 🏡 Predict Bangalore house prices
- 📍 Location-based price prediction
- 📐 Area (Square Feet) input
- 🛏️ BHK selection
- 🚿 Bathroom selection
- 🌐 User-friendly Flask web interface
- ⚡ Fast and accurate predictions

---

# 🛠️ Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Pickle
- HTML
- CSS
- JavaScript

---

# 📂 Project Structure

```text
Bangalore-House-Price-Prediction/
│
├── static/                # CSS, JavaScript and Images
├── templates/             # HTML templates
│
├── Procfile               # Deployment configuration
├── RidgeModel.pkl         # Trained Machine Learning model
├── app.py                 # Flask application
├── requirements.txt       # Required Python libraries
├── README.md              # Project documentation
```

---

# 📊 Input Features

The model predicts house prices using the following features:

- 📍 Location
- 📐 Total Square Feet
- 🛏️ Number of Bedrooms (BHK)
- 🚿 Number of Bathrooms

### 🎯 Output

Predicted House Price (in Lakhs ₹)

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Aditya-12-k/Banglore-House-Price-Prediction.git
```

## 2️⃣ Navigate to the Project Directory

```bash
cd Banglore-House-Price-Prediction
```

## 3️⃣ Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

### Activate Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

## 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Flask server:

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000/
```

---

# 🧠 Machine Learning Workflow

1. Collect house price dataset
2. Data Cleaning
3. Feature Engineering
4. Data Preprocessing
5. Train Ridge Regression Model
6. Save Trained Model
7. Build Flask Web Application
8. Predict House Price

---

# 📈 Model

The project uses the **Ridge Regression** algorithm for house price prediction.

The trained model is stored in:

```
RidgeModel.pkl
```

---



# 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Major libraries:

- Flask
- Pandas
- NumPy
- Scikit-learn

---

# 🔮 Future Improvements

- Interactive Dashboard
- Google Maps Location Integration
- More Accurate Models (XGBoost, Random Forest)
- Deploy on AWS, Render, or Railway
- Price Trend Visualization
- User Authentication

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push the branch
5. Create a Pull Request

---


---

⭐ If you found this project helpful, please consider giving it a **Star ⭐** on GitHub!
