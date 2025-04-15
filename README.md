# ISEM503-project3-real_estate_app
This is the project prototype for ISEM 503 project 3

# 🏠 Real Estate Price Prediction App

This is a Streamlit-based web application that predicts the **log-transformed price of a house** based on user inputs such as bedrooms, bathrooms, city, status, and more. It uses a **trained Random Forest Regressor pipeline** with preprocessing (categorical encoding, feature engineering) built into a single `.pkl` file for easy deployment.

---

## 🚀 Features

- Predicts log-transformed house prices (converted to actual dollar values).
- Clean UI with dropdowns and sliders for easy input.
- Uses a trained machine learning model (Random Forest).
- Integrated preprocessing pipeline using `sklearn.pipeline.Pipeline`.
- Handles unseen categorical values gracefully (OrdinalEncoder with unknown handling).

---

## 🧰 Tech Stack

- Python 🐍
- Streamlit 📺
- scikit-learn ⚙️
- pandas & numpy 🧮
- joblib (for model persistence)

---

## 📦 File Structure

ISEM503-project3-real_estate_app
|
|--app.py
|--real_estate_tuned_pipeline_model.pkl ## Trained model
|_requirements.txt

## 💻 How to Run the App

### 1. Clone the repository
```bash
git clone https://github.com/talmeezahmed/ISEM503-project3-real_estate_app.git
cd ISEM503-project3-real_estate_app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
### 3. Run the Streamlit app

```bash
streamlit run app.py
```

