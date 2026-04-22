🎬 Movie Rating Prediction System

A machine learning pipeline that predicts IMDb movie ratings based on structured features such as genre, runtime, votes, revenue, and metascore.

🚀 Overview

This project demonstrates a complete end-to-end ML inference pipeline, including:

Data preprocessing & cleaning
Feature engineering (one-hot encoding)
Schema alignment with trained model
Real-time prediction simulation
Visualization of model performance

The system takes raw movie data and transforms it into a format suitable for prediction, ensuring consistency with the trained model.

🧠 Features
✅ Data cleaning (string → numeric transformation)
✅ Genre encoding (multi-label → one-hot)
✅ Column alignment using trained schema
✅ Real-time prediction simulation
✅ Visualization (Actual vs Predicted)
✅ Model evaluation (MSE ready)
🏗️ Project Structure
.
├── model.pkl              # Trained ML model
├── columns.pkl            # Training feature schema
├── imdb_top_1000.csv      # Dataset
├── main.py                # Main pipeline script
└── README.md              # Documentation
⚙️ How It Works
1. Load Model & Schema
model = joblib.load("model.pkl")
trained_columns = joblib.load("columns.pkl")
2. Data Preprocessing
Rename columns to match training format
Convert:
Runtime → integer
Votes → integer
Revenue → float (millions)
Handle missing values
3. Feature Engineering
genre_Hotcode = df["Genre"].str.get_dummies(sep=", ")
Multi-genre → binary columns
4. Schema Alignment (Critical Step)
df = df.reindex(columns=trained_columns, fill_value=0)

Ensures:

Same feature order as training
Missing features filled with 0
5. Prediction
predictions = model.predict(df)
6. Real-Time Simulation
for i in range(n):
    pred = model.predict(row)[0]
    print(f"Pred: {pred:.2f} | Actual: {actual}")

Simulates a streaming prediction system.

7. Visualization
plt.plot(actual_ratings[:n])
plt.plot(predictions[:n])

Shows model performance over time.

📊 Example Output
[0] Pred: 8.67 | Actual: 9.2
[1] Pred: 8.68 | Actual: 9.0
[2] Pred: 8.83 | Actual: 8.6
...
📈 Visualization
Line chart comparing predicted vs actual ratings
Helps identify:
Bias toward average
Prediction variance
Model limitations
🧪 Evaluation
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(actual_ratings, predictions)
print("MSE:", mse)
⚠️ Limitations
Model tends to predict near average (regression to mean)
Limited feature set reduces prediction accuracy
Evaluation performed on same dataset (not true generalization)
🔥 Future Improvements
Add stronger features:
Director
Cast
Release year
Awards
Use train/test split for proper evaluation
Upgrade model (e.g., Random Forest)
Build API for real-time predictions
Create interactive dashboard
🧩 Tech Stack
Python
Pandas
NumPy
Matplotlib
Joblib
Scikit-learn
🧠 Key Insight

The hardest part of machine learning is not the model —
it's making real-world data match the model’s expectations.

📌 Author

Built as part of a deep dive into:

Machine Learning pipelines
Data engineering
System design thinking
