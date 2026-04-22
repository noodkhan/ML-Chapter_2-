# 🎬 Movie Rating Prediction System

A complete machine learning pipeline that trains and deploys a model to predict IMDb movie ratings using structured movie data.

---

## 🚀 Overview

This project implements an **end-to-end machine learning system**, covering:

* Model training with feature engineering
* Model serialization (save/load)
* Prediction pipeline with schema alignment
* Real-time inference simulation
* Model evaluation and visualization

Built using scikit-learn, this project focuses on **data consistency and system design**, not just model accuracy.

---

## 🧠 System Architecture

```text
Raw CSV Data
     ↓
Data Cleaning + Feature Engineering
     ↓
Train/Test Split
     ↓
Model Training (Random Forest)
     ↓
Save Model + Schema
     ↓
Inference Pipeline (Rebuild Features)
     ↓
Prediction + Visualization
```

---

## 🏗️ Project Structure

```text
.
├── train.py               # Training pipeline
├── predict.py             # Inference pipeline
├── model.pkl              # Trained model
├── columns.pkl            # Feature schema
├── imdb_top_1000.csv      # Inference dataset
├── IMDB-Movie-Data.csv    # Training dataset
└── README.md
```

---

## ⚙️ Training Pipeline

### 📥 Load & Prepare Data

```python
df = pd.read_csv("IMDB-Movie-Data.csv")
df = df.dropna()
```

---

### 🧹 Feature Selection

```python
df = df[[
    "Genre",
    "Runtime (Minutes)",
    "Votes",
    "Revenue (Millions)",
    "Metascore",
    "Rating"
]]
```

---

### 🔄 Feature Engineering

```python
genre_dummies = df["Genre"].str.get_dummies(sep=",")
df = pd.concat([df.drop("Genre", axis=1), genre_dummies], axis=1)
```

---

### ✂️ Train/Test Split

```python
X = df.drop("Rating", axis=1)
y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

### 🌲 Model Training

```python
model = RandomForestRegressor(n_estimators=200, max_depth=10)
model.fit(X_train, y_train)
```

---

### 💾 Save Model

```python
joblib.dump(model, "model.pkl")
joblib.dump(X.columns, "columns.pkl")
```

👉 Saving `columns.pkl` ensures **feature consistency during inference**.

---

## 🔮 Prediction Pipeline

### 🧩 Load Model

```python
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")
```

---

### 🔁 Align Input Data

```python
df_input = df_input.reindex(columns=columns, fill_value=0)
```

👉 This guarantees:

* Same feature order
* Missing features filled with 0

---

### 📊 Predict

```python
prediction = model.predict(df_input)
```

---

## ⚡ Real-Time Prediction Simulation

```python
for i in range(n):
    row = df.iloc[[i]]
    pred = model.predict(row)[0]
    print(f"Pred: {pred:.2f}")
```

Simulates a **streaming inference system**.

---

## 📈 Model Evaluation

### 🔍 Single Prediction Check

```python
movie_df = X_test.sample(1)

pred = model.predict(movie_df)[0]
actual = y_test.loc[movie_df.index[0]]
```

---

### 📊 Scatter Plot

```python
plt.scatter(y_test, preds)
```

---

### 📉 Perfect Prediction Line

```python
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')
```

---

## 🧪 Key Observations

* Model tends to predict near average (~8.0)
* Indicates **limited feature signal**
* Stronger features → better predictions

---

## ⚠️ Limitations

* Basic feature set (missing director, actors, year)
* No hyperparameter tuning
* Limited generalization testing
* Genre encoding may introduce sparsity

---

## 🔥 Future Improvements

* Add high-impact features:

  * Director
  * Cast
  * Release year
* Apply log scaling to votes
* Hyperparameter tuning
* Build REST API for predictions
* Deploy as web dashboard

---

## 🧩 Tech Stack

* Python
* Pandas
* Matplotlib
* Joblib
* scikit-learn

---

## 🧠 Key Insight

> Machine learning is not about the model —
> it's about making data compatible with the model.

---

## 👨‍💻 Author

Focused on:

* System design thinking
* Data pipelines
* Building scalable ML systems

---

# 🚀 Optional (if you want next level)

I can upgrade this README to:

* ⭐ GitHub badges
* 📸 Graph screenshots section
* 🎥 Demo GIF (huge for portfolio)
* 📦 “How to run” section (very important)

Just say **“make it portfolio-level”** and I’ll push it further.

<img width="1037" height="834" alt="Screenshot 2026-04-23 004227" src="https://github.com/user-attachments/assets/b7e97d4d-22d1-49d0-91e0-39ea2dbf8a54" />
