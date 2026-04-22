import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def train_model():
    df = pd.read_csv("IMDB-Movie-Data.csv")
    df = df.dropna()

    df = df[[
        "Genre",
        "Runtime (Minutes)",
        "Votes",
        "Revenue (Millions)",
        "Metascore",
        "Rating"
    ]]

    # one-hot encode
    genre_dummies = df["Genre"].str.get_dummies(sep=",")
    df = pd.concat([df.drop("Genre", axis=1), genre_dummies], axis=1)
    print(df.columns)

    X = df.drop("Rating", axis=1)
    y = df["Rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200,max_depth=10)
    model.fit(X_train, y_train)

    # SAVE model + columns
    joblib.dump(model, "model.pkl")
    joblib.dump(X.columns, "columns.pkl")
    print("Model saved!")

    return X_test, y_test  # 👈 IMPORTANT

X_test, y_test = train_model()

def load_model():
    model = joblib.load("model.pkl")
    columns = joblib.load("columns.pkl")
    return model, columns

model, columns = load_model()

def predict_movie(model, columns, input_data):
    df_input = pd.DataFrame([input_data])
    df_input = df_input.reindex(columns=columns, fill_value=0)
    return model.predict(df_input)[0]


movie_df = X_test.sample(1)

pred = model.predict(movie_df)[0]
actual = y_test.loc[movie_df.index[0]]

print("Predicted rating:", pred)
print("Actual:", actual)

preds = model.predict(X_test)
plt.scatter(y_test, preds)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted")

# perfect prediction line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')

plt.show()