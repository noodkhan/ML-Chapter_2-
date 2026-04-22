import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
import time
import matplotlib.pyplot as plt

# ! # STEP 1 : LOAD the "Brain"
model = joblib.load("model.pkl")
trained_columns = joblib.load("columns.pkl")

# ! STEP 2 : LOAD DATA
df = pd.read_csv("imdb_top_1000.csv")

# key have to be the same as model(x , y , z)
# df = df[[
#     "Genre",
#     "Runtime (Minutes)",
#     "Votes",
#     "Revenue (Millions)",
#     "Metascore",
#     "Rating"
# ]]

    #   '', '', '', '',
    #    'Runtime', 'Genre', 'IMDB_Rating', 
    # '', 'Meta_score', '',
    #    '', '', '', '', 'No_of_Votes', 'Gross'],

# column_mapping = {
#     "Runtime": "Runtime (Minutes)",
#     "No_of_Votes": "Votes",
#     "Meta_score": "Metascore",
#     "Gross": "Revenue (Millions)"
# }


# ! STEP 3 : Rename the columns in your new dataframe
column_mapping = {
    "Runtime": "Runtime (Minutes)",
    "No_of_Votes": "Votes",
    "Meta_score": "Metascore",
    "Gross": "Revenue (Millions)" , 
    "IMDB_Rating" : "Rating"
}
df = df.rename(columns=column_mapping)

# ! STEP 4 :  Prepare the data (Assuming 'df' contains the 5 rows above)
# We store the real ratings in a separate series for comparison
actual_ratings = df['Rating']
print(actual_ratings.head())

# ! STEP 5 : Clean Data (Filter)
df = df[[
        "Genre",                # 1
        "Runtime (Minutes)",    # 1
        "Votes",                # 1
        "Revenue (Millions)",   # 1
        "Metascore",            # 1
        # "Rating"                # 1
    ]]

# ! STEP 6. Combine and Clean
df['Runtime (Minutes)'] = (
    df['Runtime (Minutes)']
    .astype(str)
    .str.replace(" min", "")
    .replace("nan", "0")
    .astype(int)
)
df['Revenue (Millions)'] = (
    df['Revenue (Millions)']
    .fillna("0")
    .str.replace(",", "")
    .astype(float) / 1_000_000
)

genre_Hotcode = df["Genre"].str.get_dummies(sep=", ")
df = pd.concat([df.drop("Genre", axis=1), genre_Hotcode], axis=1)

# # 4. THE MAGIC REINDEX
# # This makes sure 'Drama' and 'Crime' are in the right columns 
# # and fills 'Action', 'Horror', etc. with 0.
df = df.reindex(columns=trained_columns, fill_value=0)

n = 50

# # 5. PREDICT
predictions = model.predict(df)
import time
for i in range(n):
    row = df.iloc[[i]]  # keep as DataFrame
    pred = model.predict(row)[0]
    actual = actual_ratings.iloc[i]
    print(f"[{i}] Pred: {pred:.2f} | Actual: {actual}")
    time.sleep(0.3)


# take first 50 samples for clarity
plt.figure()
plt.plot(actual_ratings[:n].values)
plt.plot(predictions[:n])
plt.title("Actual vs Predicted Ratings")
plt.xlabel("Movie Index")
plt.ylabel("Rating")
plt.legend(["Actual", "Predicted"])
plt.show()