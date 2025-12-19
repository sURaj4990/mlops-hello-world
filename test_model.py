import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def test_model_accuracy():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df = pd.read_csv(url, sep=";")

    X = df.drop("quality")
    y = (df["quality"]).astype(int)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load("model.pkl")

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy > 0.70, f"Model accuracy is too low: {accuracy}"