import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pathlib import Path
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier

# Define the data path relative to the script location
DATA_PATH = Path(__file__).parent / 'data' / 'transactions.csv'

def load_data():
    print("Loading data...")
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Data loaded successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Data file not found at {DATA_PATH}")
        return None

def main():
    print("Starting fraud detection analysis...")
    # Load and check data
    df = load_data()


    y = df['isFraud']
    x = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)

    print(df)

    if df is not None:
        print("Processing data...")
        print("\nFirst few rows:")
        print(df.head())
        
        a = df.columns
        # Create and show a plot
        if 'type' in df.columns:
            fig = px.pie(df, names='type', title='Distribution of Transaction Types')
            fig.show()
        else:
            print("No 'type' column found in the dataset")
        
        print(df)

        X_train, X_test, y_train, y_test = train_test_split(
            df[df.columns],
            df.values,
            test_size=0.2,
            random_state=42
        )
        clf = DecisionTreeClassifier(random_state=1)
        clf.fit(X_train, y_train)
        clf.score(X_test, y_test)
        clf.predict(X_test)

    else:
        print("Could not proceed without data.")



if __name__ == "__main__":  # This ensures the code runs only when executed directly
    main()
