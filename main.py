import numpy as np
import pandas as pd
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

    if df is not None:
        print("Processing data...")
        print("\nFirst few rows:")
        print(df.head())

        # Define target and initial feature set
        y = df['isFraud']
        # Drop identifier and target columns from features
        X = df.drop(['isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest'], axis=1, errors='ignore')
     
        # Create and show a plot
        if 'type' in df.columns:
            fig = px.pie(df, names='type', title='Distribution of Transaction Types')
            fig.show()
        else:
            print("No 'type' column found in the dataset")
        
        # Encode categorical features (like 'type') and ensure numeric-only input
        obj_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if obj_cols:
            X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

        # Ensure there are no non-numeric columns left
        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            print(f"Warning: non-numeric columns present and will be dropped: {non_numeric}")
            X = X.drop(columns=non_numeric)

        # Align X and y and split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2
        )
        clf = DecisionTreeClassifier(random_state=1)
        clf.fit(X_train, y_train)
        print("X_test:", X_test[:100])
        print("Test score:", clf.score(X_test, y_test))
        print("Sample predictions:", clf.predict(X_test)[:100])
    else:
        print("Could not proceed without data.")



if __name__ == "__main__":  # This ensures the code runs only when executed directly
    main()
