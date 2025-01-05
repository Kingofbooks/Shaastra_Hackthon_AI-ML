from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

def train_and_save_model(data_path, output_model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["credit_risk_label"])
    y = data["credit_risk_label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, output_model_path)
    print(f"Model saved to {output_model_path}")
