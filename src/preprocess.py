import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(traditional_path, alternative_path):
    # Load datasets
    traditional_data = pd.read_csv(traditional_path)
    alternative_data = pd.read_csv(alternative_path)

    # Merge datasets
    data = pd.merge(traditional_data, alternative_data, on="user_id")

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    data.fillna(imputer.fit_transform(data), inplace=True)

    # Scale numerical features
    scaler = MinMaxScaler()
    numerical_features = ["income", "debt_to_income", "sentiment_score"]
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Encode categorical features
    encoder = OneHotEncoder()
    loan_purpose_encoded = encoder.fit_transform(data[["loan_purpose"]])
    data = pd.concat([data, pd.DataFrame(loan_purpose_encoded.toarray())], axis=1)

    return data
