import pandas as pd
from sklearn.model_selection import train_test_split

def clean_data(df):
    # Supprimer les lignes avec valeurs manquantes
    df_clean = df.dropna()
    return df_clean

def encode_categoricals(X, encoders=None):
    # Encodage basé sur les données d'entraînement
    X_encoded = X.copy()
    if encoders is None:
        encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                X_encoded[col] = X[col].astype('category').cat.codes
                encoders[col] = dict(enumerate(X[col].astype('category').cat.categories))
    else:
        for col in X.columns:
            if X[col].dtype == 'object' and col in encoders:
                X_encoded[col] = X[col].map(encoders[col]).fillna(-1).astype('category').cat.codes
    return X_encoded, encoders

def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Vérifier si stratification possible
    stratify = None
    counts = y.value_counts()
    if counts.min() >= 2:
        stratify = y  # stratification possible

    # Split sans encoder pour éviter les erreurs
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.33, stratify=stratify, random_state=42)

    # Encoder avec la fonction existante, on récupère les encodeurs du train
    X_train, encoders = encode_categoricals(X_train_raw)

    # Pour le test, on encode en utilisant les encodeurs appris sur le train
    X_test, _ = encode_categoricals(X_test_raw, encoders=encoders)

    return X_train, X_test, y_train, y_test
