import pandas as pd


def add_tenure_group(df):
    
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1y", "1-2y", "2-4y", "4-6y"]
    )
    
    return df

def add_charge_level(df):
    
    df["high_charges"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
    
    return df

def add_engagement_score(df):
    
    df["engagement_score"] = df["tenure"] * df["MonthlyCharges"]
    
    return df

def add_service_count(df):
    
    services = [
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]
    
    df["service_count"] = df[services].apply(lambda x: (x == "Yes").sum(), axis=1)
    
    return df

def feature_engineering(df):
    df = add_tenure_group(df)
    df = add_charge_level(df)
    df = add_engagement_score(df)
    df = add_service_count(df)
    
    return df






