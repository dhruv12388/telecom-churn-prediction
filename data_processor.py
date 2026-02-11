import pandas as pd
from sklearn.model_selection import train_test_split

def get_clean_data():
    # Load the CSV you downloaded
    df = pd.read_csv('data.csv')
    
    # AI only likes numbers. We pick 3 important ones:
    # 1. Tenure (How long they've been a customer)
    # 2. MonthlyCharges (How much they pay)
    # 3. TotalCharges (Sum of all payments)
    
    # We convert 'TotalCharges' to numbers because it's often saved as text
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.fillna(0) # If any data is missing, fill with 0
    
    # Choose our 'Features' (X) and our 'Answer' (y)
    X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Split: 80% for learning, 20% for testing
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    print("Data processor is ready!")