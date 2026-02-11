import xgboost as xgb
import pickle
from data_processor import get_clean_data

# 1. Get the cleaned data from our other file
X_train, X_test, y_train, y_test = get_clean_data()

# 2. Create the AI (The 'Model')
# XGBoost is the specific math algorithm we are using
brain = xgb.XGBClassifier()

# 3. Training: The AI looks at the data and finds patterns
print("The AI is studying patterns in your data... please wait.")
brain.fit(X_train, y_train)

# 4. Save the Brain: This creates a file on your laptop
# 'wb' means 'Write Binary' (saving the file)
with open('customer_ai.pkl', 'wb') as f:
    pickle.dump(brain, f)

print("Success! Your AI is saved as 'customer_ai.pkl'.")
import matplotlib.pyplot as plt
from xgboost import plot_importance

# 1. Visualize which features the AI cares about most
print("Generating Feature Importance chart...")
plot_importance(brain)
plt.title("What makes a customer leave?")
plt.savefig('insights.png') # This saves the chart as an image file
plt.show()

print("Chart saved as 'insights.png'!")