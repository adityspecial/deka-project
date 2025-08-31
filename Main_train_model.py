# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib  # For saving and loading the model

# Step 1: Load Data
data = pd.read_excel("Main_final.xlsx")

# Step 2: Clean Data (Remove duplicates)
data_cleaned = data.drop_duplicates()

# Step 3: Prepare Input and Output
X = data_cleaned[['Diameter', 'ContactAngle', 'x']]
y = data_cleaned['y']

# Step 4: Split Data (for training and testing purposes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Multiple Models
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42)
}

mse_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_test)
    mse_scores[name] = mse
    print(f"Model: {name}, Mean Squared Error (Test set): {mse}")

# Step 6: Find the Best Model
best_model_name = min(mse_scores, key=mse_scores.get)
best_model = models[best_model_name]
print(f"The best model is: {best_model_name} with MSE: {mse_scores[best_model_name]}")

# Step 7: Save the best model
model_filename = 'best_model.joblib'
joblib.dump(best_model, model_filename)
print(f"Best model saved as {model_filename}")
