import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Preparation
# Combine Excel files into a single DataFrame
directory = os.getcwd()
file_names = [f for f in os.listdir(directory) if f.endswith('.xlsx')]

dfs = []
for file_name in file_names:
    diameter, contact_angle = file_name.split('_')
    contact_angle = contact_angle.split('.')[0]  # Remove extension
    df = pd.read_excel(os.path.join(directory, file_name), header=None)
    df['Diameter'] = float(diameter) / 1000.0

    #df['Diameter'] = diameter
    df['ContactAngle'] = contact_angle
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Step 2: Feature Engineering
# Extract features and target variable
X = df.iloc[:, :-2]  # All columns except last two (Diameter and ContactAngle)
y = df.iloc[:, -2]   # Last but one column (x values)
z = df.iloc[:, -1]   # Last column (y values)

# Combine X and y to drop rows containing NaN in either X or y
combined_df = pd.concat([X, y], axis=1)
combined_df.dropna(inplace=True)

# Separate X and y again after dropping rows with NaN values
X = combined_df.iloc[:, :-1]
y = combined_df.iloc[:, -1]

# Convert feature names to string
X.columns = X.columns.astype(str)

# Step 3: Model Selection
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Random Forest Regression": RandomForestRegressor(),
    "Gradient Boosting Regression": GradientBoostingRegressor()
}

# Step 4: Model Training and Evaluation
results = {}
for name, model in models.items():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = mse
    print(f"{name}: Mean Squared Error = {mse}")

# Step 5: Model Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.xlabel('Model')
plt.ylabel('Mean Squared Error')
plt.title('Comparison of Regression Models')
plt.xticks(rotation=45)
plt.show()

# Step 6: Prediction for User-defined inputs
Diameter_value = input("Enter Diameter: ")
ContactAngle_value = input("Enter Contact Angle: ")
x_value = float(input("Enter the x value for prediction: "))

best_model_name = min(results, key=results.get)
best_model = models[best_model_name]

best_model.fit(X, y)
y_pred = best_model.predict([[Diameter_value, ContactAngle_value, x_value]])

print(f"Predicted y value for x = {x_value}, Diameter = {Diameter_value}, Contact Angle = {ContactAngle_value} using {best_model_name}: {y_pred[0]}")
