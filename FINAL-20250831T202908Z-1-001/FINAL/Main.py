import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Step 2: Load Data
data = pd.read_excel("Main.xlsx")

# Step 3: Data Preprocessing (if needed)

# Step 4: Split Data
X = data[['Diameter', 'ContactAngle', 'x']]
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Choose Models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}

# Step 6: Train and Evaluate Models
results = {}
for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = (-cv_scores)**0.5
    avg_rmse = rmse_scores.mean()
    results[name] = avg_rmse

# Step 7: Select Model with Least Error
best_model = min(results, key=results.get)
print("Best Model:", best_model)

# Train the best model
best_model_instance = models[best_model]
best_model_instance.fit(X_train, y_train)

# Step 8: Evaluate the Best Model on Test Set
y_pred = best_model_instance.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (Test set):", mse)

# Step 9: Predict Output (for new data)

Diameter = float(input("Enter Diameter of Drop in mm"))
Contact_Angle = float(input("Enter Contact Angle"))
if Contact_Angle < 90:
    test = Diameter / 2000.0
elif Contact_Angle == 90:
    test = Diameter / 2000.0
else:
    test = Diameter/1000
k=0
ar=[]
i_values= []
i = 0
while i < test:
    new_inputs = [[Diameter, Contact_Angle, i]]
    predicted_output = best_model_instance.predict(new_inputs)
    ar.append(predicted_output)
    k += 1
    i_values.append(i)
    i += 0.00001  # Increment i by the desired step size

flattened_ar = [val for sublist in ar for val in sublist]

combined_data = list(zip(i_values, flattened_ar))

# Convert the combined data to a DataFrame
df = pd.DataFrame(combined_data)

# Define the filename for the Excel file
excel_file = "output.xlsx"

# Export the DataFrame to an Excel file
df.to_excel(excel_file, index=False, header=False)

print("Data has been exported to", excel_file)

#print("Predicted Output:", predicted_output)
