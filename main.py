# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Separate target and features
target_column = 'SalePrice'
numeric_cols = train_data.select_dtypes(include=[np.number]).columns.drop(target_column)
non_numeric_cols = train_data.select_dtypes(exclude=[np.number]).columns

# Fill missing values
# Numeric columns
train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].mean())
test_data[numeric_cols] = test_data[numeric_cols].fillna(train_data[numeric_cols].mean())

# Non-numeric columns
for col in non_numeric_cols:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])
    test_data[col] = test_data[col].fillna(train_data[col].mode()[0])

# Ensure test data columns match train data columns
test_data = test_data.reindex(columns=train_data.columns.drop(target_column), fill_value=0)

# Separate features and target variable
X = train_data.drop(target_column, axis=1)
y = train_data[target_column]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: OneHotEncode categorical columns and scale numeric columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), non_numeric_cols)
    ])

# Create pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_val_pred = pipeline.predict(X_val)
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")

# Predict on test data
test_predictions = pipeline.predict(test_data)

# Save predictions to a CSV file
output = pd.DataFrame({'Id': test_data.index, 'PredictedPrice': test_predictions})
output.to_csv('predictions.csv', index=False)
print("\nPredictions saved to 'predictions.csv'.")

# Save evaluation metrics
with open('metrics.txt', 'w') as f:
    f.write(f"Mean Squared Error (MSE): {mse}\n")
    f.write(f"R-squared: {r2}\n")
print("Metrics saved to 'metrics.txt'.")
