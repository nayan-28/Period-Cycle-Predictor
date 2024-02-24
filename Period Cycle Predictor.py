import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

# Load the training data
file_path = r"E:\Thesis\Health Care\Meanstrual-Cycle.xlsx"
df = pd.read_excel(file_path)

# Replace NaN values in MensesScoreDayFive to MensesScoreDayTen with 0
df[['MensesScoreDayFive', 'MensesScoreDaySix', 'MensesScoreDaySeven', 'MensesScoreDayEight', 'MensesScoreDayNine', 'MensesScoreDayTen']] = df[['MensesScoreDayFive', 'MensesScoreDaySix', 'MensesScoreDaySeven', 'MensesScoreDayEight', 'MensesScoreDayNine', 'MensesScoreDayTen']].fillna(0)

# Convert relevant columns to numeric
numeric_columns = ['EstimatedDayofOvulation', 'LengthofLutealPhase', 'TotalDaysofFertility', 'LengthofMenses', 'MensesScoreDayOne', 'MensesScoreDayTwo', 'MensesScoreDayThree', 'MensesScoreDayFour', 'MensesScoreDayFive', 'MensesScoreDaySix', 'MensesScoreDaySeven', 'MensesScoreDayEight', 'MensesScoreDayNine', 'MensesScoreDayTen']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Check if 'LengthofCycle' column is present
if 'LengthofCycle' not in df.columns:
    raise ValueError("'LengthofCycle' column not found in the DataFrame.")

# Allow user to input features for prediction
user_features = ['LengthofMenses', 'MensesScoreDayOne', 'MensesScoreDayTwo', 'MensesScoreDayThree', 'MensesScoreDayFour', 'MensesScoreDayFive', 'MensesScoreDaySix', 'MensesScoreDaySeven', 'MensesScoreDayEight', 'MensesScoreDayNine', 'MensesScoreDayTen']

# Check if user-provided features are valid
invalid_features = set(user_features) - set(df.columns)
if invalid_features:
    raise ValueError(f"Invalid feature(s): {', '.join(invalid_features)}")

# Define features and target variable
X = df[user_features]
y = df['LengthofCycle']

# Check if there are enough features for training
if len(X.columns) < 2:
    raise ValueError("Insufficient features for training. Check your data.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model (you can choose another model based on your preference)
model = xgb.XGBRegressor()
model.fit(X_train.values, y_train)

# Make predictions on the user-provided features
user_input = pd.DataFrame({feature: [float(input(f"Enter value for {feature}: "))] for feature in user_features})
predicted_length_of_cycle = model.predict(user_input.values)

print(f'Predicted Length of Cycle: {predicted_length_of_cycle[0]}')
