import pandas as pd

# Load the dataset from the root folder
df = pd.read_excel('data/wages_by_education_cleaned_v2.xlsx', sheet_name='Main', engine='openpyxl')

# Keep only the relevant columns
df_cleaned = df[['CareerField', 'CareerLevel', 'EducationLevel', 'MonthlyIncome']]

# Drop rows with missing values
df_cleaned.dropna(inplace=True)

# Save the cleaned data to the data folder
df_cleaned.to_csv('data/step2_cleaned_data.csv', index=False)

print("✅ Cleaned data saved to 'data/step2_cleaned_data.csv'")

#####
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the cleaned dataset
df = pd.read_csv('data/step2_cleaned_data.csv')

# Select categorical columns
categorical_columns = ['CareerField', 'CareerLevel', 'EducationLevel']

# Initialize the encoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Fit and transform the categorical columns
encoded_array = encoder.fit_transform(df[categorical_columns])

# Get the new column names
encoded_columns = encoder.get_feature_names_out(categorical_columns)

# Create a DataFrame from the encoded array
encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns)

# Combine with the target column
final_df = pd.concat([encoded_df, df[['MonthlyIncome']].reset_index(drop=True)], axis=1)

# Save the encoded data
final_df.to_csv('data/step3_encoded_data.csv', index=False)

print("✅ Encoded data saved to 'data/step3_encoded_data.csv'")
###
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the encoded dataset
df = pd.read_csv('data/step3_encoded_data.csv')

# Separate features and target
X = df.drop(columns=['MonthlyIncome'])
y = df['MonthlyIncome']

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print results
print("✅ Model Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")