import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
# Assuming your dataset is in a CSV file named 'sales_data.csv'
file_path = "D:\DS_DS - 4.csv"
data = pd.read_csv(file_path)

# Feature selection
features = ['TV', 'Radio', 'Newspaper']
target = 'Sales'
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DataFrame with feature names
X_train_df = pd.DataFrame(X_train, columns=features)
X_test_df = pd.DataFrame(X_test, columns=features)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train_df, y_train)

# Make predictions
predictions = model.predict(X_test_df)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Predict sales for new data
new_data = pd.DataFrame([[100, 20, 30]], columns=features)  # Example values for TV, Radio, and Newspaper expenditures
predicted_sales = model.predict(new_data)
print(f"Predicted sales for new data: {predicted_sales[0]:.2f}")
