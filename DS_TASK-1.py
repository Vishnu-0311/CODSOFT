import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = "D:\DS_DS-1.csv"
data = pd.read_csv(file_path)

# Handle missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)

# Convert categorical variables to numerical using LabelEncoder
label_encoders = {}
categorical_columns = ['Sex', 'Embarked']
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Feature selection
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)
# Filter passengers who survived
survived_passengers = data[data['Survived'] == 1]

# Print names of passengers who survived
print("Names of passengers who survived:")
print("\n")
for name in survived_passengers['Name']:
    print(name)
    
# Count the total number of passengers who survived
total_survived_passengers = data[data['Survived'] == 1]['Survived'].count()
print("\n")
print(f"Total number of passengers who survived: {total_survived_passengers}")
print("\n")
# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print(classification_report(y_test, predictions))
