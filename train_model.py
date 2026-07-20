import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# Load Dataset
data = pd.read_csv("ford_car_dataset.csv")

print("Dataset Loaded Successfully!")
print(data.head())


# Separate Features and Target
X = data.drop("price", axis=1)
y = data["price"]


# Convert Categorical Columns into Numerical Columns
X = pd.get_dummies(X)


# Save Encoded Column Names
encoded_columns = X.columns.tolist()

joblib.dump(
    encoded_columns,
    "columns.pkl"
)


# Numerical Columns
numerical_cols = [
    "year",
    "mileage",
    "tax",
    "mpg",
    "engineSize"
]


# Feature Scaling
scaler = StandardScaler()

X[numerical_cols] = scaler.fit_transform(
    X[numerical_cols]
)


# Save Scaler
joblib.dump(
    scaler,
    "scaler.pkl"
)


# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train Linear Regression Model
model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# Save Trained Model
joblib.dump(
    model,
    "LR_ford_car.pkl"
)


# Model Accuracy
accuracy = model.score(
    X_test,
    y_test
)


print("Model Trained Successfully!")
print("Model Accuracy:", accuracy)

print("All .pkl files created successfully!")