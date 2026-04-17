import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv('credit_line.csv')

# Drop unnecessary columns
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Handle missing values (numeric only)
data.fillna(data.select_dtypes(include=np.number).mean(), inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Split features and target
X = data.drop('Exited', axis=1).values
y = data['Exited'].values

# Train-test split
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
from sklearn.preprocessing import StandardScaler # pyright: ignore[reportMissingModuleSource]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build ANN
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingModuleSource]
from tensorflow.keras.layers import Dense # pyright: ignore[reportMissingModuleSource]

model = Sequential()

model.add(Dense(units=16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

# Predictions
y_pred = model.predict(X_test)
y_pred = (y_pred.flatten() > 0.5).astype(int)

print("Predictions:\n", y_pred)


new_df = pd.DataFrame([{
    'CreditScore': 650,
    'Age': 35,
    'Tenure': 5,
    'Balance': 70000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 80000,
    'Geography': 'Spain',
    'Gender': 'Male'
}])

# Apply same encoding
new_df = pd.get_dummies(new_df)

# Align columns with training data
new_df = new_df.reindex(columns=data.drop('Exited', axis=1).columns, fill_value=0)

# Scale
new_data = scaler.transform(new_df)

# Predict
prediction = model.predict(new_data)
prediction = (prediction > 0.5).astype(int)

print(prediction)