import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#using MinMax scalar to normalize the features
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
scaler = MinMaxScaler()
le = LabelEncoder()

train_data = pd.read_excel('train.xlsx')

X= train_data.drop('target',axis=1)
y = train_data['target']
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the train_data with random forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Function to predict the lables and explain it
def predict_classification(data_point):
    data_point_scaled = scaler.transform([data_point])
    prediction = rf_classifier.predict(data_point_scaled)[0]
    predicted_label = le.inverse_transform([prediction])[0]

    explanation = f"The data point is predicted to belong to class {predicted_label} because:\n"
    feature_importances = rf_classifier.feature_importances_

    for feature, importance in zip(X.columns, feature_importances):
        explanation += f"Feature {feature} importance: {importance:.4f}\n"

    return predicted_label, explanation