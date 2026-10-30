import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
uploaded = files.upload()
df = pd.read_csv('student.csv')
X = df[['Hours']]
y = df['Marks']
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", round(rmse, 2))
print("R²:", round(r2, 2))
hours = float(input("Enter study hours: "))

prediction = model.predict(
    pd.DataFrame({"Hours": [hours]}))

print("Predicted marks:", round(prediction[0], 2))
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, model.predict(X_train), color='red')
plt.title('Scores vs Hours (Training Set)')
plt.xlabel('Hours')
plt.ylabel('Marks')