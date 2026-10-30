import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

X = np.array([
    [2, 2], [2, 3], [3, 2], [3, 3],
    [4, 3], [3, 4], [4, 2], [4, 1],
    [6, 6], [6, 7], [7, 6], [7, 7],
    [8, 6], [6, 8], [7, 8], [8, 3],
])

y = np.array([0, 0, 0, 0, 0, 0, 0, 0,
              1, 1, 1, 1, 1, 1, 1, 1])

# Step 3: Train the SVM model
model = SVC(kernel="linear", C=1.0)
model.fit(X, y)

# Step 4: Create a grid for visualization
x_min, x_max = 0, 10
y_min, y_max = 0, 10

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Calculate the decision function for every grid point
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.decision_function(grid)
Z = Z.reshape(xx.shape)
# Step 5: Draw the visualization
plt.figure(figsize=(10, 7))

# Show the two classes
plt.scatter(
    X[y == 0, 0], X[y == 0, 1],
    color="blue", s=100, label="Class A"
)

plt.scatter(
    X[y == 1, 0], X[y == 1, 1],
    color="red", s=100, label="Class B"
)

# Draw the margins and decision boundary
plt.contour(
    xx, yy, Z,
    levels=[-1, 0, 1],
    colors=["green", "black", "green"],
    linestyles=["--", "-", "--"]
)

# Highlight the support vectors
sv = model.support_vectors_

plt.scatter(
    sv[:, 0], sv[:, 1],
    s=250, facecolors="none",
    edgecolors="black", linewidths=2,
    label="Support Vectors"
)

# Step 6: Predict a new data point
new_point = np.array([[5,2]])
prediction = model.predict(new_point)[0]

plt.scatter(
    new_point[0, 0], new_point[0, 1],
    color="orange", marker="*", s=350,
    edgecolors="black", label="New Data Point"
)

plt.annotate(
    "New Point",
    (new_point[0, 0], new_point[0, 1]),
    xytext=(10, 10), textcoords="offset points"
)

# Step 7: Add graph details
plt.title("Support Vector Machine (SVM)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.grid(True, linestyle=":")
plt.show()

# Step 8: Display prediction
print("New data point:", new_point[0])
print("Predicted class:",
      "Class A" if prediction == 0 else "Class B")