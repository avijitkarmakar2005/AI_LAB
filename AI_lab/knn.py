import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]
print(f"Best k: {best_k}, CV Accuracy: {max(cv_scores):.3f}\n")

plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_scores, marker='o')
plt.axvline(best_k, color='red', linestyle='--', label=f'Best k = {best_k}')
plt.xlabel('k')
plt.ylabel('Cross-validated Accuracy')
plt.title('Choosing the Best k for KNN')
plt.legend()
plt.grid(True)
plt.show()

knn_final = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
knn_final.fit(X_train, y_train)
y_pred = knn_final.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.2f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


new_sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = knn_final.predict(new_sample)
print(f"\nPrediction for {new_sample[0]}: {iris.target_names[prediction[0]]}")



def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

class SimpleKNN:
    def __init__(self, k=3, weighted=False):
        self.k = k
        self.weighted = weighted

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        distances = np.array([euclidean_distance(x, x_train) for x_train in self.X_train])
        k_indices = np.argsort(distances)[:self.k]

        if not self.weighted:
            k_labels = [self.y_train[i] for i in k_indices]
            return Counter(k_labels).most_common(1)[0][0]
        else:
            weighted_votes = {}
            for i in k_indices:
                label = self.y_train[i]
                weight = 1 / (distances[i] + 1e-8)
                weighted_votes[label] = weighted_votes.get(label, 0) + weight
            return max(weighted_votes, key=weighted_votes.get)

scratch_knn = SimpleKNN(k=best_k, weighted=True)
scratch_knn.fit(X_train, y_train)
scratch_pred = scratch_knn.predict(X_test)
scratch_acc = accuracy_score(y_test, scratch_pred)
print(f"\nFrom-scratch KNN Test Accuracy: {scratch_acc:.2f}")