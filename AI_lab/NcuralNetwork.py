import tensorflow as tf
import numpy as np

# Step 1: Input data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8],
             dtype=float)

# Step 2: Output data
# 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 0, 1, 1, 1, 1],
             dtype=float)

# Step 3: Create ANN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu',
                          input_shape=(1,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Step 4: Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 5: Train ANN
model.fit(X, y, epochs=100, verbose=1)

# Step 6: Test prediction
hours = np.array([[6.0]])

prediction = model.predict(hours)

print("Probability:", prediction[0][0])

if prediction[0][0] >= 0.5:
    print("Result: PASS")
else:
    print("Result: FAIL")