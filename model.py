import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    plt.figure(figsize=(6, 5))
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    plt.scatter(test_data,  test_labels,  c="g", label="Testing data")
    plt.scatter(test_data,  predictions,  c="r", label="Predictions")
    plt.legend()
    plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
    plt.title('Model Results')
    plt.xlabel('X axis values')
    plt.ylabel('Y axis values')
    plt.savefig('model_results.png', dpi=120)

# 1. Create data
X = np.arange(-100, 100, 4, dtype=np.float32)
y = np.arange(-90, 110, 4, dtype=np.float32)

# 2. Train / test split
X_train, X_test = X[:40], X[40:]
y_train, y_test = y[:40], y[40:]

# 3. Reshape to (batch, 1)
X_train = X_train.reshape(-1, 1)
X_test  = X_test .reshape(-1, 1)

# 4. Build model
tf.random.set_seed(42)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation="relu", input_shape=(1,)),
    tf.keras.layers.Dense(1)
])
model.compile(loss='mae', optimizer='sgd', metrics=['mae'])

# 5. Train
model.fit(X_train, y_train, epochs=100, verbose=0)

# 6. Predict & plot
y_preds = model.predict(X_test)
plot_predictions(X_train, y_train, X_test, y_test, y_preds.squeeze())

# 7. Metrics
#mae_val = tf.metrics.mean_absolute_error(y_test, y_preds.squeeze()).numpy().round(2)
#mse_val = tf.metrics.mean_squared_error(y_test, y_preds.squeeze()).numpy().round(2)

mae_val = np.round(np.mean(np.abs(y_test - y_preds.squeeze())), 2)
mse_val = np.round(np.mean(np.square(y_test - y_preds.squeeze())), 2)

# 8. Save for CML
with open('results.txt', 'w') as f:
    f.write(f'MAE = {mae_val}, MSE = {mse_val}')
print(f'MAE = {mae_val}, MSE = {mse_val}')
