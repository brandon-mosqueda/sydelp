import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import optimizers

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configuration
NUM_CLIENTS = 50
F = 20
ROUNDS = 200  # Number of communication rounds
LEARNING_RATE = 0.75
BETA = 0.99
BATCH_SIZE = 150

def euclidean_distance(m1, m2):
    return sum(np.sum((a - b)**2) for a, b in zip(m1, m2))

def krum(momentum_vectors, f):
    """
    Apply the KRUM aggregation on momentum vectors.

    Args:
        momentum_vectors (List[List[np.ndarray]]): List of momentum vectors (each is a list of numpy arrays).
        f (int): Maximum number of Byzantine clients to tolerate.

    Returns:
        List[np.ndarray]: Aggregated momentum vector chosen by KRUM.
    """
    n = len(momentum_vectors)
    scores = []

    # Compute KRUM scores
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                dist = euclidean_distance(
                    momentum_vectors[i], momentum_vectors[j])
                distances.append(dist)
        distances.sort()
        scores.append(sum(distances[:n - f - 2]))  # Exclude the f farthest

    # Select the update with minimum score
    kept_models_num: int = len(momentum_vectors) - f - 3
    selected_idx = np.argsort(scores)[:kept_models_num]

    result = []
    for i in range(len(momentum_vectors)):
        if i in selected_idx:
            result.append(momentum_vectors[i])

    return result

# Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# IID data partitioning
client_data = []
data_per_client = len(x_train) // NUM_CLIENTS
for i in range(NUM_CLIENTS):
    start = i * data_per_client
    end = (i + 1) * data_per_client
    client_data.append((x_train[start:end], y_train[start:end]))

# Model architecture

# Loss and optimizer (used only for gradient calculation)
loss_fn = keras.losses.CategoricalCrossentropy()
metric = keras.metrics.CategoricalAccuracy()

def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                      input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    return model

# Initialize global model
global_model = create_model()
global_weights = global_model.get_weights()
momentum_vectors = []
for _ in range(NUM_CLIENTS):
    momentum = [np.zeros(g.shape) for g in global_weights]
    momentum_vectors.append(momentum)

models = [create_model() for _ in range(NUM_CLIENTS)]

# Training loop
for round in range(ROUNDS):
    gradients_list = []

    for client_id in range(NUM_CLIENTS):
        model = models[client_id]
        model.set_weights(global_weights)

        x_c, y_c = client_data[client_id]
        idx = np.random.choice(np.arange(x_c.shape[0]), size=BATCH_SIZE, replace=False)
        x_c = x_c[idx]
        y_c = y_c[idx]

        with tf.GradientTape() as tape:
            preds = model(x_c, training=True)
            loss = loss_fn(y_c, preds)
        grads = tape.gradient(loss, model.trainable_weights)
        grads = [g.numpy() for g in grads]

        # Update momentum
        momentum_vectors[client_id] = [
            BETA * mom + (1 - BETA) * grad
            for mom, grad in zip(momentum_vectors[client_id], grads)
        ]

    selected_vectors = krum(momentum_vectors, F)

    avg_gradients = [np.zeros(g.shape) for g in global_weights]

    for mom in selected_vectors:
        avg_gradients = [
            m + avg
            for m, avg in zip(mom, avg_gradients)
        ]

    avg_gradients = [g / len(selected_vectors) for g in avg_gradients]

    # Server aggregation (element-wise average of momentum vectors)
    new_weights = [
        glob - LEARNING_RATE * avg
        for glob, avg in zip(global_weights, avg_gradients)
    ]
    global_weights = new_weights
    global_model.set_weights(global_weights)

    # Evaluate global model
    global_model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=[metric]
    )
    loss_val, acc_val = global_model.evaluate(x_test, y_test, verbose=0)
    print(f"Round {round + 1}: Loss = {loss_val:.4f}, Accuracy = {acc_val:.4f}")
