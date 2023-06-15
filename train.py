import math
import sys

import numpy as np
from optuna_dashboard import run_server
import optuna
from optuna.storages import RDBStorage
import pandas as pd
from pyarrow.parquet import ParquetFile
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Model
from tqdm import tqdm


BATCH_SIZE = 1000
VALIDATION_SAMPLES = 10000

sys.stderr.write("Loading labels...\n")
pq_labels = ParquetFile("../sherlock-project/data/data/raw/train_labels.parquet")
labels = pd.DataFrame(
    {
        "type": pd.Categorical(
            pq_labels.read(columns=["type"]).columns[0].to_numpy().ravel()
        )
    }
)
pq_labels = ParquetFile("../sherlock-project/data/data/raw/validation_labels.parquet")
val_labels = pd.DataFrame(
    {
        "type": pd.Categorical(
            pq_labels.read(columns=["type"]).columns[0][:VALIDATION_SAMPLES].to_numpy().ravel()
        )
    }
)
del pq_labels
num_examples = len(labels)

# Encode the labels as integers
le = LabelEncoder().fit(labels.values.ravel())
labels = le.transform(labels.values.ravel())
val_labels = le.transform(val_labels.values.ravel())
np.save("classes.npy", le.classes_)

# Load one row just to get the shape of the input
preprocessed = open("preprocessed_train.txt", "r")
matrix = np.loadtxt(preprocessed, max_rows=1)
regex_shape = matrix.shape[0]


val_matrix = np.loadtxt("preprocessed_validation.txt", max_rows=VALIDATION_SAMPLES)

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    optim_learning_rate = trial.suggest_float("optim_learning_rate", 1e-5, 1e-1, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    num_layers = trial.suggest_int("num_layers", 100, 1000, 100)

    # Define the neural network architecture
    regex_model_input = Input(shape=(regex_shape,))
    regex_model1 = BatchNormalization(axis=1)(regex_model_input)
    regex_model2 = Dense(
        num_layers * 2,
        activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(learning_rate),
    )(regex_model1)
    regex_model3 = Dropout(dropout)(regex_model2)
    regex_model4 = Dense(
        num_layers * 2,
        activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(learning_rate),
    )(regex_model3)

    merged_model2 = BatchNormalization(axis=1)(regex_model4)
    merged_model3 = Dense(
        num_layers,
        activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(learning_rate),
    )(merged_model2)
    merged_model4 = Dropout(dropout)(merged_model3)
    merged_model5 = Dense(
        num_layers,
        activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(learning_rate),
    )(merged_model4)
    merged_model_output = Dense(
        len(le.classes_),
        activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(learning_rate),
    )(merged_model5)

    # Compile the model and save the architecture
    model = Model(regex_model_input, merged_model_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=optim_learning_rate),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    preprocessed = open("preprocessed_train.txt", "r")
    i = 0
    val_loss = math.inf
    with tqdm(total=len(labels)) as pbar:
        while True:
            # Load the next batch of data
            try:
                matrix = np.loadtxt(preprocessed, max_rows=BATCH_SIZE)
            except StopIteration:
                break
            if len(matrix) == 0:
                break

            # Pick out a batch of labels and fit the model on the batch
            batch_labels = tf.keras.utils.to_categorical(
                labels[i * BATCH_SIZE : i * BATCH_SIZE + len(matrix)]
            )
            train = model.fit(
                matrix,
                batch_labels,
                epochs=10,
                validation_data=(val_matrix, tf.keras.utils.to_categorical(val_labels)),
            )
            val_loss = train.history["val_loss"][-1]
            i += 1
            pbar.update(len(matrix))

    return val_loss


storage = RDBStorage("sqlite:///db.sqlite3")
study = optuna.create_study(storage=storage)
study.optimize(objective, n_trials=100)

run_server(storage)
