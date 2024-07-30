import argparse
import os
import sys

import numpy as np
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

parser = argparse.ArgumentParser()
parser.add_argument("--sherlock-path", default="../sherlock-project/data/data/raw")
parser.add_argument("--input-dir", default=".")
parser.add_argument("--output-dir", default=".")
args = parser.parse_args()

sys.stderr.write("Loading labels...\n")
pq_labels = ParquetFile(os.path.join(args.sherlock_path, "train_labels.parquet"))
labels = pd.DataFrame(
    {
        "type": pd.Categorical(
            pq_labels.read(columns=["type"]).columns[0].to_numpy().ravel()
        )
    }
)
num_examples = len(labels)

# Encode the labels as integers
le = LabelEncoder().fit(labels.values.ravel())
labels = le.transform(labels.values.ravel())
np.save(os.path.join(args.output_dir, "classes.npy"), le.classes_)

# Load one row just to get the shape of the input
preprocessed = open(os.path.join(args.input_dir, "preprocessed_train.txt"), "r")
matrix = np.loadtxt(preprocessed, max_rows=1)
regex_shape = matrix.shape[0]

# Define the neural network architecture
regex_model_input = Input(shape=(regex_shape,))
regex_model1 = BatchNormalization(axis=1)(regex_model_input)
regex_model2 = Dense(
    1000,
    activation=tf.nn.relu,
    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
)(regex_model1)
regex_model3 = Dropout(0.35)(regex_model2)
regex_model4 = Dense(
    1000,
    activation=tf.nn.relu,
    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
)(regex_model3)

merged_model2 = BatchNormalization(axis=1)(regex_model4)
merged_model3 = Dense(
    500,
    activation=tf.nn.relu,
    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
)(merged_model2)
merged_model4 = Dropout(0.35)(merged_model3)
merged_model5 = Dense(
    500,
    activation=tf.nn.relu,
    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
)(merged_model4)
merged_model_output = Dense(
    len(le.classes_),
    activation=tf.nn.softmax,
    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
)(merged_model5)

# Compile the model and save the architecture
model = Model(regex_model_input, merged_model_output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)
open(os.path.join(args.output_dir, "nn_model_sherlock.json"), "w").write(
    model.to_json()
)

preprocessed = open(os.path.join(args.input_dir, "preprocessed_train.txt"), "r")
i = 0
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
        model.fit(
            matrix,
            batch_labels,
            epochs=10,
        )
        i += 1
        pbar.update(len(matrix))

# Save the trained model weights
model.save_weights(os.path.join(args.output_dir, "nn_model_weights_sherlock.h5"))
