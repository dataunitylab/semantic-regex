import argparse
import io
import json
import random

import matplotlib.pyplot as plt
import numpy as np
from pyarrow.parquet import ParquetFile
import shap
import tensorflow as tf

BACKGROUND_SIZE = 1000
SAMPLE_SIZE = 500


# Implements reservoir sampling
def update_sample(samples, N, sample):
    if sample is None:
        return

    if len(samples) < BACKGROUND_SIZE:
        samples.append(str(sample))
    else:
        s = int(random.random() * N)
        if s < BACKGROUND_SIZE:
            samples[s] = str(sample)


class_names = np.load("classes.npy", allow_pickle=True)
parser = argparse.ArgumentParser()
parser.add_argument("class_name", choices=class_names)
args = parser.parse_args()

# Get indexes of samples matching the given
# class in the first SAMPLE_SIZE values
pq_labels = ParquetFile("../sherlock-project/data/data/raw/train_labels.parquet")
class_idx = list(
    np.where(
        pq_labels.read(columns=["type"]).columns[0].to_numpy()[:SAMPLE_SIZE]
        == args.class_name
    )[0]
)

# See https://github.com/slundberg/shap/issues/1406
shap.explainers._deep.deep_tf.op_handlers[
    "AddV2"
] = shap.explainers._deep.deep_tf.passthrough

# Load the trained model
model = tf.keras.models.model_from_json(open("nn_model_sherlock.json").read())
model.load_weights("nn_model_weights_sherlock.h5")

# Produce a randomly sample of background from the training data
background = []
for (i, line) in enumerate(open("preprocessed_train.txt")):
    update_sample(background, i, line)

matrix = np.loadtxt(io.StringIO("".join(background)))
del background

# Load sample values matching the given class
sample = np.loadtxt(open("preprocessed_train.txt", "r"), max_rows=SAMPLE_SIZE)[
    class_idx, :
]

# Use SHAP to create a summary plot
e = shap.DeepExplainer(model, matrix)
shap_values = e.shap_values(sample)
feature_names = [l.strip() for l in open("pattern_ids.txt")]
shap.summary_plot(
    shap_values, sample, class_names=class_names, feature_names=feature_names
)
plt.savefig("shap.png")
