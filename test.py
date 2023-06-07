import sys

import numpy as np
import pandas as pd
from pyarrow.parquet import ParquetFile
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import model_from_json
from tqdm import tqdm

BATCH_SIZE = 1000

sys.stderr.write("Loading labels...\n")
pq_labels = ParquetFile("../sherlock-project/data/data/raw/test_labels.parquet")
labels = pd.DataFrame(
    {"type": pd.Categorical(pq_labels.read(columns=["type"]).columns[0].to_numpy())}
)
le = LabelEncoder()
le.classes_ = np.load("classes.npy", allow_pickle=True)
# labels = le.transform(labels.values.ravel())
num_examples = len(labels)

model = model_from_json(open("nn_model_sherlock.json", "r").read())
model.load_weights("nn_model_weights_sherlock.h5")

sys.stderr.write("Evaluating...\n")
labels_pred = [""] * len(labels)
preprocessed = open("preprocessed_test.txt", "r")
batch = 0
with tqdm(total=len(labels)) as pbar:
    while True:
        try:
            matrix = np.loadtxt(preprocessed, max_rows=BATCH_SIZE)
        except StopIteration:
            break
        if len(matrix) == 0:
            break

        y_pred = model.predict(matrix)
        y_pred = np.argmax(y_pred, axis=1)
        labels_pred[
            batch * BATCH_SIZE : batch * BATCH_SIZE + len(matrix)
        ] = le.inverse_transform(y_pred)
        batch += 1
        pbar.update(len(matrix))

print(classification_report(labels, labels_pred))
