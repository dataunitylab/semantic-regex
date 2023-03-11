import ast
import itertools
import math
import json
import numpy as np
import os
import pickle
import string
import sys

from pyarrow.parquet import ParquetFile
from tqdm import tqdm
import hyperscan
import joblib
import numpy as np
import scipy
import scipy.sparse


BATCH_SIZE = 1000
MAX_VALS = 1000
OUTPUT_FILENAME = "preprocessed.txt"

# Load the precompiled regular expression database
sys.stderr.write("Loading regexes from file…\n")
with open("hs.db", "rb") as f:
    [num_patterns, bdb] = pickle.load(f)
    db = hyperscan.loadb(bdb)

# Define a match callback for Hyperscan which updates the feature matrix
def on_match(match_id, from_idx, to_idx, flags, context):
    (str_id, count, matrix) = context
    matrix[(str_id, match_id)] = (matrix[(str_id, match_id)] * count + 1) / count


# Load the values
pq_values = ParquetFile("../webtables-extract/train_values.parquet")

# Remove the output if it exists
if os.path.exists(OUTPUT_FILENAME):
    os.remove(OUTPUT_FILENAME)

# Process batches in the input
with open(OUTPUT_FILENAME, "a") as f:
    batch = 0
    total_batches = math.ceil(pq_values.metadata.num_rows / BATCH_SIZE)
    for value_batch in tqdm(
        pq_values.iter_batches(BATCH_SIZE),
        total=total_batches,
        position=0,
        desc="Batches",
    ):
        matrix = np.zeros((value_batch.num_rows, num_patterns), dtype=np.float32)
        batch += 1
        i = 0
        for values in tqdm(
            value_batch.columns[0],
            total=value_batch.num_rows,
            position=1,
            desc="Matching",
            leave=False,
        ):
            i += 1

            # Extract a list of values from the batch
            value_array = ast.literal_eval(
                values.as_py().lstrip("".join(set(string.printable) - set(["["])))
            )

            # Encode up to MAX_VALS values for use in Hyperscan
            value_bytes = [str(v).encode("utf8") for v in value_array[:MAX_VALS]]

            # Scan each feature value for matches
            # Note: vector mode should be faster here, but seems to crash
            for v in value_bytes:
                db.scan(
                    v,
                    match_event_handler=on_match,
                    context=(i - 1, len(value_bytes), matrix),
                )

        # Save this batch to file
        np.savetxt(f, matrix, fmt="%.6g")
