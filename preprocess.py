import argparse
import ast
import math
import numpy as np
import os
import pickle
import string
import sys

from pyarrow.parquet import ParquetFile
from tqdm import tqdm
import hyperscan


BATCH_SIZE = 1000
MAX_VALS = 1000

parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=["train", "test"])
parser.add_argument("--database", default="hs.db")
parser.add_argument("--sherlock-path", default="../sherlock-project/data/data/raw")
parser.add_argument("--output-dir", default=".")
args = parser.parse_args()

output_file = os.path.join(args.output_dir, f"preprocessed_{args.dataset}.txt")

# Load the precompiled regular expression database
sys.stderr.write("Loading regexes from file…\n")
with open(args.database, "rb") as f:
    [num_patterns, bdb] = pickle.load(f)
    db = hyperscan.loadb(bdb)
    # Scratch is not correctly initialized for deserialized databses
    # see https://github.com/darvid/python-hyperscan/issues/50
    db.scratch = hyperscan.Scratch(db)


# Define a match callback for Hyperscan which updates the feature matrix
def on_match(match_id, from_idx, to_idx, flags, context):
    (str_id, count, matrix) = context
    matrix[(str_id, match_id)] = (matrix[(str_id, match_id)] * count + 1) / count


# Load the values
pq_values = ParquetFile(
    os.path.join(args.sherlock_path, f"{args.dataset}_values.parquet")
)

# Remove the output if it exists
if os.path.exists(output_file):
    os.remove(output_file)

# Process batches in the input
with open(output_file, "a") as f:
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
