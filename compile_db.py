import argparse
import hyperscan
import json
import pickle
import sys


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", default="hs.db")
args = parser.parse_args()

sys.stderr.write("Collecting patterns...\n")
regexes = set()
for line in sys.stdin:
    line = json.loads(line)
    db = hyperscan.Database()
    try:
        db.compile(
            expressions=(line.encode("utf8"),),
            ids=(0,),
            flags=(hyperscan.HS_FLAG_SINGLEMATCH | hyperscan.HS_FLAG_UTF8,),
        )
        regexes.add(line)
    except hyperscan.error:
        pass

# Build input for the final Hyperscan database
db = hyperscan.Database(mode=hyperscan.HS_MODE_BLOCK)
patterns = []
ids = []
flags = []
for i, regex in enumerate(regexes):
    print(json.dumps(regex))
    patterns.append(regex.encode("utf8"))
    ids.append(i)
    flags.append(hyperscan.HS_FLAG_SINGLEMATCH | hyperscan.HS_FLAG_UTF8)

# Compile the final database and save to file
sys.stderr.write("Compiling %d patterns...\n" % len(patterns))
db.compile(expressions=patterns, ids=ids, flags=flags)
with open(args.output, "wb") as f:
    pickle.dump([len(patterns), hyperscan.dumpb(db)], f)
