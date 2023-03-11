import ast
import hyperscan
import json
import pickle
import sys


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
    except hyperscan.error as e:
        pass

db = hyperscan.Database()
num_patterns = 0
patterns = []
ids = []
flags = []
for regex in regexes:
    patterns.append(regex.encode("utf8"))
    ids.append(num_patterns)
    flags.append(hyperscan.HS_FLAG_SINGLEMATCH | hyperscan.HS_FLAG_UTF8)
    num_patterns += 1

sys.stderr.write("Compiling %d patterns...\n" % len(patterns))
db.compile(expressions=patterns, ids=ids, flags=flags)
with open("hs.db", "wb") as f:
    pickle.dump([num_patterns, hyperscan.dumpb(db)], f)
