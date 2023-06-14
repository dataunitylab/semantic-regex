import glob
import json

# Create a dictionary of possible patterns
pat_dict = {}
for file in glob.glob("regex101/regexes/*.json"):
    try:
        obj = json.load(open(file))
        pat_dict[obj["regex"]] = file.split("/")[-1].split(".")[0]
    except json.decoder.JSONDecodeError:
        pass

# Output the index and ID of each pattern
for line in open("patterns_final.json"):
    pat = json.loads(line)
    print(pat_dict[pat])
