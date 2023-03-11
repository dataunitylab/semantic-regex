#!/bin/bash

# Download each page of search results
mkdir -p pages/
wget "https://regex101.com/api/library/1/?orderBy=MOST_POINTS&search=" -O pages/1.json
PAGES=$(jq -r .pages pages/1.json)
for i in $(seq 2 $PAGES); do
    # Fetch this page of regular expressions
    wget "https://regex101.com/api/library/$i/?orderBy=MOST_POINTS&search=" -O "pages/$i.json"
    sleep 1
done

# Extract all fragments from each page to get individual regexes
mkdir -p regexes/
jq -cr '.data[] | (.permalinkFragment + " https://regex101.com/api/regex/" + .permalinkFragment + "/" + (.version | tostring))' pages/*.json | \
    while read -r frag url; do
        # If the regex has not already been fetched, fetch it
        [ -f "regexes/$frag.json" ] || (wget -O "regexes/$frag.json" -nc "$url"; sleep 1)
    done

# Extract all PCRE regexes without newlines into a file
jq -c 'select((.flavor == "pcre") and (.regex | contains( "\n") | not)) .regex' ../regex101/regexes/* > patterns.json
