# Learning from Uncurated Regular Expressions

Dependencies of all Python code are managed with [Pipenv](https://pipenv.pypa.io/en/latest/) and can be installed with `pipenv install`.
Note that the dataset from the [Sherlock](https://github.com/mitmedialab/sherlock-project) project should be available in a copy of the repository in the same directory as this project.
[`jq`](https://jqlang.github.io/jq/) is also required for some JSON processing.

1. Download all regular expressions from regex101

`./download_patterns.sh`

2. Compile a database of all the downloaded regular expressions

`pipenv run python compile_db.py < patterns.json`

3. Preprocess the data to generate feature vectors

`pipenv run python preprocess.py train`

4. Train the model on the extracted features

`pipenv run python train.py`
