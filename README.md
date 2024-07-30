# Learning from Uncurated Regular Expressions

[![CI](https://github.com/dataunitylab/semantic-regex/actions/workflows/ci.yml/badge.svg)](https://github.com/dataunitylab/semantic-regex/actions/workflows/ci.yml)

Dependencies of all Python code are managed with [Pipenv](https://pipenv.pypa.io/en/latest/) and can be installed with `pipenv install`.
Note that the dataset from the [Sherlock](https://github.com/mitmedialab/sherlock-project) project should be available in a copy of the repository in alongside the directory for this project.
[`jq`](https://jqlang.github.io/jq/) is also required for some JSON processing.

## Model training

1. Download all regular expressions from regex101

`./download_patterns.sh`

This will create a directory `regex101` which has the individual regular expressions and `patterns.json` which contains only the expressions strings.

2. Compile a database of all the downloaded regular expressions

`pipenv run python compile_db.py < patterns.json > patterns_final.json`

`patterns_final.json` is a subset of the expressions in `patterns.json` which are supported by Hyperscan.
This step will also create `hs.db` which are the compiled regular expressions that can be used during preprocessing.

3. Preprocess the data to generate feature vectors

`pipenv run python preprocess.py train`

This will generate `preprocessed_train.txt` which contains all the feature vectors extracted using the regular expression extracted using the regular expressions.

4. Train the model on the extracted features

`pipenv run python train.py`

The model architecture will be stored in `nn_model_sherlock.json` with the weights in `nn_model_weights_sherlock.h5`.

## Evaluation

First, the test data must be preprocessed.

`pipenv run python preprocess.py test`

Then, the model can be evaluated.

`pipenv run python test.py`

## Model explanation

Explains for predictions for an individual class can be generated using [SHAP](https://shap.readthedocs.io/en/latest/).
First, follow the steps for training the model above.
The file `patterns_final.json` will be used to match the patterns back to the original regular expressions.

`pipenv run python find_patterns.py > pattern_ids.txt`

This file of pattern IDs will then be used to label the SHAP plot with the ID of the regular expression.
To generate the SHAP plot in `shap.png`, run the command below where `<class_name>` is one of the semantic types defined by Sherlock.

`pipenv run python explain.py <class_name>`

The IDs displayed in the SHAP plot can be used to reference the regular expressions by ID in the `regex101/patterns` directory or viewing it directly on regex101 at the URL `https://regex101.com/library/<ID>`.
