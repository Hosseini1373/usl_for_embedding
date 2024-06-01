usl_for_embedding
==============================

It implmenets the USL Technique from this github repo (https://github.com/TonyLianLong/UnsupervisedSelectiveLabeling) for embeddings of any data, be it text, image, ... 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py    <- Scripts to transform the dataset. Rewrite this for different datasets to prepare it for USL.
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


# Getting Started

In this project we experiment with USL and USL-t on website embedding datasets like:
MuniWebEmbeds, CivicSiteVectors and Curlie Dataset. The two datasets MuniWebEmbeds and CivicSiteVectors can be retrieved by
contancting the the Institut für Informatik (InIT) in Zürcher hochschule für angewandte wissenschaften. Curlie Dataset is an 
open source dataset, which can be retrieved from https://curlie.org/docs/en/help/getdata.html.

The following shows a way to reproduce the results of the experiments.


General Pattern of usage:
`python src/main.py --method=<usl or usl-t> --mode=<train, val or test> --dataset=<zhaw or curlie>`

## MuniWebEmbeds Dataset
#### Apply USL and train an SSL model on zhaw database (Training set (train.pkl)):
you can find the relevant Configs in in config.json (usl)
`python src/main.py --method=usl --mode=train --dataset=zhaw`



#### Evaluate the SSL model form last step on zhaw database (Validation set (val.pkl)):
you can find the relevant Configs in in config.json (usl)
`python src/main.py --method=usl --mode=eval --dataset=zhaw`
The model that is used for evaluation is always the newest trained one.

#### Apply USL and train an SSL-t model on zhaw database (Training set (train.pkl)):
you can find the relevant Configs in in config.json (usl-t)
`python src/main.py --method=usl-t --mode=train --dataset=zhaw`


#### Evaluate the SSL-t model form last step on zhaw database (Validation set (val.pkl)):
you can find the relevant Configs in in config.json (usl-t)
`python src/main.py --method=usl --mode=eval --dataset=zhaw`
The model that is used for evaluation is always the newest trained one.



## Curlie Dataset
#### Apply USL and train an SSL model on curlie database (Training set (train.pkl)):
you can find the relevant Configs in in config.json (usl_curlie)
`python src/main.py --method=usl --mode=train --dataset=curlie`


#### Evaluate the SSL model form last step on curlie database (Validation set (val.pkl)):
you can find the relevant Configs in in config.json (usl_curlie)
`python src/main.py --method=usl --mode=eval --dataset=curlie`
The model that is used for evaluation is always the newest trained one.

#### Apply USL and train an SSL-t model on curlie database (Training set (train.pkl)):
you can find the relevant Configs in in config.json (usl-t_curlie)
`python src/main.py --method=usl-t --mode=train --dataset=curlie`


#### Evaluate the SSL-t model form last step on curlie database (Validation set (val.pkl)):
you can find the relevant Configs in in config.json (usl-t_curlie)
`python src/main.py --method=usl --mode=eval --dataset=curlie`
The model that is used for evaluation is always the newest trained one.




## CivicSiteVectors Dataset
#### Apply USL and train an SSL model on zhaw database (Training set (train.pkl)):
you can find the relevant Configs in in config.json (usl)
`python src/main.py --method=usl --mode=train --dataset=zhaw`



#### Evaluate the SSL model form last step on zhaw_segments database (Validation set (val.pkl)):
you can find the relevant Configs in in config.json (usl)
`python src/main.py --method=usl --mode=eval --dataset=zhaw_segments`
The model that is used for evaluation is always the newest trained one.

#### Apply USL and train an SSL-t model on zhaw_segments database (Training set (train.pkl)):
you can find the relevant Configs in in config.json (usl-t)
`python src/main.py --method=usl-t --mode=train --dataset=zhaw_segments`


#### Evaluate the SSL-t model form last step on zhaw_segments database (Validation set (val.pkl)):
you can find the relevant Configs in in config.json (usl-t)
`python src/main.py --method=usl --mode=eval --dataset=zhaw_segments`
The model that is used for evaluation is always the newest trained one.




# Makefile guide

This project can run in various modes: training, evaluation, and testing, each of which can be executed in either USL or USL-t method. Below are the commands available through the Makefile for running these operations.
Running Operations

MuniWebEmbeds dataset:

Train a model using USL method:

`make train_usl_one`

Evaluate a model using USL method:

`make eval_usl_one`

Test a model using USL method:

`make test_usl_one`

Train a model using USL-t method:

`make train_usl_t_one`

Evaluate a model using USL-t method:

`make eval_usl_t_one`

Test a model using USL-t method:

`make test_usl_t_one`


CivicSiteVectors dataset:

`make train_usl_two`

Evaluate a model using USL method:

`make eval_usl_two`

Test a model using USL method:

`make test_usl_two`

Train a model using USL-t method:

`make train_usl_t_two`

Evaluate a model using USL-t method:

`make eval_usl_t_two`

Test a model using USL-t method:

`make test_usl_t_two`


Curlie  dataset:

`make train_usl_curlie`

Evaluate a model using USL method:

`make eval_usl_curlie`

Test a model using USL method:

`make test_usl_curlie`

Train a model using USL-t method:

`make train_usl_t_curlie`

Evaluate a model using USL-t method:

`make eval_usl_t_curlie`

Test a model using USL-t method:

`make test_usl_t_curlie`



# Python Library
The folder containing the library is inside usl_embedding folder. This pip library was developed in parallel to this work for public use that provides a simple way to apply USL and USL-t to embeddings of any kind of data.


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
