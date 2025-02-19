├── LICENSE            # License file for the project

├── Makefile           # Makefile with commands like `make data` or `make train`

├── README.md          # The top-level README for developers using this project.

│

├── data               # Data storage directory
│   ├── external       # Data from third-party sources.
│   ├── interim        # Intermediate transformed data.
│   ├── processed      # Final, cleaned data sets for modeling.
│   └── raw            # Original, unprocessed data.
│
├── docs               # Documentation (Sphinx project)
│
├── models             # Trained models, predictions, or summaries.
│
├── notebooks          # Jupyter notebooks (naming convention: `1.0-author-description`)
│
├── references         # Data dictionaries, manuals, and other explanatory materials.
│
├── reports            # Generated reports (HTML, PDF, LaTeX, etc.)
│   └── figures        # Generated figures and visualizations.
│
├── requirements.txt   # Dependencies list (generated with `pip freeze > requirements.txt`)
│
├── setup.py           # Makes the project installable with `pip install -e .`
│
├── src                # Source code for this project
│   ├── __init__.py    # Makes src a Python module
│   │
│   ├── data           # Scripts for data handling
│   │   └── make_dataset.py
│   │
│   ├── features       # Scripts for feature engineering
│   │   └── build_features.py
│   │
│   ├── models         # Scripts for model training and prediction
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  # Scripts for data visualization
│       └── visualize.py
│
└── tox.ini            # Configuration for tox testing
