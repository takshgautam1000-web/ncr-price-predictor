## 🚀 Live Demo

**[Launch NCR Price Predictor](https://ncr-price-predictor-xpe96afnkpyvtasnh4xvry.streamlit.app/)**

Try the live machine learning app to estimate property prices in the NCR region.

# NCR Price Predictor

A machine-learning project for estimating residential property prices in Gurgaon (Gurugram), India, using property characteristics such as area, BHK count, locality, society, status, flat type, builder/company information, and RERA approval.

The project includes exploratory data analysis, model training and evaluation notebooks, a saved Random Forest regression model, and a Streamlit web interface.

## Project status

- Model training and held-out evaluation completed.
- Streamlit app created and run locally.
- GitHub version control setup started.
- Cloud deployment still requires final verification of data access and model loading.

## Model performance

The selected model was evaluated on a held-out test set using a stratified 80/20 train-test split (`random_state=42`). The reported metrics were:

| Metric | Held-out test result |
|---|---:|
| R² on log-price target | 0.87155 |
| R² on actual price | 0.85164 |
| Mean Absolute Error (MAE) | ₹8,896,835 (about ₹88.97 lakh) |
| Root Mean Squared Error (RMSE) | ₹18,938,490 (about ₹1.894 crore) |

**Interpretation:** the actual-price test R² is approximately 0.852. R² is not the same as a promise that every individual prediction will be accurate by 85%. MAE and RMSE show that prediction errors can still be substantial, especially for expensive properties.

The metrics above are from the held-out evaluation model, before the selected configuration was refitted on the full cleaned dataset for deployment.

## Model details

The final model is a `RandomForestRegressor` with the following configuration:

- `n_estimators=300`
- `max_features=0.25`
- `max_depth=None`
- `min_samples_split=2`
- `min_samples_leaf=1`
- `random_state=42`
- `n_jobs=-1`

The training data contained 8,308 cleaned rows, with zero duplicate rows reported in the cleaning check.

## Features and preprocessing

The training pipeline uses these model inputs:

- `Area` (log-transformed with `log1p`)
- `BHK_Count`
- `Is_Plot` (derived from plot/non-plot property category)
- `RERA Approval`
- `Status`
- `Flat Type`
- `Locality`
- `Company Name`
- `Socity` (the source dataset's column spelling)
- `Area_per_BHK` (original area divided by BHK count)

Categorical columns are one-hot encoded, and the saved feature-column list preserves the feature layout expected by the model. `Price` is the original target and `log_price` is the transformed target. Model predictions are converted back to the price scale using `numpy.expm1`.

`Rate per sqft` is excluded from the model inputs to avoid target leakage. `price_bin` is used only to stratify the train-test split and is not a predictor. `Property Type` is not a final model feature; the app uses a Plot/Non-Plot selection to construct `Is_Plot`.

## Repository structure

```text
ncr price predictor/
├── app.py
├── requirements.txt
├── readme.md
├── .gitignore
├── .gitattributes
├── data/
│   └── raw/                         # Download the dataset here locally
├── models/
│   ├── house_price_model.pkl        # Saved Random Forest model (Git LFS)
│   └── feature_columns.pkl          # Expected model feature columns
└── notebooks/
    ├── 01_data_exploration.ipynb
    └── 02_model_training.ipynb
```

The CSV dataset and the processed CSV are not included in this repository. Notebook copies of the model artifacts are also excluded; the canonical copies are in `models/`.

## Dataset and attribution

The project uses the **Gurgaon Real Estate Dataset** published by NikhilMehrahr26 on Kaggle:

**Dataset page:** https://www.kaggle.com/datasets/nikhilmehrahr26/gurgaon-real-estate-dataset

The dataset page lists its licence as **“Other (specified in description)”**, but the description available for this project does not provide specific redistribution terms. For that reason, the raw CSV is not committed to this repository. Anyone using the project should download the dataset from the original source and review the current licence/terms before use or redistribution.

### Local dataset setup

1. Download the dataset from the Kaggle page above, subject to its terms.
2. Place the CSV under `data/raw/`.
3. Ensure there is only one CSV under `data/`, because the current app automatically loads the first CSV it finds there.

**Deployment note:** the current `app.py` reads this CSV to populate the searchable categorical dropdowns. Because the dataset is not included in the repository, a hosted deployment will need an approved way to access the CSV or a code update that provides dropdown options without bundling the dataset. Do not deploy until this dependency has been addressed.

## Run locally

### 1. Requirements

- Python
- Git (optional for local execution)
- The dataset file, downloaded separately as described above

### 2. Set up the environment

Open PowerShell in the project root folder and run:

```powershell
python -m pip install -r requirements.txt
```

### 3. Start the Streamlit app

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open that URL in a browser.

## Saved model files

- `models/house_price_model.pkl` contains the fitted Random Forest model.
- `models/feature_columns.pkl` contains the feature names and order expected by the model.

The model is tracked with Git LFS because the model file is larger than GitHub's regular 100 MiB per-file limit. Git LFS storage and bandwidth limits may apply. The Python/scikit-learn environment used to load a serialized model should be compatible with the environment used to create it.

**Security note:** only load pickle/joblib files from sources that are trusted. Loading an untrusted serialized Python object can execute code.

## Reproducing the work

Use the notebooks in order:

1. `notebooks/01_data_exploration.ipynb` — data exploration and related analysis.
2. `notebooks/02_model_training.ipynb` — cleaning/model experiments, feature engineering, evaluation, and final model creation.

To reproduce the results, use the same cleaned data, target transformation, feature engineering, stratification, random seed, and estimator settings documented above. Small differences may occur if the data or dependency versions differ.

## Limitations

- The model learns patterns from the available historical listings; it is not a professional valuation or a guarantee of a sale price.
- Predictions may be less reliable for rare localities, unusual property types, unseen categories, or properties outside the training data's range.
- The test-set R² is an aggregate metric and does not describe the error of every listing.
- The current app's dropdowns depend on the separately downloaded CSV, so cloud deployment needs this dependency to be configured first.
- The dataset's exact redistribution terms were not specified in the description supplied for this project; users must check the current Kaggle page before reuse.

## License

No separate licence for this project's code has been specified in this repository. All rights in the underlying dataset remain subject to the dataset creator's applicable terms. Refer to the Kaggle source page before reusing or redistributing the dataset or derived materials.
