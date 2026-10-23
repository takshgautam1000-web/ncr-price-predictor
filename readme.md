# NCR Price Predictor

A machine-learning project that estimates residential property prices in Gurgaon (Gurugram), India, from property details such as area, BHK count, locality, society, status, flat type, company name, and RERA approval.

The project includes exploratory data analysis, feature engineering, model training and evaluation, a saved Random Forest regression model, and a Streamlit web application.

## 🚀 Live Demo

**[Launch NCR Price Predictor](https://ncr-price-predictor-xpe96afnkpyvtasnh4xvry.streamlit.app/)**

Try the live app to enter property details and receive an estimated price in Indian rupees.

**Source code:** [GitHub repository](https://github.com/takshgautam1000-web/ncr-price-predictor)

## Project Status

- Data exploration and model training notebooks included.
- Final Random Forest model and its feature-column list saved under `models/`.
- Streamlit app deployed on Streamlit Community Cloud.
- Raw dataset excluded from the repository; the deployed app builds dropdown options from the saved model feature names instead of loading the CSV.

## Model Performance

The selected configuration was evaluated using a stratified 80/20 train-test split with `random_state=42`. The following are the previously reported held-out test results:

| Metric | Held-out test result |
|---|---:|
| R² on log-price target | 0.87155 |
| R² on actual price | 0.85164 (about 85.16%) |
| Mean Absolute Error (MAE) | ₹8,896,835 (about ₹88.97 lakh) |
| Root Mean Squared Error (RMSE) | ₹18,938,490 (about ₹1.894 crore) |

**How to interpret these metrics:** the actual-price R² of approximately 0.852 describes performance across the held-out test set. It does not mean every individual estimate is 85% accurate. MAE and RMSE show that prediction errors can still be substantial, particularly for high-priced properties.

The reported metrics come from the held-out evaluation model. The deployment model was subsequently refitted on the full cleaned dataset for making predictions.

## Model Details

The final estimator is a scikit-learn `RandomForestRegressor` with these settings:

- `n_estimators=300`
- `max_features=0.25`
- `max_depth=None`
- `min_samples_split=2`
- `min_samples_leaf=1`
- `random_state=42`
- `n_jobs=-1`

The cleaned training data contained 8,308 rows. The cleaning checks reported no duplicate rows.

## Features and Preprocessing

The final model uses the following inputs:

- `Area` — log-transformed with `numpy.log1p`.
- `BHK_Count` — number of bedrooms, hall, and kitchen.
- `Is_Plot` — binary indicator derived from the app's Plot / Non-Plot selection.
- `RERA Approval`
- `Status`
- `Flat Type`
- `Locality`
- `Company Name`
- `Socity` — the source data's original column spelling.
- `Area_per_BHK` — original area divided by BHK count.

Categorical features were one-hot encoded during training. `models/feature_columns.pkl` stores the feature names and order expected by the model. At prediction time, the app recreates the input features and aligns them to this saved list. The predicted log-price is converted back to the price scale with `numpy.expm1`.

`Rate per sqft` was excluded from the final predictors to avoid target leakage. `price_bin` was used only to stratify the train-test split and is not a model feature. `Property Type` is not a final model feature; the app uses the Plot / Non-Plot selection to derive `Is_Plot`.

## Repository Structure

```text
ncr price predictor/
├── app.py
├── requirements.txt
├── readme.md
├── .gitignore
├── .gitattributes
├── models/
│   ├── house_price_model.pkl       # Trained Random Forest model (Git LFS)
│   └── feature_columns.pkl         # Expected feature names and order
└── notebooks/
    ├── 01_data_exploration.ipynb
    └── 02_model_training.ipynb
```

The raw dataset and processed CSV are intentionally not committed. Duplicate model artifacts previously stored in the notebooks folder are also excluded; the canonical model files are under `models/`.

## Dataset and Attribution

This project uses the **Gurgaon Real Estate Dataset** published by NikhilMehrahr26 on Kaggle:

**Dataset:** [Gurgaon Real Estate Dataset on Kaggle](https://www.kaggle.com/datasets/nikhilmehrahr26/gurgaon-real-estate-dataset)

The Kaggle page lists the licence as **“Other (specified in description)”**, but the description reviewed for this project did not specify clear redistribution terms. For that reason, the dataset is not included in this repository. Anyone seeking to reproduce the notebook analysis should obtain the data from the original Kaggle page and review the current licence and terms before use or redistribution.

The deployed app does **not** require the raw CSV to populate its dropdowns; it derives the available values from the saved model feature-column list. The notebooks do require the dataset to be downloaded locally to reproduce the original analysis and model training.

## Run the App Locally

### 1. Requirements

- Python
- The project files cloned or downloaded from GitHub

The Streamlit inference app uses the saved model and feature list; the raw CSV is not required to run the app itself.

### 2. Install dependencies

Open PowerShell in the project root and run:

```powershell
python -m pip install -r requirements.txt
```

### 3. Start Streamlit

```powershell
python -m streamlit run app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`. Open it in a browser.

## Reproducing the Analysis

Use the notebooks in this order:

1. `notebooks/01_data_exploration.ipynb` — dataset exploration and analysis.
2. `notebooks/02_model_training.ipynb` — cleaning, feature engineering, model experiments, evaluation, and final model creation.

To reproduce the reported metrics, use the corresponding dataset, target transformation, feature engineering, stratified split, random seed, and estimator configuration. Results may vary if the dataset or package versions change.

## Saved Model Files and Git LFS

- `models/house_price_model.pkl` contains the fitted Random Forest model.
- `models/feature_columns.pkl` contains the feature names and their expected order.

The trained model is larger than GitHub's regular 100 MiB per-file limit, so it is tracked with Git Large File Storage (Git LFS). Git LFS storage and bandwidth quotas may apply when cloning or deploying the repository.

The Python and scikit-learn versions used to load a serialized model should be compatible with those used to create it. **Security:** only load pickle/joblib files from trusted sources; loading an untrusted serialized object can execute code.

## Limitations

- The model reflects patterns in the dataset and is not a professional valuation or a guarantee of a property's sale price.
- Predictions may be less reliable for rare or unseen categories and properties outside the range represented in the training data.
- R² is an aggregate metric and does not describe the error for every individual listing.
- Some categories omitted during one-hot encoding may not be recoverable from the saved feature-column names alone; the dropdown values are based on the saved feature list and any explicitly configured options.
- The dataset's redistribution terms were not clear from the description reviewed. Consult the current Kaggle page before reusing or redistributing the dataset or derived materials.

## License

No separate licence for this project's code has been specified. Rights and reuse conditions for the underlying dataset remain subject to the dataset creator's applicable terms. Refer to the [original Kaggle dataset page](https://www.kaggle.com/datasets/nikhilmehrahr26/gurgaon-real-estate-dataset) before reusing or redistributing dataset-derived materials.
