# MLStudio

MLStudio is a Streamlit application for training, validating, and testing
regression models without writing model-specific code.

## Run

```bash
uv run streamlit run mlstudio/main.py
```

## Workflows

### Training

Upload training data and, optionally, a separate test dataset. Select the
features, numeric target, preprocessing steps, regression model, and model
parameters.

Training rows can be selected in two ways:

- **Random percent:** reproducibly sample N percent of the uploaded rows.
- **Last percent:** use the final contiguous N percent of rows.

The fitted preprocessing and regression model are stored together in one
downloadable `.joblib` bundle. When test data is supplied, the app produces a
prediction preview and CSV download. R², MAE, RMSE, and MAPE are shown when the
test data contains the selected target.

### Validation

Upload one labeled dataset and choose one evaluation strategy:

- **Random split:** randomly split rows into training and validation sets.
- **Last split:** train on earlier rows and validate on the final rows.
- **Cross-validation:** run shuffled K-fold cross-validation.

Grid search is optional for all models.

### Test

Upload a trusted MLStudio model bundle and a test dataset. The app validates
the required feature columns, runs predictions, and provides a CSV download.
Metrics are optional and are calculated only when the saved target column is
present.

## Current Models

- Random Forest Regressor
- Gradient Boosting Regressor
- Ridge Regression

## Data Constraints

- Input files must be CSV or XLSX.
- Targets must be numeric.
- Features may be numeric, string/categorical, or boolean.
- Missing-value imputation is not implemented yet.
- MAPE excludes rows whose actual target is zero and is unavailable when all
  actual target values are zero.

Model bundles use Python serialization. Only upload bundles from trusted
sources and use compatible Python and dependency versions.
