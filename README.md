# MLStudio

MLStudio is a Streamlit application for training, validating, and testing
regression models without writing model-specific code.

## Run

```bash
uv run streamlit run mlstudio/main.py
```

## Architecture

The codebase has two explicit layers:

- `mlstudio/backend/` contains data loading, preprocessing, model definitions,
  evaluation, artifact serialization, and the train/validate/predict workflows.
  It has no Streamlit dependency.
- `mlstudio/frontend/` contains Streamlit pages and reusable rendering
  components. Pages gather user input, call one backend workflow, and render
  the returned result.

`mlstudio/main.py` is only the application entry point.

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

Prediction results also include a processed-data preview showing the encoded
data after preprocessing and the final model input after optional pipeline
plugins, including the selected feature names.

### Validation

Upload one labeled dataset and choose one evaluation strategy:

- **Random split:** randomly split rows into training and validation sets.
- **Last split:** train on earlier rows and validate on the final rows.
- **Cross-validation:** run shuffled K-fold cross-validation.

Grid search is optional for all models.

Feature selection is supplied by the optional
`mlstudio/plugins/feature_selection/` plugin. It adds the Training and
Validation controls plus a `SelectKBest` pipeline step with F Regression, MRMR
Regression, and Relief Regression scoring. Deleting that plugin directory
removes the whole feature without changing the core backend or frontend pages.

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
