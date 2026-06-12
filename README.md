# MLStudio

MLStudio is a Streamlit application for training, validating, and testing
regression models without writing model-specific code. It also includes a
PyTorch page for univariate time-series forecasting.

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

Training, validation, and test results include a processed-data preview showing
the encoded data after preprocessing and the final model input after feature
selection and target lookback, including the selected feature names. Training
shows this preview even when no test dataset is supplied.

### Validation

Upload one labeled dataset and choose one evaluation strategy:

- **Random split:** randomly split rows into training and validation sets.
- **Last split:** train on earlier rows and validate on the final rows.
- **Cross-validation:** run shuffled K-fold cross-validation.

Grid search is optional for all models.

Feature selection is available in Training and Validation as a `SelectKBest`
pipeline step with F Regression, MRMR Regression, and Relief Regression
scoring.

Target lookback appends consecutive target lags after feature selection and performs
recursive, ordered prediction from the target history saved during fitting.
Lookback models require last-row training selection. Cross-validation and grid
search use expanding-window `TimeSeriesSplit` folds; validation forecasts each
fold recursively, while grid-search scoring uses known earlier targets within
the fold to build in-sample lag features.

Targets can optionally use standard or min-max scaling. Target processing is
fitted with the model, applied to target lag features, and inverted before
predictions and metrics are returned.

### Test

Upload a trusted MLStudio model bundle and a test dataset. The app validates
the required feature columns, runs predictions, and provides a CSV download.
Metrics are optional and are calculated only when the saved target column is
present.

### Deep Learning Time Series

Select **Time Series** from the sidebar to train or test a PyTorch model from
one ordered numeric target series. The page creates sliding lookback windows
and supports MLP, 1D-CNN, RNN, GRU, LSTM, Bi-LSTM, and Conv1D-LSTM models.

Model controls include lookback, neurons, layers, learning rate, epochs, batch
size, hidden activation, output activation, target scaling, and the final
contiguous percentage of rows used for training. An optional second dataset
can be used for leakage-free recursive backtesting.

Deep-learning models are downloaded as versioned `.pt` bundles. A saved bundle
can be uploaded on the same page to forecast an arbitrary future horizon, with
an optional actual-target dataset for metrics. Forecasting always starts from
the training history stored in the bundle and feeds predictions back
recursively.

## Current Models

Supervised regression:

- Random Forest Regressor
- Gradient Boosting Regressor
- Voting Regressor

Deep-learning time series:

- MLP
- 1D-CNN
- RNN
- GRU
- LSTM
- Bi-LSTM
- Conv1D-LSTM

## Data Constraints

- Input files must be CSV or XLSX.
- Targets must be numeric.
- Features may be numeric, string/categorical, or boolean.
- Missing-value imputation is not implemented yet.
- MAPE excludes rows whose actual target is zero and is unavailable when all
  actual target values are zero.

Model bundles use Python serialization. Only upload bundles from trusted
sources and use compatible Python and dependency versions.
