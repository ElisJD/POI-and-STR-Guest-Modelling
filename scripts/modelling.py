import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from scipy.stats import gaussian_kde
import contextily as ctx
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import matplotlib.font_manager as fm
import xgboost as xgb
import shap
from scipy.spatial import cKDTree


# Function 1: Perparing Data
def prepare_data(df, feature_cols, target_col):
    """
    Prepare the data for modelling by selecting features and the target variable,
    and removing rows with NaN values in the target column.
    
    Args:
        df (pd.DataFrame): The main dataframe containing all data.
        feature_cols (list): A list of column names to be used as features (X).
        target_col (str): The name of the column to be used as the target variable (y).

    Returns:
        tuple: A tuple containing the clean features DataFrame (X) and the target Series (y).
    """
    # Create a mask to identify rows where the target column is not null
    mask = df[target_col].notnull()

    # Select the features and target using the mask to ensure alignment
    X = df.loc[mask, feature_cols]
    y = df.loc[mask, target_col]

    print(f"Data prepared for target: '{target_col}'")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Number of rows after cleaning: {len(y)}")

    return X, y

# Function 2: Training and Tuning a Model
def run_modelling_pipeline(X, y, original_y):
    """
    Runs a complete Random Forest modelling pipeline including baseline evaluation,
    hypterparameter tuning, and final evaluation.

    Args:
        X (pd.DataFrame): The feature set.
        y (pd.Series): The log-transformed target variable.
        original_y (pd.Series): The original (non-logged) target variable for MAE calculation.

    Returns:
        tuple: A tuple containing the final tuned model and a Series of its feature importances.
    """

    # 1. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

    # 2. Baseline Model
    print("\n--- Training Baseline Model ---")
    baseline_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    baseline_model.fit(X_train, y_train)
    y_pred_base = baseline_model.predict(X_test)
    r2_base = r2_score(y_test, y_pred_base)

    # Log-scale
    mae_base_log = mean_absolute_error(y_test, y_pred_base)
    rmse_base_log = np.sqrt(mean_squared_error(y_test, y_pred_base))

    # Original-scale
    mae_base_orig = mean_absolute_error(np.exp(y_test), np.exp(y_pred_base))
    rmse_base_orig = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred_base)))

    print(f"Baseline R-squared (R²): {r2_base:.2f}")
    print(f"Baseline MAE (log): {mae_base_log:.3f}")
    print(f"Baseline RMSE (log): {rmse_base_log:.3f}")
    print(f"Baseline MAE (original): {mae_base_orig:,.2f}")
    print(f"Baseline RMSE (original): {rmse_base_orig:,.2f}")

    # 3. Hyperparameter Tuning
    print("\n--- Starting Hyperparameter Tuning ---")
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 1.0]
    }
    rf = RandomForestRegressor(random_state=42)
    rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=50,
                                          cv=5, verbose=0, random_state=42, n_jobs=-1)
    rf_random_search.fit(X_train, y_train)
    print("Tuning complete.")
    print(f"Best parameters found: {rf_random_search.best_params_}")

    # 4. Evaluate Tuned Model
    best_model = rf_random_search.best_estimator_
    y_pred_tuned = best_model.predict(X_test)
    r2_tuned = r2_score(y_test, y_pred_tuned)
    # Log-scale
    mae_tuned_log = mean_absolute_error(y_test, y_pred_tuned)
    rmse_tuned_log = np.sqrt(mean_squared_error(y_test, y_pred_tuned))

    # Original-scale
    mae_tuned_orig = mean_absolute_error(np.exp(y_test), np.exp(y_pred_tuned))
    rmse_tuned_orig = np.sqrt(mean_squared_error(np.exp(y_test), np.exp(y_pred_tuned)))

    print(f"Tuned R-squared (R²): {r2_tuned:.2f}")
    print(f"Tuned MAE (log): {mae_tuned_log:.3f}")
    print(f"Tuned RMSE (log): {rmse_tuned_log:.3f}")
    print(f"Tuned MAE (original): {mae_tuned_orig:,.2f}")
    print(f"Tuned RMSE (original): {rmse_tuned_orig:,.2f}")

    # 5. Get Feature Importances
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    sorted_importances = importances.sort_values(ascending=False).head(20)

    # 6. Plot Feature Importances
    #plt.figure(figsize=(10, 6))
    #sorted_importances.head(20)[::-1].plot(kind='barh', color='blue')
    #plt.title("Top 20 Feature Importances")
    #plt.xlabel("Importance Score")
    #plt.ylabel("Features")
    #plt.grid(axis='x', linestyle='--', alpha=0.7)
    #plt.tight_layout()
    #plt.show()

    # 6. Plot Feature Importances
    plt.figure(figsize=(12, 10))
    sns.barplot(x=sorted_importances.values, y=sorted_importances.index, palette="viridis")
    plt.title('Top 20 Most Important POI Features for Explaining Airbnb Revenue', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('POI Category', fontsize=12)
    plt.tight_layout()
    plt.show()

    # 7. Print the top 10 most important features
    print("\n--- Top 10 Most Important Features ---")
    print(sorted_importances.head(10))

    # 8. Plot Predicted vs Actual for Train and Test
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Convert back from log
    y_train_actual = np.exp(y_train)
    y_train_pred_exp = np.exp(y_train_pred)
    y_test_actual = np.exp(y_test)
    y_test_pred_exp = np.exp(y_test_pred)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_train_actual, y_train_pred_exp, alpha=0.4, label='Train', color='blue')
    plt.scatter(y_test_actual, y_test_pred_exp, alpha=0.4, label='Test', color='orange')

    # Plot 1:1 ideal fit line
    min_val = min(np.min(y_train_actual), np.min(y_test_actual))
    max_val = max(np.max(y_train_actual), np.max(y_test_actual))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal Fit')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted (Train & Test)')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # 9. Residual Plot (Predicted vs. Residuals)
    train_residuals = y_train_actual - y_train_pred_exp
    test_residuals = y_test_actual - y_test_pred_exp

    plt.figure(figsize=(10, 6))
    plt.scatter(y_train_pred_exp, train_residuals, alpha=0.4, color='blue', label='Train')
    plt.scatter(y_test_pred_exp, test_residuals, alpha=0.4, color='orange', label='Test')
    plt.axhline(0, color='k', linestyle='--', lw=2)

    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot (Train & Test)')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.show()


    return best_model, sorted_importances

# Function 3: Plotting the Results
def map_gwr_coefficients(df, gwr_results, gwr_feature_columns, base_geodata, ncols=3):
    """
    Plot GWR coefficient maps for each feature.
    """
    num_features = len(gwr_feature_columns)
    nrows = int(np.ceil(num_features / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 6 * nrows))
    axes = axes.flatten()

    for feature_to_map_index, ax in zip(range(1, num_features + 1), axes):
        feature_name = gwr_feature_columns[feature_to_map_index - 1]
        df[f'gwr_coeff_{feature_name}'] = gwr_results.params[:, feature_to_map_index]

        # Plot base layer
        base_geodata.plot(ax=ax, color='lightgrey')

        # Plot GWR coefficients
        df.plot(
            column=f'gwr_coeff_{feature_name}',
            ax=ax,
            legend=True,
            legend_kwds={'label': f"GWR Coefficient for {feature_name}", 'orientation': "horizontal"},
        )
        ax.set_title(f'{feature_name}', fontsize=14)
        ax.set_axis_off()

    # Hide unused axes
    for ax in axes[num_features:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

# Function 4: Calculate t-values
def calculate_t_values(df, gwr_results, gwr_feature_columns):
    """
    Calculate t-values for each feature and add to the dataframe.
    """
    for feature_to_map_index in range(1, len(gwr_feature_columns) + 1):
        feature_name = gwr_feature_columns[feature_to_map_index - 1]

        coeffs = gwr_results.params[:, feature_to_map_index]
        std_errors = gwr_results.bse[:, feature_to_map_index]

        df[f't_value_{feature_name}'] = coeffs / std_errors

# Function 5: Categorise Significance
def categorise_significance(df, gwr_feature_columns, threshold=2):
    """
    Categorize each observation as Significant or Non-significant based on t-values.
    """
    for feature_name in gwr_feature_columns:
        t_col = f't_value_{feature_name}'
        sig_col = f'significance_{feature_name}'

        df[sig_col] = np.where(
            (df[t_col] > threshold) | (df[t_col] < -threshold),
            'Significant',
            'Non-significant'
        )

# Function 6: Plot Significance Maps
def plot_significance(df, gwr_feature_columns, base_geodata, ncols=3):
    """
    Plot significance maps (Significant vs Non-significant) for each feature.
    """
    significant_colour = '#66C2A5'  # Soft green
    non_significant_colour = '#FC8D62'  # Soft red

    num_features = len(gwr_feature_columns)
    nrows = int(np.ceil(num_features / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 6 * nrows))
    axes = axes.flatten()

    for feature_name, ax in zip(gwr_feature_columns, axes):
        sig_col = f'significance_{feature_name}'

        base_geodata.plot(ax=ax, color='lightgrey')

        df.plot(
            column=sig_col,
            ax=ax,
            cmap=ListedColormap([non_significant_colour, significant_colour]),
            legend=True,
            legend_kwds={'title': 'Significance'}
        )

        ax.set_title(f'{feature_name}', fontsize=14)
        ax.set_axis_off()

    for ax in axes[num_features:]:
        ax.set_visible(False)

    plt.suptitle("GWR: Significant vs Non-Significant Coefficients", fontsize=24)
    plt.tight_layout()
    plt.show()

# Function 7: Geographically Weighted Random Forest
def geographically_weighted_rf(gdf, target, predictors, k_neighbors=100, epsg=27700, return_feature_importance=True):
    # Reproject to metric CRS
    gdf = gdf.to_crs(epsg=epsg).copy()
    
    # Coordinates for spatial neighbors
    coords = np.array([[geom.centroid.x, geom.centroid.y] for geom in gdf.geometry])
    tree = cKDTree(coords)

    predictions = []
    importances = []

    for i, point in enumerate(coords):
        distances, indices = tree.query(point, k=k_neighbors)

        local_data = gdf.iloc[indices]
        local_data = local_data.dropna(subset=[target] + predictors)

        if len(local_data) < 10:
            predictions.append(np.nan)
            if return_feature_importance:
                importances.append([np.nan] * len(predictors))
            continue

        X_local = local_data[predictors].values
        y_local = local_data[target].values

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_local, y_local)

        X_pred = gdf.iloc[[i]][predictors].values
        pred = rf.predict(X_pred)[0]
        predictions.append(pred)

        if return_feature_importance:
            importances.append(rf.feature_importances_)

        if i % 50 == 0:
            print(f"Processed {i}/{len(gdf)} points")

    # Add predictions to GeoDataFrame
    gdf["grf_prediction"] = predictions

    # Add feature importances
    if return_feature_importance:
        importance_cols = [f"{col}_importance" for col in predictors]
        importances = np.array(importances)
        for i, col in enumerate(importance_cols):
            gdf[col] = importances[:, i]

    return gdf, target