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
from matplotlib.patches import Patch
from geo_northarrow import add_north_arrow
from matplotlib_scalebar.scalebar import ScaleBar

from matplotlib.font_manager import FontProperties
# Load Cambria font
cambria_path = "/Library/Fonts/Microsoft/Cambria.ttf"  # Adjust path if needed
cambria_prop = fm.FontProperties(fname=cambria_path)


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
    sorted_importances = importances.sort_values(ascending=False).head(6)

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
    #plt.figure(figsize=(12, 10))
    #sns.barplot(x=sorted_importances.values, y=sorted_importances.index, palette="viridis")
    #plt.xlabel('Importance Score', fontsize=12)
    #plt.ylabel('POI Category (log)', fontsize=12)
    #plt.tight_layout()
    #plt.show()

    def clean_label(label):
        return (
            label.replace('_density_log', '')
                .replace('_', ' ')
                .title()
        )

    sorted_importances_clean = sorted_importances.copy()
    sorted_importances_clean.index = sorted_importances_clean.index.map(clean_label)

    # --- Step 2: Plot ---
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=sorted_importances_clean.values,
        y=sorted_importances_clean.index
    )

    # --- Step 3: Axis & font formatting ---
    plt.xlabel('Importance Score', fontsize=16, fontproperties=cambria_prop)
    plt.ylabel('POI Category', fontsize=16, fontproperties=cambria_prop)

    # Set tick font
    plt.xticks(fontproperties=cambria_prop, fontsize=12)
    plt.yticks(fontproperties=cambria_prop, fontsize=12)

    plt.tight_layout()
    plt.show()

    # 7. Print the top 10 most important features
    print("\n--- Top 6 Most Important Features ---")
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
            legend_kwds={'label': f"Coefficient", 'orientation': "horizontal"},
        )
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

def plot_combined_gwr_maps(df, gwr_feature_columns, base_geodata):
    """
    Plot side-by-side coefficient and significance maps for each GWR feature.
    """
    num_features = len(gwr_feature_columns)
    fig, axes = plt.subplots(nrows=num_features, ncols=2, figsize=(14, 4 * num_features))
    if num_features == 1:
        axes = np.array([axes])  # Ensure 2D array for consistency

    # Get global min/max for all coefficients to normalize color scale
    coeff_mins = [df[f'gwr_coeff_{f}'].min() for f in gwr_feature_columns]
    coeff_maxs = [df[f'gwr_coeff_{f}'].max() for f in gwr_feature_columns]
    vmin = min(coeff_mins)
    vmax = max(coeff_maxs)

    significance_cmap = ListedColormap(['#FC8D62', '#66C2A5'])  # Non-significant (red), Significant (green)
    significance_legend = [
        Patch(facecolor='#66C2A5', edgecolor='black', label='Significant'),
        Patch(facecolor='#FC8D62', edgecolor='black', label='Non-significant')
    ]

    panel_labels = [chr(65 + i) for i in range(2 * num_features)]

    for row_idx, feature in enumerate(gwr_feature_columns):
        pretty_name = feature.replace('_density_log', '').replace('_', ' ').title()

        # Coefficient plot
        ax_coeff = axes[row_idx, 0]
        base_geodata.plot(ax=ax_coeff, color='lightgrey')
        df.plot(
            column=f'gwr_coeff_{feature}',
            ax=ax_coeff,
            cmap='viridis',
            legend=False,
            vmin=vmin,
            vmax=vmax,
        )
        ax_coeff.text(
            0.5, -0.08, f"{pretty_name} – Coefficient",
            transform=ax_coeff.transAxes,
            ha='center', va='top',
            fontsize=14, fontproperties=cambria_prop
        )
        ax_coeff.text(-0.05, 1.05, panel_labels[row_idx * 2],
                    transform=ax_coeff.transAxes,
                    fontsize=14, va='top', ha='left',
                    fontproperties=cambria_prop)
        ax_coeff.set_axis_off()

        # Significance plot
        ax_sig = axes[row_idx, 1]
        base_geodata.plot(ax=ax_sig, color='lightgrey')
        df.plot(
            column=f'significance_{feature}',
            ax=ax_sig,
            cmap=significance_cmap,
            legend=False,
        )
        ax_sig.text(
            0.5, -0.08, f"{pretty_name} – Significance",
            transform=ax_sig.transAxes,
            ha='center', va='top',
            fontsize=14, fontproperties=cambria_prop
        )
        ax_sig.text(-0.05, 1.05, panel_labels[row_idx * 2 + 1],
                    transform=ax_sig.transAxes,
                    fontsize=14, va='top', ha='left',
                    fontproperties=cambria_prop)
        ax_sig.set_axis_off()
        ax_sig.text(-0.05, 1.05, panel_labels[row_idx * 2 + 1], transform=ax_sig.transAxes,
                    fontsize=14, va='top', ha='left', fontproperties=cambria_prop)

    # Add shared colorbar for coefficients
    cax = fig.add_axes([0.14, 0.07, 0.28, 0.01])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Coefficient Value', fontsize=14, fontproperties=cambria_prop)
    # Set tick labels font
    for label in cbar.ax.get_xticklabels():
        label.set_fontproperties(cambria_prop)
        label.set_fontsize(12)

    # Add legend for significance
    legend = fig.legend(
    handles=significance_legend,
    title='Significance',
    loc='lower center',
    bbox_to_anchor=(0.725, 0.06),
    ncol=2,
    fontsize=14
    )

    # Set font for legend title and labels
    legend.get_title().set_fontproperties(cambria_prop)
    legend.get_title().set_fontsize(14)
    for text in legend.get_texts():
        text.set_fontproperties(cambria_prop),
        text.set_fontsize(12)

    plt.tight_layout(rect=[0, 0.1, 1, 0.97])

def plot_combined_gwr_maps_side_legends(df, gwr_feature_columns, base_geodata):
    """
    Plot side-by-side coefficient and significance maps for each GWR feature,
    with vertical legends on the right-hand side.
    """
    num_features = len(gwr_feature_columns)
    fig, axes = plt.subplots(nrows=num_features, ncols=2, figsize=(18, 4 * num_features))
    if num_features == 1:
        axes = np.array([axes]) 

    # --- Shared value range for coefficient color scale ---
    coeff_mins = [df[f'gwr_coeff_{f}'].min() for f in gwr_feature_columns]
    coeff_maxs = [df[f'gwr_coeff_{f}'].max() for f in gwr_feature_columns]
    vmin, vmax = min(coeff_mins), max(coeff_maxs)

    significance_cmap = ListedColormap(['#FC8D62', '#66C2A5']) 
    significance_legend = [
        Patch(facecolor='#66C2A5', edgecolor='black', label='Significant'),
        Patch(facecolor='#FC8D62', edgecolor='black', label='Non-significant')
    ]

    panel_labels = [chr(65 + i) for i in range(2 * num_features)]

    for row_idx, feature in enumerate(gwr_feature_columns):
        pretty_name = feature.replace('_density_log', '').replace('_', ' ').title()

        # Coefficient map
        ax_coeff = axes[row_idx, 0]
        base_geodata.plot(ax=ax_coeff, color='lightgrey')
        df.plot(
            column=f'gwr_coeff_{feature}', ax=ax_coeff,
            cmap='viridis', vmin=vmin, vmax=vmax, legend=False
        )
        ax_coeff.text(
            0.5, -0.08, f"{pretty_name} – Coefficient",
            transform=ax_coeff.transAxes,
            ha='center', va='top',
            fontsize=18, fontproperties=cambria_prop
        )
        ax_coeff.text(-0.05, 1.05, panel_labels[row_idx * 2],
                      transform=ax_coeff.transAxes,
                      fontsize=18, va='top', ha='left',
                      fontproperties=cambria_prop)
        ax_coeff.set_axis_off()
        # For the first map, add a north arrow
        if row_idx == 0:
            add_north_arrow(ax_coeff, scale=1.2, xlim_pos=.1025, ylim_pos=.880, color='#000', text_scaler=0, text_yT=-1.25)

        # Significance map
        ax_sig = axes[row_idx, 1]
        base_geodata.plot(ax=ax_sig, color='lightgrey')
        df.plot(
            column=f'significance_{feature}', ax=ax_sig,
            cmap=significance_cmap, legend=False
        )
        ax_sig.text(
            0.5, -0.08, f"{pretty_name} – Significance",
            transform=ax_sig.transAxes,
            ha='center', va='top',
            fontsize=18, fontproperties=cambria_prop
        )
        ax_sig.text(-0.05, 1.05, panel_labels[row_idx * 2 + 1],
                    transform=ax_sig.transAxes,
                    fontsize=18, va='top', ha='left',
                    fontproperties=cambria_prop)
        ax_sig.set_axis_off()
        # For the last map, add a scale bar
        if row_idx == num_features - 1:
            scalebar = ScaleBar(1, location="lower right", units="m", color="black", font_properties={"size": 12})
            ax_sig.add_artist(scalebar)

    # --- Add vertical coefficient colorbar on the right ---
    cax = fig.add_axes([0.81, 0.42, 0.015, 0.25])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('Coefficient Value', fontsize=16, fontproperties=cambria_prop, labelpad=10)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(cambria_prop)
        label.set_fontsize(14)

    # --- Add vertical significance legend below the colorbar ---
    legend_ax = fig.add_axes([0.81, 0.28, 0.08, 0.1])  # [left, bottom, width, height]
    legend_ax.axis('off')
    legend = legend_ax.legend(
        handles=significance_legend,
        title='Significance',
        loc='upper left',
        frameon=False,
        fontsize=16
    )
    legend.get_title().set_fontproperties(cambria_prop)
    legend.get_title().set_fontsize(16)
    for text in legend.get_texts():
        text.set_fontproperties(cambria_prop)
        text.set_fontsize(16)
    plt.tight_layout(rect=[0, 0, 0.9, 1])

# Function 7: Geographically Weighted Random Forest
def geographically_weighted_random_forest(gdf, target, X_cols, coords, k_neighbours=100, min_local_data=10):
    """
    Runs Geographically Weighted Random Forest (GWRF) for a given target variable.

    Parameters:
    - gdf: GeoDataFrame with spatial units and input features.
    - target: str, name of the target variable.
    - X_cols: list of str, predictor variable column names.
    - coords: Nx2 array of centroid coordinates.
    - k_neighbours: int, number of neighbours for local fitting.
    - min_local_data: int, minimum valid observations required to train local RF.

    Returns:
    - gdf: GeoDataFrame with prediction, local R2, feature importances, and top POI columns added.
    """

    # Step 1: Build spatial index
    tree = cKDTree(coords)

    # Step 2: Prepare outputs
    predictions = []
    local_r2s = []
    feature_importance_list = []

    for i, point in enumerate(coords):
        distances, indices = tree.query(point, k=k_neighbours)
        raw_local_data = gdf.iloc[indices]

        # Step 3: Drop missing values
        local_data = raw_local_data.dropna(subset=[target] + X_cols)
        if len(local_data) < min_local_data:
            predictions.append(np.nan)
            local_r2s.append(np.nan)
            feature_importance_list.append([np.nan] * len(X_cols))
            continue

        # Step 4: Bisquare weights
        valid_idx_mask = raw_local_data.index.isin(local_data.index)
        valid_distances = distances[valid_idx_mask]
        D = valid_distances.max()
        weights = (1 - (valid_distances / D) ** 2) ** 2
        weights[valid_distances >= D] = 0
        prob = weights / weights.sum()

        # Step 5: Bootstrap sample
        X_local = local_data[X_cols].values
        y_local = local_data[target].values
        sample_idx = np.random.choice(len(X_local), size=len(X_local), p=prob, replace=True)
        X_weighted = X_local[sample_idx]
        y_weighted = y_local[sample_idx]

        # Step 6: Fit and evaluate
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_weighted, y_weighted)
        y_local_pred = rf.predict(X_local)
        r2_local = r2_score(y_local, y_local_pred)

        # Step 7: Predict for focal point
        X_pred = gdf.iloc[[i]][X_cols].values
        pred = rf.predict(X_pred)[0]

        # Step 8: Save outputs
        predictions.append(pred)
        local_r2s.append(r2_local)
        feature_importance_list.append(rf.feature_importances_)

        if i % 50 == 0:
            print(f"Processed {i}/{len(gdf)} polygons")

    # Step 9: Store back into GeoDataFrame
    gdf["grf_prediction"] = predictions
    gdf["grf_local_r2"] = local_r2s

    importances_df = pd.DataFrame(feature_importance_list,
                                  columns=[f"{col}_importance" for col in X_cols])
    gdf = gdf.join(importances_df)

    # Step 10: Identify most important POI
    importance_cols = [f"{col}_importance" for col in X_cols]
    gdf["top_poi"] = gdf[importance_cols].idxmax(axis=1)
    gdf["top_poi_clean"] = gdf["top_poi"].str.replace("_density_log_importance", "", regex=False)
    gdf["top_poi_clean"] = gdf["top_poi_clean"].str.replace("_log_importance", "", regex=False)

    return gdf

# Function 8: Evaluate Geographically Weighted Random Forest
def evaluate_grf_model(gdf, target_col, prediction_col="grf_prediction"):
    # Filter valid observations
    valid_mask = gdf[prediction_col].notna() & gdf[target_col].notna()
    y_true_log = gdf.loc[valid_mask, target_col]
    y_pred_log = gdf.loc[valid_mask, prediction_col]

    # Metrics on log scale
    rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    mae_log = mean_absolute_error(y_true_log, y_pred_log)
    r2 = r2_score(y_true_log, y_pred_log)

    # Metrics on original scale
    y_true_orig = np.exp(y_true_log)
    y_pred_orig = np.exp(y_pred_log)
    rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)

    print("GRF Evaluation Metrics")
    print("---------------------------------")
    print(f"R² (log scale):             {r2:.3f}")
    print(f"RMSE (log scale):           {rmse_log:.3f}")
    print(f"MAE  (log scale):           {mae_log:.3f}")
    print(f"RMSE (original scale):    {rmse_orig:,.2f}")
    print(f"MAE  (original scale):     {mae_orig:,.2f}")