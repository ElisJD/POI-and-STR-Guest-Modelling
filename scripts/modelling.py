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

    ## 8. Plot Predicted vs Actual for Train and Test
    #y_train_pred = best_model.predict(X_train)
    #y_test_pred = best_model.predict(X_test)

    ## Convert back from log
    #y_train_actual = np.exp(y_train)
    #y_train_pred_exp = np.exp(y_train_pred)
    #y_test_actual = np.exp(y_test)
    #y_test_pred_exp = np.exp(y_test_pred)

    #plt.figure(figsize=(10, 6))
    #plt.scatter(y_train_actual, y_train_pred_exp, alpha=0.4, label='Train', color='blue')
    #plt.scatter(y_test_actual, y_test_pred_exp, alpha=0.4, label='Test', color='orange')

    ## Plot 1:1 ideal fit line
    #min_val = min(np.min(y_train_actual), np.min(y_test_actual))
    #max_val = max(np.max(y_train_actual), np.max(y_test_actual))
    #plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal Fit')

    #plt.xlabel('Actual Values')
    #plt.title('Actual vs Predicted (Train & Test)')
    #plt.ylabel('Predicted Values')
    #plt.legend()
    #plt.grid(False)
    #plt.tight_layout()
    #plt.show()

    ## 9. Residual Plot (Predicted vs. Residuals)
    #train_residuals = y_train_actual - y_train_pred_exp
    #test_residuals = y_test_actual - y_test_pred_exp

    #plt.figure(figsize=(10, 6))
    #plt.scatter(y_train_pred_exp, train_residuals, alpha=0.4, color='blue', label='Train')
    #plt.scatter(y_test_pred_exp, test_residuals, alpha=0.4, color='orange', label='Test')
    #plt.axhline(0, color='k', linestyle='--', lw=2)
    
    #plt.xlabel('Predicted Values')
    #plt.ylabel('Residuals (Actual - Predicted)')
    #plt.title('Residual Plot (Train & Test)')
    #plt.legend()
    #plt.grid(False)
    #plt.tight_layout()
    #plt.show()


    return best_model, sorted_importances