"""
Exercise description
--------------------

Description:
In the context of Mercadolibre's Marketplace an algorithm is needed to predict if an item listed in the markeplace is new or used.

Your tasks involve the data analysis, designing, processing and modeling of a machine learning solution 
to predict if an item is new or used and then evaluate the model over held-out test data.

To assist in that task a dataset is provided in `MLA_100k_checked_v3.jsonlines` and a function to read that dataset in `build_dataset`.

For the evaluation, you will use the accuracy metric in order to get a result of 0.86 as minimum. 
Additionally, you will have to choose an appropiate secondary metric and also elaborate an argument on why that metric was chosen.

The deliverables are:
--The file, including all the code needed to define and evaluate a model.
--A document with an explanation on the criteria applied to choose the features, 
  the proposed secondary metric and the performance achieved on that metrics. 
  Optionally, you can deliver an EDA analysis with other formart like .ipynb



"""

import json
import re
import numpy as np
import pandas as pd
import logging
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Configuracion logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# You can safely assume that `build_dataset` is correctly implemented
def build_dataset():
    logging.info("Building dataset...")
    data = [json.loads(x) for x in open("MLA_100k_checked_v3.jsonlines")]
    target = lambda x: x.get("condition")
    N = -10000
    X_train = data[:N]
    X_test = data[N:]
    y_train = [target(x) for x in X_train]
    y_test = [target(x) for x in X_test]
    for x in X_test:
        del x["condition"]
    logging.info("Dataset built successfully.")
    return X_train, y_train, X_test, y_test

def feature_engineering(df_train, df_test):
    """
    Performs feature engineering on the training and test datasets.

    This process includes:
    - Transforming payment methods into binary variables.
    - Extracting the seller's location.
    - Converting shipping information into categorical variables.
    - Processing the warranty field into a binary variable.
    - Extracting date components (day, month, year).
    - Creating a variable to indicate the presence of a video.
    - Removing irrelevant columns for the model.

    Parameters:
    ----------
    df_train : pd.DataFrame
        Training DataFrame before preprocessing.
    df_test : pd.DataFrame
        Test DataFrame before preprocessing.

    Returns:
    -------
    pd.DataFrame, pd.DataFrame
        Processed training and test DataFrames with engineered features.
    """
    logging.info("Starting feature engineering...")

    # Procesamiento de métodos de pago
    logging.info("Payment methods processing...")
    columna_metodos_pago = "non_mercado_pago_payment_methods"
    for df in [df_train, df_test]:
        if columna_metodos_pago in df.columns:
            metodos_unicos = set()
            for lista_metodos in df[columna_metodos_pago].dropna():
                if isinstance(lista_metodos, list):
                    metodos_unicos.update([metodo['id'] for metodo in lista_metodos])
            for metodo in metodos_unicos:
                df[f"pago_{metodo}"] = df[columna_metodos_pago].apply(
                    lambda x: 1 if isinstance(x, list) and any(metodo == metodo_dict['id'] for metodo_dict in x) else 0
                )
            df.drop(columns=[columna_metodos_pago], inplace=True)
    
    # Procesamiento de ubicación
    logging.info("Location processing...")
    for df in [df_train, df_test]:
        if 'seller_address' in df.columns:
            df["ubicacion_concatenada"] = df['seller_address'].apply(
                lambda x: f"{x.get('state', {}).get('id', '')}" # too much if adding  _{x.get('city', {}).get('id', '')}
                if isinstance(x, dict) else None
            )
            df.drop(columns=['seller_address'], inplace=True)
    
    # Procesamiento de shipping
    logging.info("Shipping processing...")
    for df in [df_train, df_test]:
        if 'shipping' in df.columns:
            df["retiro_en_persona"] = df['shipping'].apply(
                lambda x: x.get("local_pick_up", False) if isinstance(x, dict) else False
            )
            df["envio_gratis"] = df['shipping'].apply(
                lambda x: x.get("free_shipping", False) if isinstance(x, dict) else False
            )
            df.drop(columns=['shipping'], inplace=True)
    
    # Procesamiento de garantía
    logging.info("Warranty processing...")
    palabras_garantia = ["si", "sí", "mes", "meses", "dias", "días", "año", "años", "vida", "total", "con garantía", "de fabrica"]
    año_mal_escrito = r'\bano\b'
    años_mal_escrito = r'\banos\b'
    
    for df in [df_train, df_test]:
        if 'warranty' in df.columns:
            df["garantia"] = df['warranty'].apply(
                lambda x: None if pd.isna(x) else (
                    1 if any(palabra in str(x).lower() for palabra in palabras_garantia) or 
                         re.search(año_mal_escrito, str(x).lower()) or 
                         re.search(años_mal_escrito, str(x).lower()) else 0
                )
            )
    
    # Procesamiento de fecha de creación
    logging.info("Date created processing...")
    for df in [df_train, df_test]:
        if 'date_created' in df.columns:
            df['date_created'] = pd.to_datetime(df['date_created'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
            df["date_created_dia"] = df['date_created'].dt.day
            df["date_created_mes"] = df['date_created'].dt.month
            df["date_created_anio"] = df['date_created'].dt.year
    
    # Procesamiento de video
    logging.info("Video data processing...")
    for df in [df_train, df_test]:
        if 'video_id' in df.columns:
            df["tiene_video"] = df['video_id'].notna().astype(int)

    # Eliminar columnas excluidas del analisis
    logging.info("Dropping columns...")
    columnas_a_eliminar = ['sub_status', 'deal_ids', 'variations', 'attributes', 'tags', 'coverage_areas', 'descriptions', 'pictures',
                        'site_id', 'listing_source', 'parent_item_id', 'category_id', 'last_updated', 'international_delivery_mode',
                        'differential_pricing', 'thumbnail', 'title', 'secure_thumbnail', 'stop_time', 'subtitle', 'start_time',
                        'permalink', 'warranty', 'date_created', 'video_id', "id", "ubicacion_concatenada"]
    df_train.drop(columns=[col for col in columnas_a_eliminar if col in df_train.columns], inplace=True)
    df_test.drop(columns=[col for col in columnas_a_eliminar if col in df_test.columns], inplace=True)

    logging.info("Feature engineering completed.")
    return df_train, df_test

def data_encoding(df_train, df_test):
    """
    Applies data encoding to the training and test datasets.

    This process includes:
    - Applying One-Hot Encoding to categorical variables.
    - Converting boolean variables (True/False) to 1/0.
    - Ensuring consistency between training and test datasets:
    - Missing columns in the test set are added with NaN values.
    - The test dataset columns are reordered to match the training dataset.

    Parameters:
    ----------
    df_train : pd.DataFrame
        Training DataFrame before encoding.
    df_test : pd.DataFrame
        Test DataFrame before encoding.

    Returns:
    -------
    pd.DataFrame, pd.DataFrame
        Encoded training and test DataFrames with transformed features.
    """

    logging.info("Starting data encoding...")
    
    df_train = pd.get_dummies(df_train, drop_first=True)
    df_test = pd.get_dummies(df_test, drop_first=True)
    
    bool_cols_train = df_train.select_dtypes(include=['bool']).columns
    bool_cols_test = df_test.select_dtypes(include=['bool']).columns

    # Convertir solo las columnas booleanas a 1 y 0 de manera segura
    df_train[bool_cols_train] = df_train[bool_cols_train].astype(int)
    df_test[bool_cols_test] = df_test[bool_cols_test].astype(int)

    # Asegurar que ambas tienen las mismas columnas
    missing_cols = set(df_train.columns) - set(df_test.columns)
    for col in missing_cols:
        df_test[col] = np.nan
    
    df_test = df_test[df_train.columns]  # Reordenar columnas para coincidir con train
    
    logging.info("Data encoding completed.")
    return df_train, df_test


def objective(trial):
    """
    Objective function for hyperparameter optimization of LightGBM using Optuna.

    This process:
    - Defines a hyperparameter search space for LightGBM.
    - Trains a model with the parameters suggested by Optuna.
    - Evaluates the model on the validation set using the AUC metric.
    - Optuna optimizes the hyperparameters based on maximizing the AUC.

    Optimized hyperparameters:
    - learning_rate: Learning rate (loguniform between 0.005 and 0.2).
    - num_leaves: Number of leaves in each tree (int between 20 and 300).
    - max_depth: Maximum depth of trees (int between 3 and 12).
    - min_child_samples: Minimum number of samples per leaf (int between 10 and 100).
    - subsample: Row sampling rate (uniform between 0.5 and 1.0).
    - colsample_bytree: Fraction of features selected per tree (uniform between 0.5 and 1.0).
    - reg_alpha: L1 regularization (loguniform between 1e-8 and 10.0).
    - reg_lambda: L2 regularization (loguniform between 1e-8 and 10.0).
    - n_estimators: Number of trees (int between 100 and 1000).

    Parameters:
    ----------
    trial : optuna.Trial
        Instance of Optuna that suggests hyperparameters.

    Returns:
    -------
    float
        AUC score obtained on the test set with the evaluated hyperparameters.
    """

    # Definir los hiperparámetros que Optuna optimizará
    param_grid = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    }

    # Crear dataset de LightGBM
    dtrain = lgb.Dataset(df_train_FE_EN, label=y_train)
    dvalid = lgb.Dataset(df_test_FE_EN, label=y_test, reference=dtrain)

    model = lgb.train(
        param_grid,
        dtrain,
        valid_sets=[dvalid],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
    )

    # Predecir probabilidades y calcular AUC
    preds = model.predict(df_test_FE_EN)
    auc = roc_auc_score(y_test, preds)

    return auc


def feature_selection(X_train, y_train, n_features_to_select=10):
    """
    Performs feature selection using Recursive Feature Elimination (RFE).

    This process:
    - Uses a `RandomForestClassifier` as the base estimator.
    - Iteratively removes less relevant features until `n_features_to_select` is reached.
    - Reduces dataset dimensionality to prevent overfitting and improve model efficiency.

    Parameters:
    ----------
    X_train : pd.DataFrame
        DataFrame containing the training set features before selection.
    y_train : pd.Series
        Series with the labels corresponding to the training set.
    n_features_to_select : int, optional (default=10)
        Number of features to select in the final model.

    Returns:
    -------
    list
        List containing the names of the selected features.
    """

    logging.info("Starting feature selection with RFE...")
    
    # Definir el modelo base para RFE
    model = RandomForestClassifier(n_estimators=100, random_state=1994, n_jobs=-1)
    selector = RFE(model, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X_train, y_train)
    
    # Seleccionar las características
    selected_features = X_train.columns[selector.support_].tolist()
    
    logging.info(f"Selected {len(selected_features)} features using RFE.")
    return selected_features


if __name__ == "__main__":
    logging.info("Loading dataset...")
    
    # Train and test data following sklearn naming conventions
    X_train, y_train, X_test, y_test = build_dataset()
    
    # Convertir y_train e y_test a 1 / 0
    y_train = pd.Series(y_train).map({"new": 1, "used": 0})
    y_test = pd.Series(y_test).map({"new": 1, "used": 0})
    
    # Convertir X_train y X_test a DataFrames de Pandas
    df_train = pd.DataFrame(X_train)
    df_test = pd.DataFrame(X_test)
    
    # Aplicar Feature Engineering
    df_train_FE, df_test_FE = feature_engineering(df_train, df_test)
    
    # Aplicar Data Encoding
    df_train_FE_EN, df_test_FE_EN = data_encoding(df_train_FE, df_test_FE)

    logging.info("Preprocessing completed.")

    # Aplicar Feature Selection
    selected_features = [
        'listing_type_id_free', 'listing_type_id_gold', 'listing_type_id_gold_special', 'listing_type_id_silver',
        "catalog_product_id",
        "garantia",
        "automatic_relist",
        "retiro_en_persona",
        "envio_gratis",
        "pago_MLABC",
        "pago_MLAWT",
        "pago_MLATB",
        "pago_MLAOT",
        "pago_MLAMP",
        "price",
        "initial_quantity",
        "available_quantity",
        "sold_quantity",
        "official_store_id",
    ]
    logging.info("Selected features: {}".format(selected_features))
    df_train_FE_EN = df_train_FE_EN[selected_features]
    df_test_FE_EN = df_test_FE_EN[selected_features]

    # Optimización con Optuna
    logging.info("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    # Obtener los mejores hiperparámetros
    best_params = study.best_params
    logging.info(f"Best hyperparameters: {best_params}")

    # Dividir el dataset de entrenamiento en train y validation (80%-20%)
    X_train_final, X_valid, y_train_final, y_valid = train_test_split(
        df_train_FE_EN, y_train, test_size=0.2, random_state=8021994, stratify=y_train
    )

    # Crear datasets de LightGBM
    dtrain = lgb.Dataset(X_train_final, label=y_train_final)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    # Entrenar el modelo final con los mejores hiperparámetros encontrados
    final_model = lgb.train(
        best_params,
        dtrain,
        valid_sets=[dtrain, dvalid],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
    )

    # Evaluación final en el conjunto de test
    final_preds = final_model.predict(df_test_FE_EN)
    final_auc = roc_auc_score(y_test, final_preds)
    logging.info(f"Final AUC: {final_auc}") # Final AUC: 0.9249279972542158