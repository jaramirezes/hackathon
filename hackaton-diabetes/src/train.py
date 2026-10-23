"""Entrenamiento reproducible del modelo de predicción de diabetes.

Replica, fuera del notebook, el pipeline completo: limpieza de ceros imposibles,
partición estratificada, búsqueda de hiperparámetros, calibración del umbral de
tamizaje y evaluación sobre el conjunto reservado.

Uso:
    python src/train.py

Genera:
    models/modelo_diabetes.joblib
    reports/resumen_metricas.json
    reports/metricas_busqueda.csv
    reports/metricas_prueba.csv
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)  # avisos de versión, no errores

SEED = 42
RECALL_OBJETIVO = 0.85
COLS_CERO_IMPOSIBLE = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "diabetes.csv"
URL_ESPEJO = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"


def cargar_datos() -> pd.DataFrame:
    """Lee el CSV local; si no existe, lo descarga del espejo público y lo guarda."""
    if RUTA_DATOS.exists():
        df = pd.read_csv(RUTA_DATOS)
    else:
        df = pd.read_csv(URL_ESPEJO)
        RUTA_DATOS.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RUTA_DATOS, index=False)
    # Un cero en estas variables es biológicamente imposible: es un dato faltante.
    df[COLS_CERO_IMPOSIBLE] = df[COLS_CERO_IMPOSIBLE].replace(0, np.nan)
    return df


def construir_pipeline(modelo, escalar: bool = True) -> Pipeline:
    """Imputación + escalamiento opcional + clasificador, en un solo objeto sin fuga de información."""
    pasos = [("imputador", SimpleImputer(strategy="median"))]
    if escalar:
        pasos.append(("escalador", StandardScaler()))
    pasos.append(("clasificador", modelo))
    return Pipeline(pasos)


def elegir_umbral(modelo, X_train, y_train, cv) -> float:
    """Umbral más alto que mantiene recall >= objetivo, calibrado FUERA de muestra."""
    proba = cross_val_predict(modelo, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    rejilla = np.arange(0.05, 0.96, 0.01)
    barrido = pd.DataFrame({
        "umbral": rejilla,
        "precision": [precision_score(y_train, (proba >= t).astype(int), zero_division=0) for t in rejilla],
        "recall": [recall_score(y_train, (proba >= t).astype(int)) for t in rejilla],
    })
    validos = barrido[barrido["recall"] >= RECALL_OBJETIVO]
    return float(validos.loc[validos["precision"].idxmax(), "umbral"])


def main() -> None:
    for carpeta in ("models", "reports"):
        (RAIZ / carpeta).mkdir(parents=True, exist_ok=True)

    df = cargar_datos()
    X, y = df.drop(columns="Outcome"), df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rejillas = {
        "Regresión Logística": (
            construir_pipeline(LogisticRegression(max_iter=5000, class_weight="balanced",
                                                  solver="liblinear", random_state=SEED)),
            {"clasificador__C": [0.01, 0.1, 0.5, 1, 3, 10], "clasificador__penalty": ["l1", "l2"]},
        ),
        "Random Forest": (
            construir_pipeline(RandomForestClassifier(class_weight="balanced", random_state=SEED,
                                                      n_jobs=-1), escalar=False),
            {"clasificador__n_estimators": [300, 600], "clasificador__max_depth": [4, 6, None],
             "clasificador__min_samples_leaf": [1, 3, 6]},
        ),
    }

    busquedas, filas_cv = {}, []
    for nombre, (pipe, rejilla) in rejillas.items():
        busqueda = GridSearchCV(pipe, rejilla, scoring="roc_auc", cv=cv, n_jobs=-1).fit(X_train, y_train)
        busquedas[nombre] = busqueda
        marcas = cross_validate(busqueda.best_estimator_, X_train, y_train, cv=cv,
                                scoring=["roc_auc", "f1", "recall", "precision"], n_jobs=-1)
        filas_cv.append({"Modelo": nombre, "ROC-AUC": busqueda.best_score_,
                         "F1": marcas["test_f1"].mean(), "Recall": marcas["test_recall"].mean(),
                         "Precisión": marcas["test_precision"].mean(),
                         "Mejores parámetros": str(busqueda.best_params_)})
        print(f"{nombre:22s} ROC-AUC (CV) = {busqueda.best_score_:.4f}  {busqueda.best_params_}")

    nombre_final = max(busquedas, key=lambda k: busquedas[k].best_score_)
    modelo_final = busquedas[nombre_final].best_estimator_
    umbral = elegir_umbral(modelo_final, X_train, y_train, cv)
    print(f"\nModelo final: {nombre_final}  ·  umbral de tamizaje = {umbral:.2f}")

    modelo_final.fit(X_train, y_train)
    proba_test = modelo_final.predict_proba(X_test)[:, 1]

    filas_test = []
    for u, etiqueta in [(0.50, "Umbral por defecto"), (umbral, "Umbral de tamizaje")]:
        pred = (proba_test >= u).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        filas_test.append({
            "Operación": etiqueta, "Umbral": u,
            "Exactitud": accuracy_score(y_test, pred),
            "Precisión": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred, zero_division=0),
            "Especificidad": tn / (tn + fp), "VPN": tn / (tn + fn),
            "ROC-AUC": roc_auc_score(y_test, proba_test),
            "PR-AUC": average_precision_score(y_test, proba_test),
            "VP": tp, "FP": fp, "FN": fn, "VN": tn,
        })

    tabla_cv, tabla_test = pd.DataFrame(filas_cv), pd.DataFrame(filas_test)
    print("\n" + tabla_test.round(3).to_string(index=False))

    tabla_cv.round(4).to_csv(RAIZ / "reports" / "metricas_busqueda.csv", index=False)
    tabla_test.round(4).to_csv(RAIZ / "reports" / "metricas_prueba.csv", index=False)
    joblib.dump({"pipeline": modelo_final, "umbral": umbral, "variables": list(X.columns),
                 "semilla": SEED, "modelo": nombre_final}, RAIZ / "models" / "modelo_diabetes.joblib")

    operacion = tabla_test.iloc[1]
    with open(RAIZ / "reports" / "resumen_metricas.json", "w", encoding="utf-8") as f:
        json.dump({
            "modelo_final": nombre_final,
            "hiperparametros": {k: str(v) for k, v in busquedas[nombre_final].best_params_.items()},
            "umbral_tamizaje": round(umbral, 2),
            "roc_auc_validacion_cruzada": round(float(busquedas[nombre_final].best_score_), 4),
            "prueba": {k: round(float(operacion[k]), 4) for k in
                       ["Exactitud", "Precisión", "Recall", "F1", "Especificidad", "VPN", "ROC-AUC", "PR-AUC"]},
            "n_entrenamiento": int(len(y_train)), "n_prueba": int(len(y_test)),
        }, f, ensure_ascii=False, indent=2)

    print(f"\nArtefactos escritos en {RAIZ / 'models'} y {RAIZ / 'reports'}")


if __name__ == "__main__":
    main()
