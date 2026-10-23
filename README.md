# Mini-Hackatón · Modelo de Machine Learning para predicción de diabetes

> Reto: **generar un modelo de inteligencia artificial de Machine Learning**.
> Dataset: *Pima Indians Diabetes Database* — 768 registros, 8 variables clínicas, variable objetivo binaria.

**Autor:** Equipo JUAN ANTONIO RAMIREZ ESCAMILLA, MANUEL DE J. R. VELAZQUEZ TAPIA, EDUARDO ABURTO ANGULO, JUAN SINUE MONTENEGRO VALLEJO
**Programa:** MASTER en Inteligencia Artificial, Universidad Tecmilenio · Octubre 2026

---

## El resultado en una línea

Una **regresión logística regularizada** con el umbral calibrado en **0.35** detecta el **88.9 % de los casos positivos** (ROC-AUC 0.813 en prueba, IC 95 % 0.741–0.874) y, cuando descarta, acierta **9 de cada 10 veces** (VPN 0.914).

| Métrica (conjunto de prueba, n = 154) | Umbral 0.50 | **Umbral 0.35 (operación)** |
|---|---|---|
| Recall / sensibilidad | 0.704 | **0.889** |
| Precisión | 0.603 | 0.571 |
| F1 | 0.650 | **0.696** |
| Exactitud | 0.734 | 0.727 |
| Especificidad | 0.750 | 0.640 |
| Valor predictivo negativo | 0.824 | **0.914** |
| Falsos negativos | 16 | **6** |
| Falsos positivos | 25 | 36 |
| ROC-AUC | 0.813 | 0.813 |

El modelo se optimizó para **no dejar pasar casos positivos**, no para maximizar la exactitud: un falso positivo cuesta una prueba confirmatoria, un falso negativo cuesta una paciente sin seguimiento.

---

## Estructura del repositorio

```
.
├── notebooks/
│   └── Mini_Hackaton_Diabetes_Modelo_ML.ipynb   # Notebook completo, ya ejecutado (15 secciones)
├── src/
│   └── train.py                                 # Entrenamiento reproducible desde la terminal
├── data/
│   └── diabetes.csv                             # Dataset (768 × 9)
├── models/
│   └── modelo_diabetes.joblib                   # Pipeline + umbral, listo para producción
├── reports/
│   ├── resumen_metricas.json                    # Métricas finales en formato legible por máquina
│   ├── metricas_validacion_cruzada.csv          # Comparativa de los 6 algoritmos
│   ├── metricas_busqueda.csv                    # Resultado de GridSearchCV
│   ├── metricas_prueba.csv                      # Desempeño en el conjunto reservado
│   └── figures/                                 # 10 figuras en PNG
├── requirements.txt
└── README.md
```

---

## Cómo ejecutarlo

**En Google Colab** — el notebook descarga el dataset solo y corre de principio a fin sin configuración previa:

1. Subir `notebooks/Mini_Hackaton_Diabetes_Modelo_ML.ipynb` a Colab (o abrirlo desde GitHub con `Archivo → Abrir cuaderno → GitHub`).
2. `Entorno de ejecución → Ejecutar todas las celdas`.

Tiempo total de ejecución: ~70 segundos.

**En local:**

```bash
pip install -r requirements.txt
python src/train.py          # reentrena, reevalúa y regenera models/ y reports/
```

---

## Los datos

| | |
|---|---|
| Fuente original | National Institute of Diabetes and Digestive and Kidney Diseases |
| Versión usada | *Pima Indians Diabetes Database*, 768 × 9 (idéntica a la publicada en Kaggle como `mragpavank/diabetes`) |
| Población | Mujeres de ascendencia Pima, 21 años o más |
| Variable objetivo | `Outcome` — 1 = diagnóstico positivo (268 casos, 34.9 %) |

**Hallazgo de calidad de datos:** el `0` en glucosa, presión arterial, pliegue cutáneo, insulina e IMC es biológicamente imposible — codifica un dato faltante. Al recodificarlo aparece el faltante real: **insulina 48.7 %**, pliegue cutáneo 29.6 %, presión arterial 4.6 %, IMC 1.4 %, glucosa 0.7 %. Tratar esos ceros como mediciones reales es el error más común en los análisis de este dataset y sesga a la baja todas las medias.

![Datos faltantes](reports/figures/01_faltantes.png)

---

## Metodología

```
Ceros imposibles → NaN
        ↓
Partición estratificada 80/20  (el conjunto de prueba no se toca hasta el final)
        ↓
Pipeline:  imputación por mediana → escalamiento → clasificador
        ↓
Validación cruzada estratificada 5-fold sobre entrenamiento → se eligen 6 algoritmos
        ↓
GridSearchCV sobre los dos finalistas  (métrica: ROC-AUC)
        ↓
Calibración del umbral con predicciones fuera de muestra (cross_val_predict)
        ↓
Evaluación única sobre el conjunto de prueba + intervalos bootstrap
```

Tres decisiones que sostienen la validez del resultado:

- **La imputación vive dentro del `Pipeline`**, así la mediana se calcula solo con los datos de entrenamiento de cada partición. Imputar antes de partir habría inflado las métricas.
- **El umbral se calibra con `cross_val_predict` sobre el entrenamiento**, nunca sobre el conjunto de prueba. De otro modo el punto de corte se ajustaría a los datos con los que se reporta el resultado.
- **Semilla fija (42)** en partición, modelos y remuestreos: los números del README son exactamente los que produce el notebook al ejecutarse.

---

## Resultados

### Comparación de los seis algoritmos (validación cruzada 5-fold)

| Modelo | ROC-AUC | ± sd | PR-AUC | F1 | Recall | Precisión |
|---|---|---|---|---|---|---|
| **Regresión Logística** | **0.844** | 0.016 | 0.748 | 0.672 | 0.701 | 0.649 |
| SVM (RBF) | 0.837 | 0.031 | 0.709 | 0.690 | 0.762 | 0.637 |
| Random Forest | 0.832 | 0.024 | 0.721 | 0.687 | 0.729 | 0.651 |
| KNN (k=15) | 0.832 | 0.030 | 0.716 | 0.632 | 0.571 | 0.719 |
| Gradient Boosting | 0.819 | 0.016 | 0.713 | 0.631 | 0.598 | 0.672 |
| Árbol de Decisión | 0.786 | 0.027 | 0.634 | 0.637 | 0.748 | 0.561 |

Tras `GridSearchCV`, la regresión logística (L2, C = 0.5) sube a **0.845**. Los ensambles no la superan: con 768 registros y 8 variables, su capacidad adicional no encuentra señal extra que explotar.

![Ranking de modelos](reports/figures/05_ranking_modelos.png)

### El efecto del umbral

![Matrices de confusión](reports/figures/07_matrices_confusion.png)

![Curvas ROC y PR](reports/figures/08_curvas_roc_pr.png)

En el conjunto de prueba los tres mejores modelos quedan separados por centésimas de ROC-AUC, menos que el ancho del intervalo de confianza. La selección se sostiene en la validación cruzada y en la interpretabilidad, no en el decimal del conjunto de prueba.

### Intervalos de confianza (bootstrap, 1 000 remuestreos)

| Métrica | Estimación | IC 95 % |
|---|---|---|
| ROC-AUC | 0.812 | 0.741 – 0.874 |
| Recall | 0.890 | 0.800 – 0.968 |
| Precisión | 0.571 | 0.467 – 0.674 |
| F1 | 0.694 | 0.602 – 0.776 |

Con 154 registros de prueba, un número puntual sin intervalo exagera la precisión del resultado.

---

## Interpretabilidad

![Importancia por permutación](reports/figures/09_importancia_permutacion.png)

**La glucosa concentra el poder predictivo:** permutarla cuesta 0.157 de ROC-AUC, más de cinco veces lo que cuesta el IMC (0.029). Leído como razón de momios, cada desviación estándar adicional de glucosa multiplica por **3.19** la probabilidad relativa de diagnóstico positivo; el IMC la duplica (×2.00) y el número de embarazos la multiplica por 1.44. Presión arterial, pliegue cutáneo e insulina resultan prácticamente irrelevantes una vez consideradas las demás.

![Razones de momios](reports/figures/10_razones_momios.png)

---

## Limitaciones

1. **Población específica:** mujeres de ascendencia Pima de 21 años o más, un grupo con prevalencia excepcionalmente alta. Los coeficientes no son trasladables sin recalibrar.
2. **Muestra pequeña:** 768 registros, 154 en prueba. De ahí los intervalos de confianza.
3. **Faltante masivo en insulina** (48.7 %) y pliegue cutáneo (29.6 %): la imputación por mediana es defendible, pero aplana su variabilidad real.
4. **Corte transversal:** asociación, no causalidad.
5. **Sin validación externa ni temporal.**
6. **No es un dispositivo diagnóstico:** la salida prioriza pruebas confirmatorias, no sustituye criterio clínico.
7. **Riesgo de sesgo:** entrenado en población de alta prevalencia, tiende a sobrestimar el riesgo en poblaciones de baja prevalencia.

---

## Uso del modelo entrenado

```python
import joblib, pandas as pd

paquete = joblib.load("models/modelo_diabetes.joblib")
pipeline, umbral = paquete["pipeline"], paquete["umbral"]   # umbral = 0.35

paciente = pd.DataFrame([{
    "Pregnancies": 6, "Glucose": 165, "BloodPressure": 82, "SkinThickness": 35,
    "Insulin": 175, "BMI": 36.8, "DiabetesPedigreeFunction": 0.78, "Age": 52,
}])

probabilidad = pipeline.predict_proba(paciente)[0, 1]
decision = "Derivar a prueba confirmatoria" if probabilidad >= umbral else "Seguimiento de rutina"
print(f"{probabilidad:.3f} → {decision}")
```

El pipeline imputa internamente, así que acepta pacientes con datos faltantes (`NaN`) sin preprocesamiento adicional.

---

## Stack

`Python 3` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `joblib`

Sin dependencias fuera de lo que trae Google Colab preinstalado.

---

*La IA no sustituye el criterio del profesional; lo aumenta. El modelo ordena a quién mirar primero; la decisión sigue siendo de quien tiene la responsabilidad.*
