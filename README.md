# 📉 Telco Customer Churn Prediction & Retention Optimization

![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.2%2B-150458?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-Red?logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-4B0082)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.11%2B-11557C)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13%2B-3776AB)
![Jupyter](https://img.shields.io/badge/Jupyter-IPykernel-F37626?logo=jupyter&logoColor=white)
![uv](https://img.shields.io/badge/Package_Manager-uv-DE5FE9?logo=astral&logoColor=white)

Solución de Machine Learning de extremo a extremo (*End-to-End*) para la predicción de fuga de clientes (*Customer Churn*) en el sector de las telecomunicaciones. 

Este proyecto combina técnicas avanzadas de preprocesamiento, manejo de datos desbalanceados, optimización del umbral de decisión enfocado en negocio e interpretabilidad de modelos mediante **SHAP values**.

---

## 📌 Resumen Ejecutivo y Caso de Negocio

La cancelación de clientes (*churn*) es uno de los principales problemas de rentabilidad en la industria Telco. Captar un cliente nuevo cuesta entre 5 y 7 veces más que retener a uno existente.

* **Objetivo:** Identificar proactivamente a los clientes con alta probabilidad de fuga antes de que ejecuten la baja, permitiendo al equipo comercial lanzar campañas de fidelización dirigidas.
* **Métrica clave de negocio:** Minimizar los **Falsos Negativos (FN)** (clientes que el modelo no detecta y terminan cancelando) maximizando el **Recall** para la clase positiva (*Churn*), sin destruir la precisión operativa del equipo de retención.

---

## 📊 Dataset y Auditoría de Datos

El dataset utilizado contiene **7,043 registros** y **21 variables** con información demográfica, servicios contratados y métricas financieras de cada cliente (*IBM Telco Customer Churn*).

* **Variable Objetivo (`Churn`):** Presenta un desbalanceo moderado (~73.5% *No Churn* vs. ~26.5% *Churn*).
* **Calidad de Datos:**
  * Tratamiento de inconsistencias en `TotalCharges` (cadenas de texto vacías `" "`). Se realizó coerción numérica descubriendo que pertenecían a clientes con antigüedad `tenure = 0` (recién dados de alta sin facturación emitida), imputando razonadamente $0$.
  * Eliminación de identificadores únicos (`customerID`) para prevenir memorización.

---

## 🛠️ Pipeline de Preprocesamiento y Feature Engineering

1. **Codificación de variables Categóricas:**
   * **Mapeo Binario:** Asignación $1/0$ a variables dicotómicas de entrada (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`); `Churn` se codifica como objetivo.
      * **Codificación Ordinal:** Mapeo jerárquico para `Contract` (`Month-to-month`: 1, `One year`: 2, `Two year`: 3) respetando la permanencia.
   * **One-Hot Encoding:** Aplicado con `drop_first=True` a variables nominales (`InternetService`, `PaymentMethod`, etc.) para evitar la trampa de las variables ficticias (*Dummy Variable Trap*).
2. **Estrategia de Validación:**
   * División **80/20** estratificada (`train_test_split` con `stratify=y`) para preservar exactamente la proporción original del 26.5% de *churn* en ambos conjuntos.
3. **Transformaciones integradas:** imputación numérica y One-Hot Encoding se ajustan únicamente con el conjunto de entrenamiento y se empaquetan junto con el modelo para reutilizarlos en inferencia. XGBoost no necesita escalado.

---

## 🚀 Modelo XGBoost seleccionado

El pipeline productivo se limita a `XGBClassifier`. Para compensar el desbalanceo,
calcula `scale_pos_weight` como la razón entre las clases negativas y positivas
del conjunto de entrenamiento (sin usar el conjunto de prueba). El umbral de
decisión por defecto es `0.35` y se puede ajustar en `config/model.yaml`.

En la ejecución de referencia con el CSV IBM (partición estratificada 80/20,
`random_state=42`), el test obtuvo recall **0.8262**, precisión **0.4703**,
PR-AUC **0.6401** y **65 falsos negativos** con el umbral 0.35. El pipeline
registra estas y otras métricas tanto de train como de test en MLflow; los
resultados nuevos se deben consultar allí, no en los experimentos históricos
del notebook.

---

## 🔍 Interpretabilidad del Modelo (SHAP Values)

Para abrir la "caja negra" de XGBoost y justificar las decisiones predictivas ante el área de negocio, se utilizaron valores **SHAP (SHapley Additive exPlanations)**.

Las cinco características con mayor media del valor absoluto SHAP en la ejecución
de referencia fueron **Contract, tenure, MonthlyCharges,
InternetService_Fiber optic** y **TotalCharges**. El ranking se recalcula al
entrenar; sus primeras cinco features se registran en Feast.

Conclusiones y plan de acción: 

* **Migración de Contratos**: Los clientes mes a mes (Month-to-month) son los más propensos a irse. Acción: Crear descuentos o beneficios para migrar a estos clientes a contratos de 1 o 2 años.
* **Fidelización en el "Año 1"**: El riesgo de fuga se concentra masivamente en clientes con baja antigüedad (tenure). Acción: Diseñar un programa de onboarding intensivo durante los primeros 6 meses.
* **Revisión del producto Fibra Óptica**: Si la fibra aparece como un predictor alto de fuga, indica insatisfacción con el servicio o un precio percibido muy alto frente a la competencia. Acción: Auditar la calidad técnica y competitividad de la oferta de fibra.
* **Importe en factura mensual**: Con respecto a los pagos mensuales, vemos que a facturas bajas hay una alta probabilidad que el cliente se mantenga, pero para facturas altas no hay comportamiento claro que determine una fuga o mantenerse, por lo que el precio en sí no causa la fuga sino la relación entre el precio y el valor percibido. Esta asociado con el resto de servicios contratados (fibra, streaming, multiples líneas) por lo que habría que analizarlo en detalle el coste por servicio.

---

## 🎯 Optimización de Umbrales y Análisis de Costes de Campaña

Este apartado resume experimentos del notebook exploratorio; la búsqueda
financiera de umbrales y sus funciones de coste no forman parte del pipeline
refactorizado. El entrenamiento actual usa el umbral configurable `0.35`.

En el notebook se evaluaron tres enfoques para la toma de decisiones:

```text
+-------------------------------------------------------------------------------+
|                    ESTRATEGIAS DE OPTIMIZACIÓN DE UMBRAL                      |
+------------------------------------+------------------------------------------+
|  Optimización Estadística (Youden) |    Optimización Financiera (Costes)      |
|  - Peso simétrico 1:1              |    - Peso asimétrico 33.3:1              |
|  - Umbral: 0.2351                  |    - Umbral: ~0.0400                     |
|  - Recall: 82.89%                  |    - Recall: 98.93%                      |
+------------------------------------+------------------------------------------+
```

### 1. Umbral Estadístico (Índice de Youden $J$)
Busca la máxima separación matemática entre aciertos y falsas alarmas ($J = \text{TPR} - \text{FPR}$):
* **Umbral Óptimo:** `0.2351`
* **Recall Alcanzado:** `82.89%` | **FPR Tolerado:** `31.01%`

### 2. Matriz de Coste Financiero en Euros
Incorpora la asimetría económica real negociada con el área financiera:
* **Coste por Falso Negativo (FN):** $500\ \text{€}$ (Valor del Cliente / LTV perdido).
* **Coste por Falso Positivo (FP):** $15\ \text{€}$ (Coste de la campaña/descuento de retención).
* **Coste por Verdadero Positivo (TP):** $15\ \text{€}$ (Coste de la campaña aplicada).

$$\text{Coste Total} = (FN \times 500) + (FP \times 15) + (TP \times 15)$$

Al ser la pérdida de un cliente **$33.3$ veces más costosa** que una falsa alarma (500 / 15), el umbral financiero óptimo se reduce a `~0.04`, alcanzando un **Recall del 98.93%** y minimizando el impacto económico global en la cuenta de resultados.

### 3. Estrategias Bajo Restricciones Operativas (OPEX y Capacidad)
Para entornos de negocio con recursos limitados, que se trata de una realidad, el proyecto incluye dos funciones parametrizadas:
* **Límite de Presupuesto (`presupuesto_max_campana`):** Minimiza la pérdida sujetándose a un techo máximo de gasto directo en promociones. Teniendo en cuenta los mismos costes que en el anterior método pero ahora fijando un presupuesto máximo de `11.500€`, se obtiene un umbral financiero de `0.16`, con un **Recall del 90.37%** 
Esté método arroja un volumen de `36` Falsos Negativos (Clientes Perdidos) que no llegamos a mantener y un volumen de `424` de Falsos Positivos (Campañas Desperdiciadas).

* **Capacidad Fija en Call Center (`Top-N Target`):** Ordena la cartera de mayor a menor probabilidad y selecciona a los $N$ clientes con mayor riesgo, determinando el **umbral implícito** resultante. Quedandonos con los `200` clientes con mayor probabilidad de abandono, tenemos un umbral implícito resultante de `0.6042`, con un **Recall conseguido de 38.77%**. Capturamos un total de `145` de `374` clientes totales en churn.

---

## 💼 Plan de Acción Comercial y Recomendaciones

**Segmentación por Valor y Riesgo:**
   * **Riesgo Crítico (Gasto Alto + Antigüedad baja):** Campañas proactivas de *"Optimización de Tarifa"* a los 3-6 meses a cambio de contratos de permanencia de 12 meses.
   * **VIPs de Alto Valor (Gasto Alto + Antigüedad alta):** Enrutamiento prioritario en soporte técnico, ofertas de equipamiento y fidelización sin cambios agresivos de tarifa.

---

## 📁 Estructura del Repositorio

```text
├── api/                         # API REST de predicción
├── config/                      # Configuración del modelo y entrenamiento
├── data/                        # Dataset y artefactos locales (ignorados por Git)
├── feature_store/               # Configuración local de Feast
├── models/                      # Artefactos de modelo locales (ignorados por Git)
├── notebooks/                   # Documentación y notebooks del proyecto
├── src/telco_churn/
│   ├── data_loader.py            # Carga y validación de los datos
│   ├── preprocessing.py         # Preprocesamiento reutilizable
│   ├── train.py                 # Entrenamiento XGBoost, SHAP y logging MLflow
│   ├── evaluate.py              # Métricas de entrenamiento y prueba
│   └── feature_store.py         # Registro y materialización en Feast
├── tests/
├── pyproject.toml               # Dependencias y configuración del paquete
├── requirements.txt             # Exportación reproducible del lock de uv
└── uv.lock
```

---

## ⚙️ Requisitos, Instalación y Configuración del Entorno con `uv`

El proyecto requiere Python 3.13 y [uv](https://docs.astral.sh/uv/). `uv sync`
crea/reutiliza `.venv` y sincroniza las versiones fijadas en `uv.lock`.

En Windows PowerShell:

```powershell
uv sync
Invoke-WebRequest `
  -Uri "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" `
  -OutFile "data/data_churn.csv"
```

El CSV y los artefactos generados se mantienen localmente y no se versionan.
`requirements.txt` es una exportación de `uv.lock`; para desarrollo se recomienda
usar `uv sync`.

### Orden de ejecución local

1. Ejecuta `uv sync` y descarga el CSV siguiendo los comandos anteriores.
2. Inicia MLflow en una terminal y déjalo activo durante el entrenamiento:

```powershell


```

3. En otra terminal, ejecuta el pipeline de entrenamiento:

```powershell
uv run python -m telco_churn.train
```

El entrenamiento usa una partición estratificada 80/20; calcula
`scale_pos_weight` como negativos/positivos únicamente sobre train. Se registran
en MLflow parámetros y métricas con prefijos `train_` y `test_` (incluyendo
recall, precisión, F1, ROC-AUC, PR-AUC y matriz de confusión). El umbral de
decisión y los parámetros están en `config/model.yaml`. El pipeline sklearn
completo (preprocesamiento + XGBoost) se registra como modelo versionado
`telco-churn-xgboost`; el alias local `champion` se actualiza a la versión
recién entrenada. Puede consultarse en la UI de MLflow, en **Models**, o en la
página del experimento/run.

El ranking de valores SHAP medios absolutos se guarda en
`models/feature_importance.csv`. Sus primeras cinco features se registran como
un Feature View en Feast y se materializan en el almacén online SQLite local.
Además, `explainability/psi_reference.json` queda adjunto al run de MLflow:
contiene solo estadísticas de `X_train` para esas features (estrategia de
binning, límites/categorías, proporciones de referencia, tamaño de train y
epsilon). Los campos numéricos con pocos valores distintos se tratan como
categóricos; los demás usan bins por cuantiles. El pipeline futuro de evaluación
debe reutilizar esos límites y categorías para calcular PSI sobre datos nuevos,
sin volver a ajustar bins sobre el conjunto comparado.
El fichero se crea en `data/feast/online_store.db` (junto con
`registry.db` y `shap_features.parquet`). Estos artefactos son locales y están
ignorados por Git; la configuración de Feast está en
`feature_store/feature_store.yaml`.

### API de predicción

4. Una vez que el entrenamiento termina y crea el modelo, inicia la API en otra
terminal:

```powershell
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
```

La API carga `models:/telco-churn-xgboost@champion` desde MLflow, no una copia
local del modelo. Ese artefacto es un `sklearn.Pipeline`: al recibir los campos
crudos, `predict_proba()` ejecuta primero el paso `preprocessor` y después
XGBoost. La etiqueta de respuesta usa el umbral configurable (`0.35` por
defecto), mientras que la probabilidad se devuelve sin discretizar.

La documentación interactiva estará disponible en `http://127.0.0.1:8000/docs`.
El endpoint `POST /predict` recibe las columnas de cliente excepto `Churn` y
devuelve la predicción binaria, probabilidad y umbral aplicado. `GET /health`
informa la URI de modelo MLflow cargada.

### Notebook exploratorio

El notebook original `01_telco_churn_pipeline.ipynb` se conserva en la raíz. La
implementación ejecutable del modelo seleccionado (XGBoost con
`scale_pos_weight`) está en `src/telco_churn/`; los experimentos con los otros
modelos del notebook no forman parte del pipeline de entrenamiento.
