# 📉 Telco Customer Churn Prediction & Retention Optimization

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0%2B-150458?logo=pandas&logoColor=white)
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
   * **Mapeo Binario:** Asignación $1/0$ a variables dicotómicas (`Partner`, `Dependents`, `PaperlessBilling`, `Churn`).
   * **Codificación Ordinal:** Mapeo jerárquico para `Contract` (`Month-to-month`: 0, `One year`: 1, `Two year`: 2) respetando la permanencia.
   * **One-Hot Encoding:** Aplicado con `drop_first=True` a variables nominales (`InternetService`, `PaymentMethod`, etc.) para evitar la trampa de las variables ficticias (*Dummy Variable Trap*).
2. **Estrategia de Validación:**
   * División **80/20** estratificada (`train_test_split` con `stratify=y`) para preservar exactamente la proporción original del 26.5% de *churn* en ambos conjuntos.
3. **Escalado de Características:**
   * Aplicación de `StandardScaler` sobre variables continuas (`tenure`, `MonthlyCharges`, `TotalCharges`) para garantizar la convergencia de modelos lineales.

---

## 🚀 Comparativa de Modelos y Resultados

Se evaluaron e iteraron diversos algoritmos partiendo de un modelo base (*Baseline*) lineal hasta ensembles basados en árboles con ajuste de pérdida y ajuste de umbral:

Dada la naturaleza desbalanceada del dataset (~26.5% positivos), la evaluación se prioriza sobre la **curva Precision-Recall (PR-AUC)** en lugar de ROC-AUC, evitando la distorsión producida por la abundancia de Verdaderos Negativos.

| Modelo | Accuracy | Precision (Churn) | **Recall (Churn)** | ROC-AUC | Falsos Negativos (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 0.81 | 0.66 | 0.56 | 0.8421 | 164 |
| **Random Forest Classifier** | 0.79 | 0.63 | 0.49 | 0.8248 | 190 |
| **XGBoost (Standard)** | 0.79 | 0.63 | 0.55 | 0.8366 | 170 |
| **XGBoost (`scale_pos_weight=2.75`)** | 0.76 | 0.53 | **0.75** | 0.8354 | **92** |
| **XGBoost + Threshold Tuning ($0.35$)** | 0.71 | 0.47 | **0.83** | 0.8390 | **65** |

> **Ganancia de Negocio:** Mediante la combinación del parámetro de ponderación de clase (`scale_pos_weight`) y la optimización del umbral de decisión ($0.35$), se logró reducir las fugas no detectadas de **164 a 65 clientes (reducción del 60.3% en Falsos Negativos)**, elevando el **Recall del 56% al 83%**.

---

## 🔍 Interpretabilidad del Modelo (SHAP Values)

Para abrir la "caja negra" de XGBoost y justificar las decisiones predictivas ante el área de negocio, se utilizaron valores **SHAP (SHapley Additive exPlanations)**.

Las características más influyentes a la hora de otorgar probabilidad a variable objetivo Churn (0 o 1) son las siguientes: **Contract, tenure, MonthlyCharges e InternetService_Fiber optic**.

Conclusiones y plan de acción: 

* **Migración de Contratos**: Los clientes mes a mes (Month-to-month) son los más propensos a irse. Acción: Crear descuentos o beneficios para migrar a estos clientes a contratos de 1 o 2 años.
* **Fidelización en el "Año 1"**: El riesgo de fuga se concentra masivamente en clientes con baja antigüedad (tenure). Acción: Diseñar un programa de onboarding intensivo durante los primeros 6 meses.
* **Revisión del producto Fibra Óptica**: Si la fibra aparece como un predictor alto de fuga, indica insatisfacción con el servicio o un precio percibido muy alto frente a la competencia. Acción: Auditar la calidad técnica y competitividad de la oferta de fibra.
* **Importe en factura mensual**: Con respecto a los pagos mensuales, vemos que a facturas bajas hay una alta probabilidad que el cliente se mantenga, pero para facturas altas no hay comportamiento claro que determine una fuga o mantenerse, por lo que el precio en sí no causa la fuga sino la relación entre el precio y el valor percibido. Esta asociado con el resto de servicios contratados (fibra, streaming, multiples líneas) por lo que habría que analizarlo en detalle el coste por servicio.

---

## 🎯 Optimización de Umbrales y Análisis de Costes de Campaña

El umbral por defecto ($0.50$) es ineficiente en datos desbalanceados. Se evalúan tres enfoques para la toma de decisiones:

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
├── data/
│   └── telco_churn.csv               # Dataset original de Telco Customer Churn
├── /
│   └── 01_telco_churn_pipeline.ipynb # Pipeline completo: EDA, ML, SHAP y Costes
├── README.md                         # Documentación ejecutiva y técnica del proyecto
└── requirements.txt                  # Dependencias (scikit-learn, xgboost, shap, pandas, matplotlib)
```

---

## ⚙️ Requisitos, Instalación y Configuración del Entorno con `uv`

Para replicar los experimentos y ejecutar las funciones de optimización:

Este proyecto utiliza [uv](https://docs.astral.sh/uv/) como gestor rápido de paquetes y entornos virtuales de Python.

### Prerrequisitos

Tener instalado `uv`. Si aún no lo tienes, instálalo con:

- **macOS / Linux:**
  ```bash
  curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
  ```
- **Windows**
  ```bash
  powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
  ```
- Tras ello, ejecutar lo siguiente: 
  ```bash
   git clone [https://github.com/sperezsa/telco-Churn.git](https://github.com/sperezsa/telco-Churn.git)
   cd telco-Churn
   uv sync
   ```
- Si usas VS Code:
1. Abre el archivo .ipynb en VS Code.
2. Arriba a la derecha, haz clic en Select Kernel $\rightarrow$ Python Environments.
3. Selecciona el intérprete de la carpeta `.venv` de tu proyecto.
4. Ejecuta las celdas directamente.
