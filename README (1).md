# Predicción del ausentismo laboral mediante Scikit-Learn Pipeline

## Descripción del proyecto

Este proyecto desarrolla un flujo completo de Machine Learning para predecir la cantidad de horas de ausentismo laboral. El trabajo utiliza `Pipeline` y `ColumnTransformer` de Scikit-Learn para integrar el preprocesamiento de variables numéricas y categóricas con un modelo de Regresión Lineal.

El propósito principal es construir una solución reproducible, evitar la fuga de datos o *data leakage* y asegurar que las mismas transformaciones se apliquen durante el entrenamiento, la evaluación y la inferencia sobre nuevos registros.

## Objetivo

Construir un pipeline de Machine Learning capaz de:

- Procesar variables numéricas y categóricas de manera diferenciada.
- Imputar posibles valores faltantes.
- Estandarizar las variables numéricas.
- Codificar las variables categóricas mediante One-Hot Encoding.
- Entrenar un modelo básico de regresión.
- Evaluar el modelo con datos no utilizados durante el entrenamiento.
- Realizar predicciones sobre nuevos registros.
- Guardar y reutilizar el pipeline entrenado mediante `joblib`.

## Dataset utilizado

El proyecto utiliza el dataset de ausentismo laboral trabajado durante las etapas previas del curso. El archivo empleado contiene inicialmente 700 registros y 12 columnas.

La variable objetivo es:

- `Absenteeism Time in Hours`: cantidad de horas de ausentismo laboral registradas.

Entre las variables predictoras se encuentran:

- `Reason for Absence`
- `Transportation Expense`
- `Distance to Work`
- `Age`
- `Daily Work Load Average`
- `Body Mass Index`
- `Education`
- `Children`
- `Pets`
- `Date`

La columna `ID` se conserva en el dataset original, pero se excluye del entrenamiento porque funciona como identificador y no representa una característica predictiva.

## Preparación de los datos

Durante la exploración inicial se verificaron los tipos de datos, los valores faltantes, los valores únicos y los registros duplicados.

Principales decisiones adoptadas:

- Se eliminaron 8 registros completamente duplicados.
- No se detectaron valores nulos en el dataset analizado.
- `Date` se convirtió a formato `datetime`.
- A partir de `Date` se generaron las variables `Month` y `DayOfWeek`.
- La fecha original se eliminó después de generar las nuevas características.
- `ID` fue excluida de las variables predictoras.

Después de eliminar duplicados, el análisis continuó con 692 observaciones.

## Tipo de dato y tipo de variable

La clasificación de las variables no se realizó únicamente a partir del tipo técnico detectado por Pandas. También se consideró el significado conceptual de cada característica.

Por ejemplo:

- `Reason for Absence` está almacenada como número entero, pero sus valores representan categorías de motivos de ausencia.
- `Education` también está codificada numéricamente, pero representa niveles educativos.
- `Month` y `DayOfWeek` se tratan como variables categóricas temporales.
- `Age`, `Distance to Work` y `Transportation Expense` representan magnitudes cuantitativas.

### Variables numéricas

- `Transportation Expense`
- `Distance to Work`
- `Age`
- `Daily Work Load Average`
- `Body Mass Index`
- `Children`
- `Pets`

### Variables categóricas

- `Reason for Absence`
- `Education`
- `Month`
- `DayOfWeek`

## Prevención de data leakage

La fuga de datos ocurre cuando información que debería permanecer fuera del entrenamiento influye directa o indirectamente en el ajuste del modelo.

Para evitarla, la división entre entrenamiento y prueba se realiza antes de ajustar cualquier transformación:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

El pipeline se entrena únicamente con los datos de entrenamiento:

```python
model_pipeline.fit(X_train, y_train)
```

Por lo tanto, las medianas, los parámetros de escalado, las categorías frecuentes y las columnas generadas por One-Hot Encoding se aprenden exclusivamente a partir de `X_train`.

El conjunto de prueba se utiliza solamente para evaluar la capacidad de generalización:

```python
y_pred = model_pipeline.predict(X_test)
```

## Pipeline de preprocesamiento

### Pipeline numérico

Las variables numéricas atraviesan dos transformaciones:

1. `SimpleImputer(strategy="median")`: completa posibles valores faltantes mediante la mediana aprendida en el conjunto de entrenamiento.
2. `StandardScaler()`: estandariza las variables numéricas.

```python
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

### Pipeline categórico

Las variables categóricas también atraviesan dos transformaciones:

1. `SimpleImputer(strategy="most_frequent")`: completa posibles valores faltantes con la categoría más frecuente.
2. `OneHotEncoder(handle_unknown="ignore")`: transforma las categorías en variables binarias.

```python
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
```

El parámetro `handle_unknown="ignore"` permite procesar categorías nuevas durante la inferencia sin producir errores de formato.

## ColumnTransformer

Los pipelines numérico y categórico se integran mediante `ColumnTransformer`:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ]
)
```

De esta forma, cada grupo de variables recibe automáticamente el tratamiento correspondiente.

## Pipeline final y modelo

El preprocesador se integra con un modelo de `LinearRegression` dentro de un único pipeline:

```python
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])
```

El flujo completo ejecuta de manera secuencial:

1. Imputación de variables numéricas.
2. Estandarización de variables numéricas.
3. Imputación de variables categóricas.
4. Codificación One-Hot.
5. Entrenamiento de la Regresión Lineal.

## Resultados obtenidos

El modelo obtuvo las siguientes métricas sobre el conjunto de prueba:

| Métrica | Resultado |
|---|---:|
| MAE | 5.8581 horas |
| RMSE | 15.0523 horas |
| R² | 0.0638 |

### Interpretación

El MAE indica que las predicciones presentan, en promedio, una diferencia absoluta aproximada de 5.86 horas respecto de los valores reales.

El RMSE fue considerablemente mayor que el MAE, lo que indica la existencia de observaciones con errores elevados. El análisis mostró dificultades especialmente en registros con cantidades extremas de horas de ausentismo. Por ejemplo, se observó un caso con 120 horas reales para el cual el modelo predijo aproximadamente 10.04 horas.

El R² de 0.0638 indica que la Regresión Lineal explica aproximadamente el 6.38 % de la variabilidad de las horas de ausentismo en el conjunto de prueba. Por lo tanto, el desempeño predictivo del modelo base es limitado.

También se observaron algunas predicciones negativas. Estas no tienen una interpretación válida en términos de horas de ausentismo y representan una limitación de la Regresión Lineal, que no restringe automáticamente sus resultados a valores no negativos.

El objetivo principal de la entrega no es alcanzar el mejor desempeño posible, sino demostrar una implementación correcta, consistente y reutilizable del pipeline.

## Predicción sobre nuevos datos

El pipeline puede recibir un nuevo registro en formato `DataFrame` y aplicar automáticamente todas las transformaciones necesarias:

```python
nuevo_registro = pd.DataFrame({
    "Reason for Absence": [23],
    "Transportation Expense": [235],
    "Distance to Work": [20],
    "Age": [35],
    "Daily Work Load Average": [240.0],
    "Body Mass Index": [25],
    "Education": [1],
    "Children": [1],
    "Pets": [0],
    "Month": [7],
    "DayOfWeek": [2]
})

prediccion = model_pipeline.predict(nuevo_registro)
print(prediccion[0])
```

No es necesario imputar, escalar ni codificar manualmente el nuevo registro.

## Persistencia mediante joblib

El pipeline completo se guarda mediante `joblib`:

```python
joblib.dump(model_pipeline, "pipeline_ausentismo.joblib")
```

El objeto guardado incluye:

- Imputadores.
- Escalador.
- One-Hot Encoder.
- ColumnTransformer.
- Regresión Lineal.

Para reutilizarlo:

```python
pipeline_cargado = joblib.load("pipeline_ausentismo.joblib")
prediccion_recargada = pipeline_cargado.predict(nuevo_registro)
```

La predicción del pipeline recuperado debe coincidir con la predicción efectuada antes de guardar el archivo.

## Estructura recomendada del repositorio

```text
pipeline-ausentismo-laboral/
├── README.md
├── requirements.txt
├── Preentrega_Pipeline_Ausentismo.ipynb
├── Absenteeism_data.csv
└── pipeline_ausentismo.joblib
```

## Instalación

1. Clonar el repositorio:

```bash
git clone URL_DEL_REPOSITORIO
cd pipeline-ausentismo-laboral
```

2. Crear un entorno virtual, de manera opcional pero recomendada:

```bash
python -m venv .venv
```

3. Activar el entorno virtual.

En Windows:

```bash
.venv\Scripts\activate
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

4. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

5. Ejecutar Jupyter Notebook:

```bash
jupyter notebook
```

6. Abrir `Preentrega_Pipeline_Ausentismo.ipynb` y ejecutar las celdas en orden.

## Dependencias principales

- Python
- pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- joblib
- Jupyter Notebook

## Cumplimiento de los criterios de evaluación

| Criterio | Implementación |
|---|---|
| Pipeline de preprocesamiento y estimador | `Pipeline` final con `ColumnTransformer` y `LinearRegression` |
| Prevención de fuga de datos | División Train/Test antes de ejecutar `.fit()` |
| Tratamiento numérico | Imputación por mediana y `StandardScaler` |
| Tratamiento categórico | Imputación por moda y `OneHotEncoder` |
| Categorías desconocidas | `handle_unknown="ignore"` |
| Persistencia | Exportación y recuperación mediante `joblib` |
| Reutilización | Predicción sobre un nuevo `DataFrame` |
| Documentación | README, notebook comentado y estructura clara |

## Conclusiones

En este proyecto se desarrolló un flujo completo de Machine Learning para abordar la predicción de horas de ausentismo laboral.

El uso conjunto de `Pipeline` y `ColumnTransformer` permitió diferenciar el tratamiento de variables numéricas y categóricas y encapsular todas las etapas de preprocesamiento junto con el estimador final.

La división entre entrenamiento y prueba se realizó antes de ajustar las transformaciones. Esto permitió que los parámetros de imputación, escalado y codificación fueran aprendidos exclusivamente a partir del conjunto de entrenamiento, evitando `data leakage`.

Aunque la Regresión Lineal obtuvo un desempeño predictivo limitado, el pipeline cumple el objetivo metodológico: es funcional, reproducible, permite ejecutar `.predict()` sobre nuevos registros y puede guardarse y reutilizarse mediante `joblib`.

Como trabajo futuro se podrían comparar algoritmos capaces de representar relaciones no lineales, analizar estrategias específicas para valores extremos y evaluar transformaciones sobre la variable objetivo.

## Autora

Milagros Belén Brucellaria  
Curso de Data Science
