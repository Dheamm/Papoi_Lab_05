import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round
from pyspark.sql.types import DoubleType, IntegerType

# PASO 1: Generar CSV limpio y tipado

csv_path = "dataset_steam.csv"
if not os.path.exists(csv_path):
    print("Convirtiendo Excel a CSV limpio con pandas...")
    df_raw = pd.read_excel("dataset_steam.xlsx")
    # Si viene empaquetado en texto, unificamos y separamos por comas
    if len(df_raw.columns) <= 2:
        lineas = df_raw.iloc[:, 0].astype(str) + df_raw.iloc[:, 1].fillna("").astype(str)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(df_raw.columns[0] + "\n")
            f.write("\n".join(lineas))
    else:
        df_raw.to_csv(csv_path, index=False)

spark = SparkSession.builder \
    .appName("Lab05_Celula3_DataFrame") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

df_steam = spark.read.csv(csv_path, header=True, inferSchema=True)

df_steam = df_steam \
    .withColumn("to_beat_main", col("to_beat_main").cast(DoubleType())) \
    .withColumn("metacritic_rating", col("metacritic_rating").cast(DoubleType())) \
    .withColumn("to_beat_completionist", col("to_beat_completionist").cast(DoubleType()))


# PASO 2: Pipeline Base + 2 Funciones (Matías Palma)

# 1. Funciones vistas en clase: select() y filter()
df_base = df_steam.select(
    "id", "name", "year", "metacritic_rating", 
    "reviewer_rating", "to_beat_main", "to_beat_completionist", "extra_content_length"
).filter(
    col("metacritic_rating").isNotNull() & (col("to_beat_main") > 0)
)

# 2. Función investigada 1: drop() -> descarta columna redundante
df_limpio = df_base.drop("extra_content_length")

# 3. Función investigada 2: withColumn() -> horas extra requeridas para completarlo al 100%
df_transformado = df_limpio.withColumn(
    "horas_extra_100", 
    round(col("to_beat_completionist") - col("to_beat_main"), 2)
)

print("\n--- Salida DataFrame Transformado (Primeras 10 filas) ---")
df_transformado.show(10, truncate=False)

# ==========================================
# PASO 3: Pendiente para Ítalo Osorio
# ==========================================
# Ítalo:
# 1. groupBy("year").agg(...)
# 2. orderBy(...)
# 3. .explain(True)

spark.stop()