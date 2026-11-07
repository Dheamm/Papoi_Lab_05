import io
import pandas as pd
from pyspark import SparkConf, SparkContext

# 1. Configuración de la sesión de Spark
conf = SparkConf().setAppName("Laboratorio05_Spark").setMaster("local[*]")
sc = SparkContext.getOrCreate(conf=conf)

# 2. Carga directa del nuevo archivo CSV (sin depender de openpyxl ni io)
df_pandas = pd.read_csv('dataset_steam.csv')

# Formatear los registros como tuplas (clave: año, valor: (nombre, rating))
steam_records = []
for _, row in df_pandas.iterrows():
    try:
        year = int(row['year']) if pd.notna(row['year']) else 0
        name = str(row['name'])
        rating = float(row['metacritic_rating']) if pd.notna(row['metacritic_rating']) else 0.0
        steam_records.append((year, (name, rating)))
    except Exception:
        continue

# Crear el RDD base con 4 particiones iniciales
rdd_base = sc.parallelize(steam_records, numSlices=4)



# 4 TRANSFORMACIONES RDD

# 1. mapPartitions: Procesa particiones completas localmente
def procesar_particion_local(iterator):
    conteo = 0
    suma_rating = 0.0
    for key, value in iterator:
        conteo += 1
        suma_rating += value[1]
    yield (conteo, suma_rating)

rdd_map_partitions = rdd_base.mapPartitions(procesar_particion_local)

# 2. aggregateByKey: Agrupa y acumula datos por clave cambiando la estructura
rdd_aggregated = rdd_base.aggregateByKey(
    (0.0, 0),
    lambda acc, val: (acc[0] + val[1], acc[1] + 1),
    lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])
)

# 3. sample: Genera una muestra aleatoria del 10% de los datos sin reemplazo
rdd_sampled = rdd_base.sample(withReplacement=False, fraction=0.1, seed=42)

# 4. coalesce: Reduce el número de particiones de forma eficiente a 2
rdd_coalesced = rdd_sampled.coalesce(numPartitions=2)

print("¡Transformaciones ejecutadas correctamente!")


# 4 ACCIONES RDD


print("\n" + "="*50)
print("EJECUTANDO LAS 4 ACCIONES NO CONVENCIONALES")
print("="*50)


# Acción 1: top()
# Obtiene los k elementos mayores según un criterio.

print("\n[Acción 1 - top]: Top 3 juegos con mejor rating Metacritic:")
top_3_juegos = rdd_base.top(3, key=lambda registro: registro[1][1])
for anio, (nombre, rating) in top_3_juegos:
    print(f"  - Año: {anio} | Juego: {nombre} | Rating: {rating}")



# Acción 2: takeSample() 
# Extrae un número fijo exacto de muestras al driver.
print("\n[Acción 2 - takeSample]: Muestra aleatoria exacta de 3 registros:")
muestra_fija = rdd_coalesced.takeSample(withReplacement=False, num=3, seed=99)
for anio, (nombre, rating) in muestra_fija:
    print(f"  - [{anio}] {nombre} (Rating: {rating})")



# Acción 3: countByValue()
# Conteo de frecuencia de cada elemento único localmente en el driver.
print("\n[Acción 3 - countByValue]: Frecuencia de juegos por año (primeros 5 años):")
conteo_anios = rdd_coalesced.keys().countByValue()
for anio, total in list(conteo_anios.items())[:5]:
    print(f"  - Año {anio}: {total} juego(s)")



# Acción 4: fold()
# Agrega los elementos en workers y luego en el driver con elemento neutro.
print("\n[Acción 4 - fold]: Agregación global sobre rdd_map_partitions:")
# rdd_map_partitions entrega tuplas (conteo_particion, suma_rating_particion)
total_global = rdd_map_partitions.fold(
    (0, 0.0),
    lambda acc, elem: (acc[0] + elem[0], acc[1] + elem[1])
)

total_registros, suma_ratings = total_global
promedio_rating = suma_ratings / total_registros if total_registros > 0 else 0.0

print(f"  - Total registros evaluados en particiones: {total_registros}")
print(f"  - Suma acumulada de ratings: {suma_ratings:.2f}")
print(f"  - Rating promedio global: {promedio_rating:.2f}")

print("\n" + "="*50)
print("¡Acciones completadas con éxito!")
print("="*50)
sc.stop()