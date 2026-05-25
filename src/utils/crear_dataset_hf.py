import os
import pandas as pd
import numpy as np
import librosa
from datasets import Dataset, Audio, DatasetDict, Features, Value

RUTA_CSV = "datos/processed/metadata_clean.csv"
RUTA_AUDIOS = "datos/processed/audio"
RUTA_SALIDA = "datos/dataset"

os.makedirs(RUTA_SALIDA, exist_ok=True)

df = pd.read_csv(RUTA_CSV)

rutas = []
textos = []
textos_es = []
ids = []
generos = []
edades = []
regiones = []

for _, row in df.iterrows():
    archivo = row["archivo_wav"]
    ruta = os.path.join(RUTA_AUDIOS, archivo)
    if not os.path.exists(ruta):
        print(f"⚠ Audio no encontrado: {ruta}")
        continue
    rutas.append(ruta)
    textos.append(row["texto_aimara"])
    textos_es.append(row["texto_español"])
    ids.append(row["ID_hablante"])
    generos.append(row["genero"])
    edades.append(str(row["edad"]))
    regiones.append(row["dialect_region"])

features = Features({
    "audio": Audio(sampling_rate=16000),
    "text": Value("string"),
    "texto_espanol": Value("string"),
    "ID_hablante": Value("string"),
    "genero": Value("string"),
    "edad": Value("string"),
    "dialect_region": Value("string"),
})

dataset = Dataset.from_dict(
    {"audio": rutas, "text": textos, "texto_espanol": textos_es,
     "ID_hablante": ids, "genero": generos, "edad": edades,
     "dialect_region": regiones},
    features=features,
)

dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
dataset_final = DatasetDict({
    "train": dataset_split["train"],
    "validation": dataset_split["test"],
})

dataset_final.save_to_disk(RUTA_SALIDA)
n_train = len(dataset_final["train"])
n_val = len(dataset_final["validation"])
print(f"Dataset guardado en {RUTA_SALIDA}")
print(f"Train: {n_train} | Validation: {n_val}")
