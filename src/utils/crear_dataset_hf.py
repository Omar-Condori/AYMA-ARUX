import os
import pandas as pd
import librosa
from datasets import Dataset, Audio, DatasetDict

RUTA_CSV = "datos/processed/metadata_clean.csv"
RUTA_AUDIOS = "datos/processed/audio"
RUTA_SALIDA = "datos/dataset"

os.makedirs(RUTA_SALIDA, exist_ok=True)

df = pd.read_csv(RUTA_CSV)

datos = []
for _, row in df.iterrows():
    archivo = row["archivo_wav"]
    ruta = os.path.join(RUTA_AUDIOS, archivo)
    if not os.path.exists(ruta):
        print(f"⚠ Audio no encontrado: {ruta}")
        continue
    audio, sr = librosa.load(ruta, sr=16000)
    datos.append({
        "audio": {"path": ruta, "array": audio, "sampling_rate": 16000},
        "text": row["texto_aimara"],
        "texto_espanol": row["texto_español"],
        "ID_hablante": row["ID_hablante"],
        "genero": row["genero"],
        "edad": row["edad"],
        "dialect_region": row["dialect_region"],
    })

dataset = Dataset.from_list(datos)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
dataset_final = DatasetDict({
    "train": dataset_split["train"],
    "validation": dataset_split["test"],
})

dataset_final.save_to_disk(RUTA_SALIDA)
print(f"Dataset guardado en {RUTA_SALIDA}")
print(f"Train: {len(dataset_final['train'])} | Validation: {len(dataset_final['validation'])}")
