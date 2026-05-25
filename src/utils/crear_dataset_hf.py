import os
import pandas as pd
from datasets import Dataset, Audio, DatasetDict, Features, Value

RUTA_CSV = "datos/processed/metadata_clean.csv"
RUTA_AUDIOS = "datos/processed/audio"
RUTA_SALIDA = "datos/dataset"

os.makedirs(RUTA_SALIDA, exist_ok=True)

df = pd.read_csv(RUTA_CSV)

speakers_unicos = sorted(df["ID_hablante"].unique())
total_speakers = len(speakers_unicos)  # 110 SPK IDs
num_grupos = 6
tam_grupo = total_speakers // num_grupos

speaker_to_group = {}
for i, spk in enumerate(speakers_unicos):
    grupo = min(i // tam_grupo, num_grupos - 1)
    speaker_to_group[spk] = f"G{grupo+1}"

train_groups = {"G1", "G2", "G3", "G4"}
val_groups = {"G5"}
test_groups = {"G6"}

df["grupo_hablante"] = df["ID_hablante"].map(speaker_to_group)

train_df = df[df["grupo_hablante"].isin(train_groups)]
val_df = df[df["grupo_hablante"].isin(val_groups)]
test_df = df[df["grupo_hablante"].isin(test_groups)]

def df_to_dataset(df_split, nombre):
    rutas, textos, textos_es, ids, generos, edades, regiones = [], [], [], [], [], [], []
    for _, row in df_split.iterrows():
        archivo = row["archivo_wav"]
        ruta = os.path.join(RUTA_AUDIOS, archivo)
        if not os.path.exists(ruta):
            print(f"⚠ [{nombre}] Audio no encontrado: {ruta}")
            continue
        rutas.append(ruta)
        textos.append(row["texto_aimara"])
        textos_es.append(row["texto_español"])
        ids.append(row["ID_hablante"])
        generos.append(row["genero"])
        edades.append(str(row["edad"]))
        regiones.append(row["dialect_region"])
    print(f"  {nombre}: {len(rutas)} archivos | {df_split['ID_hablante'].nunique()} hablantes")
    return Dataset.from_dict(
        {"audio": rutas, "text": textos, "texto_espanol": textos_es,
         "ID_hablante": ids, "genero": generos, "edad": edades,
         "dialect_region": regiones},
        features=Features({
            "audio": Audio(sampling_rate=16000),
            "text": Value("string"), "texto_espanol": Value("string"),
            "ID_hablante": Value("string"), "genero": Value("string"),
            "edad": Value("string"), "dialect_region": Value("string"),
        }),
    )

print("Creando datasets por grupo de hablante:")
print(f"  {len(train_groups)} grupos train, {len(val_groups)} val, {len(test_groups)} test")
print()

dataset_final = DatasetDict({
    "train": df_to_dataset(train_df, "Train"),
    "validation": df_to_dataset(val_df, "Val"),
    "test": df_to_dataset(test_df, "Test"),
})

dataset_final.save_to_disk(RUTA_SALIDA)
print(f"\nDataset guardado en {RUTA_SALIDA}")
print(f"Train: {len(dataset_final['train'])} | Val: {len(dataset_final['validation'])} | Test: {len(dataset_final['test'])}")
