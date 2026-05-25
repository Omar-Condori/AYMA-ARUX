import os
import pandas as pd
import librosa
import numpy as np

RUTA_CSV = "datos/processed/metadata_clean.csv"
RUTA_AUDIOS = "datos/processed/audio"

df = pd.read_csv(RUTA_CSV)
print(f"Total archivos: {len(df)}")

total_seg = df["duracion_seg"].sum()
total_horas = total_seg / 3600
print(f"Duración total: {total_seg:.2f}s = {total_horas:.2f} horas = {total_horas*60:.1f} minutos")

num_hablantes = df["ID_hablante"].nunique()
print(f"Hablantes únicos: {num_hablantes}")

print(f"\n--- Por género ---")
print(df["genero"].value_counts())

print(f"\n--- Por rango de edad ---")
df["rango_edad"] = pd.cut(df["edad"].astype(int), bins=[0, 20, 30, 40, 50, 60, 100], labels=["18-20", "21-30", "31-40", "41-50", "51-60", "60+"])
print(df["rango_edad"].value_counts().sort_index())

print(f"\n--- Por región dialectal ---")
print(df["dialect_region"].value_counts())

print(f"\n--- SNR desde CSV ---")
print(df["snr_db"].describe())

snr_reales = []
errores = []
print(f"\n--- Calculando SNR real desde audios ({len(df)} archivos) ---")
for _, row in df.iterrows():
    archivo = row["archivo_wav"]
    ruta = os.path.join(RUTA_AUDIOS, archivo)
    if not os.path.exists(ruta):
        errores.append(archivo)
        continue
    try:
        audio, sr = librosa.load(ruta, sr=16000)
        señal = np.mean(audio ** 2)
        ruido_idx = np.where(np.abs(audio) < np.percentile(np.abs(audio), 10))[0]
        if len(ruido_idx) > 0:
            ruido = np.mean(audio[ruido_idx] ** 2)
            if ruido > 0:
                snr = 10 * np.log10(señal / ruido)
                snr_reales.append(snr)
    except Exception as e:
        errores.append(archivo)

snr_reales = np.array(snr_reales)
print(f"SNR real calculado en {len(snr_reales)} archivos:")
print(f"  Media: {snr_reales.mean():.2f} dB")
print(f"  Mín:   {snr_reales.min():.2f} dB")
print(f"  Máx:   {snr_reales.max():.2f} dB")
print(f"  Std:   {snr_reales.std():.2f} dB")
if errores:
    print(f"Errores: {len(errores)} archivos")
