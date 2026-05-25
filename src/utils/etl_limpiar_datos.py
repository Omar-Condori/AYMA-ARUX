import os, re, unicodedata
import pandas as pd
import librosa
import soundfile as sf
import numpy as np

RUTA_CSV = "datos/datos_corpus.csv"
RUTA_AUDIOS_RAW = "datos/raw/audio"
RUTA_AUDIOS_LIMPIO = "datos/processed/audio"
RUTA_CSV_LIMPIO = "datos/processed/metadata_clean.csv"

os.makedirs(RUTA_AUDIOS_LIMPIO, exist_ok=True)

def limpiar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).strip()
    texto = re.sub(r'\s+', ' ', texto)
    texto = unicodedata.normalize('NFC', texto)
    return texto

def procesar_audio(ruta_entrada, ruta_salida):
    try:
        audio, sr = librosa.load(ruta_entrada, sr=None, mono=True)
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        max_val = np.max(np.abs(audio_16k))
        if max_val > 0:
            audio_16k = audio_16k / max_val
        sf.write(ruta_salida, audio_16k, 16000, subtype='PCM_16')
        duracion = librosa.get_duration(path=ruta_salida)
        return True, round(duracion, 2)
    except Exception as e:
        return False, str(e)

df = pd.read_csv(RUTA_CSV)
print(f"Total filas en CSV: {len(df)}")

df["texto_aimara"] = df["texto_aimara"].apply(limpiar_texto)
df["texto_español"] = df["texto_español"].apply(limpiar_texto)

antes = len(df)
df = df[df["texto_aimara"] != ""]
df = df[df["texto_español"] != ""]
print(f"Filas con texto vacío eliminadas: {antes - len(df)}")

resultados = []
for _, row in df.iterrows():
    archivo = row["archivo_wav"]
    entrada = os.path.join(RUTA_AUDIOS_RAW, archivo)
    salida = os.path.join(RUTA_AUDIOS_LIMPIO, archivo)
    if os.path.exists(entrada):
        exito, info = procesar_audio(entrada, salida)
        if exito:
            resultados.append({**row.to_dict(), "duracion_seg": info})
        else:
            print(f"  Error al procesar {archivo}: {info}")
    else:
        print(f"  Audio no encontrado: {archivo}")

df_limpio = pd.DataFrame(resultados)
print(f"\nTotal archivos procesados: {len(df_limpio)}")
print(f"Duración total: {df_limpio['duracion_seg'].sum():.2f}s")

df_limpio.to_csv(RUTA_CSV_LIMPIO, index=False, encoding="utf-8")
print(f"\nCSV limpio guardado en: {RUTA_CSV_LIMPIO}")
print("ETL completado")
