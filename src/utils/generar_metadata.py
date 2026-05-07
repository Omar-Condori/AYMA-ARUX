import os
import csv
import librosa

CARPETA_AUDIOS = "01_corpus/audio_limpio"
SALIDA_CSV = "01_corpus/transcripciones/metadata.csv"

os.makedirs("01_corpus/transcripciones", exist_ok=True)

filas = []

for hablante in sorted(os.listdir(CARPETA_AUDIOS)):
    carpeta_hablante = os.path.join(CARPETA_AUDIOS, hablante)
    if not os.path.isdir(carpeta_hablante):
        continue
    for archivo in sorted(os.listdir(carpeta_hablante)):
        if not archivo.endswith(".wav"):
            continue
        ruta = os.path.join(carpeta_hablante, archivo)
        duracion = librosa.get_duration(path=ruta)
        nombre_sin_ext = archivo.replace(".wav", "")
        partes = nombre_sin_ext.split("_")
        bloque = partes[2] if len(partes) >= 3 else "?"
        filas.append({
            "ID_enunciado": nombre_sin_ext,
            "ID_hablante": hablante,
            "bloque": bloque,
            "archivo_wav": archivo,
            "texto_aimara": "",      # llenarás después
            "texto_español": "",     # llenarás después
            "duracion_seg": round(duracion, 2),
            "snr_db": "",            # llenarás después
            "calidad_audio": ""      # llenarás después
        })

with open(SALIDA_CSV, "w", newline="", encoding="utf-8") as f:
    campos = ["ID_enunciado","ID_hablante","bloque","archivo_wav",
              "texto_aimara","texto_español","duracion_seg","snr_db","calidad_audio"]
    writer = csv.DictWriter(f, fieldnames=campos)
    writer.writeheader()
    writer.writerows(filas)

print(f"✅ metadata.csv creado con {len(filas)} filas")
print(f"📁 Guardado en: {SALIDA_CSV}")