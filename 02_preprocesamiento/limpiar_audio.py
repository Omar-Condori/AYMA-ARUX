import librosa
import soundfile as sf
import os

INPUT = "01_corpus/audio_crudo"
OUTPUT = "01_corpus/audio_limpio"

def procesar_audio(ruta_entrada, ruta_salida):
    audio, sr = librosa.load(ruta_entrada, sr=None, mono=True)
    audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    sf.write(ruta_salida, audio_16k, 16000, subtype='PCM_16')
    print(f"✅ Procesado: {ruta_salida}")

for hablante in os.listdir(INPUT):
    for archivo in os.listdir(f"{INPUT}/{hablante}"):
        if archivo.endswith(".wav"):
            entrada = f"{INPUT}/{hablante}/{archivo}"
            salida  = f"{OUTPUT}/{hablante}/{archivo}"
            procesar_audio(entrada, salida)