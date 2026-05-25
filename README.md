## 🧠 Siglas del proyecto

| Sigla | Significado en inglés        | Significado simple                          |
|-------|------------------------------|---------------------------------------------|
| ASR   | Automatic Speech Recognition | El sistema escucha y convierte voz en texto |
| MT    | Machine Translation          | El sistema traduce texto a otro idioma      |
| TTS   | Text To Speech               | El sistema habla convirtiendo texto en voz  |




## 📁 Estructura del proyecto

| Carpeta                       | Qué debe contener                                                        |
|------------------------------ |--------------------------------------------------------------------------|
| `datos/raw/audio/`            | Los WAV originales (SPK00001_00001.wav ...)                              |
| `datos/datos_corpus.csv`      | Transcripciones aimara + español de cada audio                           |
| `datos/dataset/`              | Dataset HuggingFace listo para entrenar — lo genera `crear_dataset_hf.py`|
| `modelos/asr/checkpoints/`    | Checkpoints durante entrenamiento LoRA                                   |
| `modelos/asr/final/`          | Modelo LoRA entrenado — lo genera `entrenar_lora_whisper.py`             |
| `src/utils/`                  | Utilidades: limpiar audio, generar metadata, crear dataset               |
| `src/services/`               | Servicios: ASR, MT, TTS, evaluaciones                                    |
| `src/pipeline/`               | Pipeline completo ASR → MT → TTS                                         |
| `notebooks/`                  | Notebook de Google Colab para entrenar                                   |
| `resultados/`                 | Salidas del pipeline, evaluaciones                                       |




# Crear entorno virtual
python3 -m venv venv

# Activar en Mac
source venv/bin/activate

# Verás esto cuando esté activo:
(venv) omarcondori@MacBook-Pro-de-Omar AYMA-ARUX %

# Instalar librerías
pip install -r requirements.txt