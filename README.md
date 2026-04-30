## 🧠 Siglas del proyecto

| Sigla | Significado en inglés        | Significado simple                          |
|-------|------------------------------|---------------------------------------------|
| ASR   | Automatic Speech Recognition | El sistema escucha y convierte voz en texto |
| MT    | Machine Translation          | El sistema traduce texto a otro idioma      |
| TTS   | Text To Speech               | El sistema habla convirtiendo texto en voz  |




## 📁 Estructura del proyecto

| Carpeta                       | Qué debe contener                                                        |
|------------------------------ |--------------------------------------------------------------------------|
| `01_corpus/audio_crudo/`      | Los WAV tal como salen de la grabadora — sin tocar                       |
| `01_corpus/audio_limpio/`     | Los WAV ya procesados a 16kHz mono — después de correr `limpiar_audio.py`|
| `01_corpus/anotaciones_elan/` | Los archivos `.eaf` que exportas desde el programa ELAN                  |
| `01_corpus/transcripciones/`  | Archivos `.csv` con el texto aimara y español de cada audio              |
| `01_corpus/dataset_hf/`       | El dataset listo para entrenar — lo genera `crear_dataset_hf.py`         |
| `03_asr/checkpoints/`         | Se llena sola cuando entrenas LoRA — guarda el progreso                  |
| `03_asr/modelo_lora_aimara/`  | El modelo final ya entrenado — lo genera `entrenar_lora_whisper.py`      |
| `03_asr/resultados/`          | Archivos `.json` o `.csv` con los números WER y CER                      |
| `04_mt/resultados/`           | Archivos con los números BLEU y chrF                                     |
| `05_tts/audio_sintetizado/`   | Los audios WAV que genera el sistema TTS                                 |
| `05_tts/resultados/`          | Archivos con los números MCD, PESQ, DNSMOS                               |
| `06_pipeline/resultados/`     | Resultados finales de todo el sistema completo                           |
| `notebooks/`                  | Los archivos `.ipynb` que usas en Google Colab                           |




# Crear entorno virtual
python3 -m venv venv

# Activar en Mac
source venv/bin/activate

# Verás esto cuando esté activo:
(venv) omarcondori@MacBook-Pro-de-Omar AYMA-ARUX %

# Instalar librerías
pip install -r requirements.txt