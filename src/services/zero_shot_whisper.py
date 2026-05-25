from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch, librosa

MODEL = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(MODEL)
model     = WhisperForConditionalGeneration.from_pretrained(MODEL)

def transcribir(ruta_audio):
    audio, _ = librosa.load(ruta_audio, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(**inputs, language="ay", task="transcribe")
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

# Prueba rápida
print(transcribir("datos/raw/audio/SPK00001_00001.wav"))