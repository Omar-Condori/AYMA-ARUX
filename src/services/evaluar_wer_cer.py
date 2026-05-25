import evaluate
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
import librosa, json

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def evaluar_modelo(model, processor, muestras):
    referencias, predicciones = [], []
    for muestra in muestras:
        audio, _ = librosa.load(muestra["audio"], sr=16000)
        inputs   = processor(audio, sampling_rate=16000, return_tensors="pt")
        ids      = model.generate(**inputs)
        pred     = processor.batch_decode(ids, skip_special_tokens=True)[0]
        referencias.append(muestra["texto"])
        predicciones.append(pred)
    return {
        "WER": wer_metric.compute(predictions=predicciones, references=referencias),
        "CER": cer_metric.compute(predictions=predicciones, references=referencias),
    }

# Cargar muestras de prueba
with open("01_corpus/test_samples.json") as f:
    muestras = json.load(f)

# Evaluar ANTES de LoRA (zero-shot)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
modelo_base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
print("ZERO-SHOT:", evaluar_modelo(modelo_base, processor, muestras))

# Evaluar DESPUÉS de LoRA
modelo_lora = PeftModel.from_pretrained(modelo_base, "modelos/asr/final")
print("POST-LORA:", evaluar_modelo(modelo_lora, processor, muestras))