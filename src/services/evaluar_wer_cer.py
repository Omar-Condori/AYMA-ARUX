import evaluate
import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from datasets import load_from_disk

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

MODEL = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(MODEL)

dataset = load_from_disk("datos/dataset")
val_set = dataset["validation"]

def evaluar(model, processor, val_set):
    referencias, predicciones = [], []
    for i, muestra in enumerate(val_set):
        audio = muestra["audio"]["array"]
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(**inputs, language="es", task="transcribe")
        pred = processor.batch_decode(ids, skip_special_tokens=True)[0]
        ref = muestra["text"]
        referencias.append(ref)
        predicciones.append(pred)
        print(f"  [{i+1}/{len(val_set)}] REF: {ref}")
        print(f"                  PRED: {pred}")
    return {
        "WER": wer_metric.compute(predictions=predicciones, references=referencias),
        "CER": cer_metric.compute(predictions=predicciones, references=referencias),
    }

print("Evaluando modelo BASE (sin LoRA)...")
modelo_base = WhisperForConditionalGeneration.from_pretrained(MODEL)
result_base = evaluar(modelo_base, processor, val_set)
print(f"\nBASE → WER: {result_base['WER']*100:.2f}% | CER: {result_base['CER']*100:.2f}%\n")

print("Evaluando modelo con LoRA (fine-tuned)...")
modelo_lora = WhisperForConditionalGeneration.from_pretrained(MODEL)
modelo_lora = PeftModel.from_pretrained(modelo_lora, "modelos/asr/final")
result_lora = evaluar(modelo_lora, processor, val_set)
print(f"\nLoRA  → WER: {result_lora['WER']*100:.2f}% | CER: {result_lora['CER']*100:.2f}%")

print("\n=== RESUMEN ===")
print(f"Modelo base:  WER {result_base['WER']*100:.2f}% | CER {result_base['CER']*100:.2f}%")
print(f"Modelo LoRA:  WER {result_lora['WER']*100:.2f}% | CER {result_lora['CER']*100:.2f}%")
print(f"Mejora WER:   {(result_base['WER'] - result_lora['WER']) * 100:.2f}%")
