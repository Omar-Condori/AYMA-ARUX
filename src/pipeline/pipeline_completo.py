# Pipeline voz a voz completo: ASR → MT → TTS
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                           NllbTokenizer, AutoModelForSeq2SeqLM,
                           VitsModel, VitsTokenizer)
from peft import PeftModel
import librosa, torch, soundfile as sf

def voz_a_voz(ruta_audio_entrada, ruta_audio_salida, direccion="aym-spa"):
    # 1. ASR — voz a texto
    proc_asr  = WhisperProcessor.from_pretrained("openai/whisper-small")
    model_asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model_asr = PeftModel.from_pretrained(model_asr, "modelos/asr/final")

    audio, _ = librosa.load(ruta_audio_entrada, sr=16000)
    inputs   = proc_asr(audio, sampling_rate=16000, return_tensors="pt")
    ids      = model_asr.generate(**inputs, language="ay")
    texto_asr = proc_asr.batch_decode(ids, skip_special_tokens=True)[0]
    print(f"ASR → {texto_asr}")

    # 2. MT — texto a texto
    src, tgt = ("ayr_Latn", "spa_Latn") if direccion == "aym-spa" else ("spa_Latn", "ayr_Latn")
    tok_mt  = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=src)
    mod_mt  = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    inputs_mt  = tok_mt(texto_asr, return_tensors="pt")
    ids_mt     = mod_mt.generate(**inputs_mt, forced_bos_token_id=tok_mt.lang_code_to_id[tgt])
    texto_trad = tok_mt.batch_decode(ids_mt, skip_special_tokens=True)[0]
    print(f"MT  → {texto_trad}")

    # 3. TTS — texto a voz
    tok_tts  = VitsTokenizer.from_pretrained("facebook/mms-tts-aym")
    mod_tts  = VitsModel.from_pretrained("facebook/mms-tts-aym")
    inputs_tts = tok_tts(texto_trad, return_tensors="pt")
    with torch.no_grad():
        audio_out = mod_tts(**inputs_tts).waveform.squeeze().numpy()
    sf.write(ruta_audio_salida, audio_out, 16000)
    print(f"TTS → guardado en {ruta_audio_salida}")

if __name__ == "__main__":
    voz_a_voz("datos/processed/audio/SPK00001_00001.wav", "resultados/pipeline/resultado.wav", direccion="aym-spa")