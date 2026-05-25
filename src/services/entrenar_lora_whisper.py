from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorSpeechSeq2SeqWithPadding
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk

MODEL = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(MODEL, language="aymara", task="transcribe")
model     = WhisperForConditionalGeneration.from_pretrained(MODEL)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_from_disk("datos/dataset")

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

args = Seq2SeqTrainingArguments(
    output_dir="modelos/asr/checkpoints",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    learning_rate=1e-4,
    warmup_steps=50,
    fp16=True,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    predict_with_generate=True,
    generation_max_length=128,
    logging_steps=25,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
)
trainer.train()
model.save_pretrained("modelos/asr/final")
processor.save_pretrained("modelos/asr/final")
print("Modelo guardado en modelos/asr/final")
