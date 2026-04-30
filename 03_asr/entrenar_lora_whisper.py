from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk

MODEL = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(MODEL)
model     = WhisperForConditionalGeneration.from_pretrained(MODEL)

# Configuración LoRA r=8
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Verifica que sea < 1%

dataset = load_from_disk("01_corpus/dataset_hf")

args = Seq2SeqTrainingArguments(
    output_dir="03_asr/checkpoints",
    per_device_train_batch_size=4,
    num_train_epochs=10,
    learning_rate=1e-4,
    fp16=True,
    save_steps=100,
    eval_strategy="steps",
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)
trainer.train()
model.save_pretrained("03_asr/modelo_lora_aimara")