from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

save_path = "./t5-base-offline"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print("✅ Model saved to:", save_path)
