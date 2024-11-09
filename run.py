from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t12_35M_UR50D")

inputs = tokenizer("A", return_tensors="pt")
outputs = model(**inputs)

print(outputs)