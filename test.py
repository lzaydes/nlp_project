import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PhiForCausalLM

torch.set_default_device("cuda")

model = PhiForCausalLM.from_pretrained("susnato/phi-2", torch_dtype=torch.float16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("susnato/phi-2", trust_remote_code=True)


prompt = "Instruct:\nYou will be given a claim and using commonsense reasoning, you need to respond with SUPPORTS or REFUTES, depending on whether you support or refute the claim.\nClaim:Whats the best part of an ISIS joke? The Execution\nIs the claim fairness?\nRespond with SUPPORTS or REFUTES\nOutput:"

inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
