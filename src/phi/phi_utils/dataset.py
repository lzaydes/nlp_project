import pandas as pd
from torch.utils.data import Dataset
from utils.file_utils import load_jsonl
from phi.phi_utils.constants import PHI_ZERO_SHOT_EVAL_PROMPT, PHI_FEW_SHOT_EVAL_PROMPT, PHI_ZERO_SHOT_EVIDENCE_EVAL_PROMPT, PHI_ZERO_SHOT_EVIDENCE_PROMPT
from transformers import AutoTokenizer, AutoModelForCausalLM

class PhiPromptDataset(Dataset):
    def __init__(self, annotations_filepath, prompt_type, evidence_filepath = None):
        self.data = load_jsonl(annotations_filepath)
        self.prompt_type = prompt_type

        if evidence_filepath is not None: 
            self.evidence_data = load_jsonl(evidence_filepath)
        else:
            self.evidence_data = None

    def __len__(self):
        return len(self.data)

    ############################################################
    # TODO: Please complete the implementation for the
    # the following transform functions and __getitem__ fn, that you 
    # will use in def __getitem__ to convert a sample into prompt.
    # You can use the templates provided to in the constants.py file

    # End of TODO.
    ##################################################
    def __getitem__(self, idx):

        prompt = ""
        ##################################################
        # TODO: Please complete the implementation of __getitem__
        # You may use if-else statements to choose the prompt
        # transform as per the prompt type given to you.
        if self.prompt_type == 'zero_eval':
            prompt = PHI_ZERO_SHOT_EVAL_PROMPT
            prompt = prompt.format(**self.data[idx])

        elif self.prompt_type == 'few_eval':
            prompt = PHI_FEW_SHOT_EVAL_PROMPT
            prompt = prompt.format(**self.data[idx])

        elif self.prompt_type == 'zero_evidence':
            prompt = PHI_ZERO_SHOT_EVIDENCE_PROMPT

            self.data[idx].update({"information": self.generate_information(idx)})
            prompt = prompt.format(**self.data[idx])
            generate_evidence(prompt)

        elif self.prompt_type == 'zero_evidence_eval':
            prompt = PHI_ZERO_SHOT_EVIDENCE_EVAL_PROMPT
            evidence = self.evidence_data

            self.data[idx].update(self.evidence_data[idx])

            prompt = prompt.format(**self.data[idx])
        # End of TODO.
        ##################################################
        
        return prompt

    def generate_information(self, idx):
        domain = self.data[idx]["domain"]

        if domain == "sbic":
            return "The claim comes from the Social Bias Inference Corpus dataset, which contains real-world human-generated social media posts.\n"
        elif domain == "toxigen":
            return "The claim comes from the GPT Toxicity dataset, which contains machine-generated toxic and benign statements about minority groups.\n"
        elif domain == "hsd":
            return "The claim comes from the Hate Speech Detection dataset, which contains human-generated hate speech from a white supremacy forum. \n"
        elif domain == "mgfn":
            return "The claim comes from the Machine Generated Fake News UniLC dataset, which contains machine-generated news.\n"
        elif domain == "climate":
            return "The claim comes from the Climate Fever dataset, which is a dataset of human generated climate-change-related claims.\n"
        elif domain == "health":
            return "The claim comes from a Health fact-checking dataset, which contains human generated healh claims.\n"
        
def generate_evidence(prompt, model_id_or_path="mistralai/Mistral-7B-Instruct-v0.2"):
    evidence = ''
    model, tokenizer = None, None

    device = "cuda" # the device to load the model onto
   
    model = AutoModelForCausalLM.from_pretrained(model_id_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
    
    encodeds = tokenizer.apply_chat_template(prompt, return_tensors="pt")

    model_inputs = encodeds.to(device)
    model.to(device)

    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)

    evidence = decoded[0]

    print(evidence)
    return evidence