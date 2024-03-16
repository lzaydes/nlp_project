import pandas as pd
from torch.utils.data import Dataset
from utils.file_utils import load_jsonl
from phi.phi_utils.constants import PHI_ZERO_SHOT_EVAL_PROMPT, PHI_FEW_SHOT_EVAL_PROMPT, PHI_ZERO_SHOT_EVIDENCE_EVAL_PROMPT, PHI_ZERO_SHOT_EVIDENCE_PROMPT

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
            return prompt
        elif self.prompt_type == 'few_eval':
            prompt = PHI_FEW_SHOT_EVAL_PROMPT
        elif self.prompt_type == 'zero_evidence':
            prompt = PHI_ZERO_SHOT_EVIDENCE_PROMPT
        elif self.prompt_type == 'zero_evidence_eval':
            prompt = PHI_ZERO_SHOT_EVIDENCE_EVAL_PROMPT
        
        prompt = prompt.format(**self.data[idx])
        # End of TODO.
        ##################################################
        
        return prompt


    def zero_shot_evidence_prompt_transform(self, idx):
        prompt = ""


        return prompt

    # gtp4 for classification/evidence output?
    # zero_shot_evidence_prompt generates data for a given claim
    # it seems that this function evaluates the results? and compares them to evidence in dummy_evidence.json
    def zero_shot_evidence_evaluate_prompt_transform(self, idx):
        prompt = PHI_ZERO_SHOT_EVIDENCE_EVAL_PROMPT
        evidence = self.evidence_data

        prompt = prompt.format(**self.data[idx])

        ###             append evidence         ###
        ###             TODO                    ###

        

        return prompt
        
