PHI_ZERO_SHOT_EVAL_PROMPT = '''
Instruct:
You will be given a claim and using commonsense reasoning, you need to respond with SUPPORTS or REFUTES, depending on whether you support or refute the claim.
Claim:{claim}
Is the claim {task_type}? 
Respond with SUPPORTS or REFUTES
Output:
'''

PHI_FEW_SHOT_EVAL_PROMPT = '''
Instruct:
You will be given a claim and using commonsense reasoning, you need to respond with SUPPORTS or REFUTES, depending on whether you support or refute the claim.

Following are some examples:
{examples}

Now Your Turn
Claim:{claim}
Is the claim {task_type}? 
Respond with SUPPORTS or REFUTES
Output:
'''

PHI_ZERO_SHOT_EVIDENCE_PROMPT = '''
Instruct:
You will be given a claim and information and you have to determine the fairness or factuality of the claim, given "task_type".
If "task_type" is "fairness", determine the fairness of the claim using information as additional context.
If "task_type" is "fact", determine the factuality of the claim using information as additional context.

You have to generate a detailed evidence for the fairness or factuality of the claim. Your evidence must be within one sentences long.

Claim: {claim}
task_type: {task_type}
Information: {information}
Evidence Output:
'''

PHI_ZERO_SHOT_EVIDENCE_EVAL_PROMPT = '''
Instruct:
You will be given a claim and evidence for the claim. Using commonsense reasoning, claim and evidence, you need to respond with SUPPORTS or REFUTES, depending on whether you support or refute the claim.
Claim:{claim}
Evidence: {evidence}
Is the claim {task_type}? 
Respond with SUPPORTS or REFUTES
Output:
'''
