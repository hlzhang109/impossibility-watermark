import guidance
from guidance import models, gen, select, user, system, assistant

@guidance
def annotate_entropy_by_instructions(lm, prompt, persona=None):
    if persona:
        with system():
            lm += f"{persona}"
    with user():
        lm += f"""\
        ### Introduction
        In this context, entropy refers to the unpredictability and variety of the responses. 
        Instructions with high entropy will lead to responses that are less predictable and more varied, while low entropy instructions will be more predictable and uniform.
        
        ### Task Description: 
        1. Please assess the entropy of the instructions based on the following criteria:
            - Unpredictability Potential: How likely are the instructions to elicit a wide range of unpredictable responses?
            - Variety Potential: How likely are the instructions to generate responses with varied language and content?
            - Informativeness Potential: How likely are the responses to provide diverse and informative content beyond the basic requirements?
        2. For the given instructions, assign a score from 0 to 9, where 0 indicates very low entropy potential and 9 indicates very high entropy potential.

        ### The instructions to evaluate:
        {prompt}
        """
    with assistant():
        lm += f"""\
        ### Entropy Assessment: 
        Entropy Score: {gen(regex='[0-9]', name='entropy_level_prompt')}
        """
    return lm

@guidance
def annotate_entropy_by_instructions_w_exp(lm, prompt, persona=None):
    if persona:
        with system():
            lm += f"{persona}"
    with user():
        lm += f"""\
        ### Introduction
        In this context, entropy refers to the unpredictability and variety of the responses. 
        Instructions with high entropy will lead to responses that are less predictable and more varied, while low entropy instructions will be more predictable and uniform.
        
        ### Task Description: 
        1. Please assess the entropy of the instructions based on the following criteria:
            - Unpredictability Potential: How likely are the instructions to elicit a wide range of unpredictable responses?
            - Variety Potential: How likely are the instructions to generate responses with varied language and content?
            - Informativeness Potential: How likely are the responses to provide diverse and informative content beyond the basic requirements?
        2. For the given instructions:
            - provide a one-sentence analysis of their potential entropy.
            - assign a score from 0 to 9, where 0 indicates very low entropy potential and 9 indicates very high entropy potential.

        ### The instructions to evaluate:
        {prompt}
        """
    with assistant():
        lm += f"""\
        ### Entropy Assessment: 
        Entropy Analysis: {gen(f'entropy_analysis', stop='.')}.
        Entropy Score: {gen(regex='[0-9]', name='entropy_level_prompt_w_exp')}
        """
    return lm

@guidance
def annotate_entropy_by_instructions_and_responses(lm, prompt, response_a, response_b, persona=None):
    if persona:
        with system():
            lm += f"{persona}"
    with user():
        lm += f"""\
        ### Introduction
        In this context, entropy refers to the unpredictability and variety of the responses. 
        Instructions with high entropy will lead to responses that are less predictable and more varied, while low entropy instructions will be more predictable and uniform.
        
        ### Task Description: 
        1. Please assess the entropy of the instructions based on the following criteria:
            - Unpredictability Potential: How likely are the instructions to elicit a wide range of unpredictable responses?
            - Variety Potential: How likely are the instructions to generate responses with varied language and content?
            - Informativeness Potential: How likely are the responses to provide diverse and informative content beyond the basic requirements?
        2. For the given instructions:
            - assign a score from 0 to 9, where 0 indicates very low entropy potential and 9 indicates very high entropy potential.

        ### The instructions to evaluate:
        {prompt}

        ### Sample generation #1:
        {response_a}

        ### Sample generation #1:
        {response_b}
        """
    with assistant():
        lm += f"""\
        ### Entropy Assessment: 
        Entropy Score: {gen(regex='[0-9]', name='entropy_level_prompt_and_responses')}
        """
    return lm