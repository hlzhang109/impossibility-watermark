import random
import nltk
from nltk.tokenize import sent_tokenize
import guidance
from guidance import models, gen, select, user, assistant
import hydra
import logging
import textwrap

from oracles.custom import Oracle
from oracles.utils import *

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

def extract_dict(output, keys):
    return {k: output[k] for k in keys}

def generate_comparison_prompts(instruction, output_1, output_2):
    prompt_1 = textwrap.dedent(f"""\
        ### Task Description: 
        1. You will be comparing two responses to a given prompt to determine which response is better. 
        2. Carefully analyze both responses, considering factors such as:
            - Repetition: Does either response repeat information unnecessarily?
            - Grammar: Are there any grammatical errors, capitalization errors, or punctuation mistakes in either response?
            - Coherence: Is each response well-structured and easy to understand?
            - Relevance: Does each response directly address the given prompt?
            - Accuracy: Is the information provided in each response accurate and factual?
        3. Keep in mind that having grammatical errors, repetitions, capitalization errors, or punctuation mistakes would greatly degrade the quality of a response.

        ### The instructions prompting the responses: 
        {instruction}

        ### Response A: 
        {output_1}

        ### Response B: 
        {output_2}""")

    prompt_2 = f"""So, what's your decision? If you think Response A is better than Response B, respond with an "A". If you think Response B is better than Response A, respond with a "B". If you think they
    have equal quality, respond with "0". Only respond with these."""

    return prompt_1, prompt_2

class RelativeOracle3(Oracle):  
    def __init__(self, cfg, pipeline = None, llm = None) -> None:
        super().__init__(cfg, pipeline, llm)

    def evaluate(self, instruction, output_1, output_2, **kwargs):
        if self.use_gpt:
            return self._gpt_evaluate(instruction, output_1, output_2, **kwargs)
        else:
            return self._guidance_evaluate(instruction, output_1, output_2, **kwargs)

    def _gpt_evaluate(self, instruction, output_1, output_2, **kwargs):
        prompt_1, prompt_2 = generate_comparison_prompts(instruction, output_1, output_2)

        # TODO: Turn the number of retries into a parameter.
        for _ in range(3):
            first, second = query_openai_with_history(prompt_1, prompt_2, model=self.cfg.model_id)
            log.info(f"Model's First Response: {first.content}")
            log.info(f"Model's Second Response: {second.content}")
            log.info("---------------------------------------------------------------")
        
            analysis = first.content
            answer = second.content[0]

            if answer in ["A", "B", "0"]:
                log.info(f"Valid answer: {answer}")
                # TODO: This is abysmal code.
                answer = "Equal" if answer == "0" else answer
                return {
                    "analysis": analysis, 
                    "answer": answer
                }
        
        # TODO: Improve the Exception handling.
        raise Exception("Oracle couldn't produce a valid answer.")

    def _guidance_evaluate(self, instruction, output_1, output_2, **kwargs):
        analysis = self.llm + produce_quality_analysis(instruction, output_1, output_2)
        analysis = analysis["analysis"]

        output = self.llm + produce_answer(analysis)

        log.info(f"Answer: {output['answer']}")
        log.info(f"Answer Type: {type(output['answer'])}")

        # TODO: Put this function somewhere else.
        # Mapping function
        def map_answer_to_text(answer):
            answer = int(answer)
            mapping = {1: "A", 2: "B", 0: "Equal"}
            return mapping.get(answer, "Invalid")
        
        answer = map_answer_to_text(output['answer'])

        log.info(f"Mapped Answer: {answer}")

        return {
            "analysis": analysis, 
            "answer": answer
        }

    def evaluate_with_retries(self, instruction, output_1, output_2, max_retries=3, **kwargs):
        for attempt in range(max_retries):
            try:
                return self.evaluate(instruction, output_1, output_2, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    log.warning(f"Evaluation attempt {attempt + 1} failed with error: {e}. Retrying...")
                else:
                    log.error(f"All {max_retries} attempts failed.")
                    raise

    def extract_label(self, evaluation):
        if "A" in evaluation["answer"]:
            return 1
        elif "Equal" in evaluation["answer"]:
            return 3
        elif "B" in evaluation["answer"]:
            return 5
        else:
            # TODO: This log throws an error.
            # log.info(f"Invalid prediction label: {evaluation[eval_key]}")
            # log.info(f"Invalid parsed values: {response, status}")
            return -1

    def is_quality_preserved(self, instruction, output_1, output_2, **kwargs):
        
        original = self.evaluate(instruction, output_1, output_2, **kwargs) 
        log.info(f"Original: {original}")
        followup = self.evaluate(instruction, output_2, output_1, **kwargs) # switched outputs
        log.info(f"Followup: {followup}")
        
        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)

        # TODO: This can be optimized to make a single GPT API call if the original prediction is off.

        log.info(f"Original Prediction: {original_pred}")
        log.info(f"Followup Prediction: {followup_pred}")
        
        if original_pred in [3,4,5] and followup_pred in [1,2,3]:
            quality_preserved = True
        else:
            quality_preserved = False

        original = add_prefix_to_keys(original, "original_")
        followup = add_prefix_to_keys(followup, "followup_")
        original.update({**followup})
        original.update({"quality_preserved": quality_preserved})

        log.info(f"Original Dict: {original}")
        return original

    def test(self, instruction, output_1, output_2, label, **kwargs):
        original_label = numerize_label(downscale_label(label))
        followup_label = numerize_label(downscale_label(invert_label(label)))

        original = self.evaluate_with_retries(instruction, output_1, output_2, **kwargs) 
        followup = self.evaluate_with_retries(instruction, output_2, output_1, **kwargs) # switched outputs

        original_pred = self.extract_label(original)
        followup_pred = self.extract_label(followup)

        # assign correctness points
        pred_correct = 0
        if (original_label == original_pred) and (followup_label == followup_pred):
            pred_correct = 1 # both are correct and positionally invariant
        elif (original_label == original_pred) or (followup_label == followup_pred):
            pred_correct = 0.5 # one was correct, but some positional bias was present

        # prepare output
        original = add_prefix_to_keys(original, "original_")
        followup = add_prefix_to_keys(followup, "followup_")
        original.update({
            **followup,
            "original_label": original_label,
            "followup_label": followup_label,
            "original_pred": original_pred, 
            "followup_pred": followup_pred,
            "pred_correct": pred_correct,
        })

        return original

@guidance
def produce_quality_analysis(lm, instruction, output_1, output_2):
    with user():
        lm += f"""\
        ### Task Description: 
        1. You will be comparing two responses to a given prompt to determine which response is better. 
        2. Carefully analyze both responses, considering factors such as:
            - Repetition: Does either response repeat information unnecessarily?
            - Grammar: Are there any grammatical errors, capitalization errors, or punctuation mistakes in either response?
            - Coherence: Is each response well-structured and easy to understand?
            - Relevance: Does each response directly address the given prompt?
            - Accuracy: Is the information provided in each response accurate and factual?
        3. Keep in mind that having grammatical errors, repetitions, capitalization errors, or punctuation mistakes would greatly degrade the quality of a response.

        ### The instructions prompting the responses: 
        {instruction}

        ### Response A: 
        {output_1}

        ### Response B: 
        {output_2}"""

    with assistant():
        lm += f"""
        ```json
        {{
            "analysis": "{gen('analysis', stop='"')}",
        }}```"""

    return lm

@guidance
def produce_answer(lm, analysis):
    with user():
        lm += f"""\
        ### Task Description: 
        1. Read the following analysis given by an LLM.
        2. In the "answer" field, select one of the following options based on the analysis:
            - 1 if the analysis indicates that Response A is better than Response B.
            - 2 if the analysis indicates that Response B is better than Response A.
            - 0 if the analysis indicates that Responses A and B have similar quality.

        ### Example:
        - Analysis: "Response A is more coherent and relevant than Response B."
        Expected Answer: 1

        ### Analysis:
        {analysis}"""
    
    with assistant():
        lm +=f"""
    ```json 
    {{
        "answer": {select(options=[1, 2, 0], name='answer')}"
    }}
    ```"""
        
    return lm

if __name__ == "__main__":

    @hydra.main(version_base=None, config_path="../conf", config_name="config")
    def test(cfg):
        import time
        from utils import diff
        import textwrap

        instruction = "Analyze the role of symbolism in 'To Kill a Mockingbird' and its impact on understanding the novel's themes."

        output_1 = """
        Symbolism plays a significant role in "To Kill a Mockingbird" by Harper Lee, enriching the novel's themes and adding depth to its characters and narrative. Here are some key symbols and their impact on understanding the novel's themes:
        1. **Mockingbird**: One of the central symbols in the novel, the mockingbird represents innocence and goodness. Atticus Finch advises his children that it's a sin to kill a mockingbird because they don't harm anyone; they only bring joy with their songs. This symbol is closely tied to characters like Tom Robinson, Boo Radley, and even Scout Finch herself, who are all innocent figures harmed by prejudice and ignorance.
        2. **The Radley House**: The Radley house symbolizes mystery, fear, and the unknown. It reflects the town's prejudices and gossip about Boo Radley, who becomes a scapegoat for many of Maycomb's problems. The children's fascination with Boo and their eventual understanding of his true nature contribute to the novel's themes of empathy and understanding.
        3. **The Treehouse**: The treehouse near the Radley house serves as a symbol of childhood innocence and a safe haven for Scout, Jem, and Dill. It's where they imagine Boo's life and where they confront the harsh realities of the adult world, such as racism and injustice.
        4. **The Mad Dog**: The rabid dog, Tim Johnson, symbolizes the destructive force of racism and prejudice in Maycomb. Atticus's shooting of the dog parallels his fight against the town's deeply ingrained prejudices, highlighting the novel's themes of courage and moral integrity.
        5. **The Courthouse and Jail**: These buildings symbolize the legal system and society's flawed notions of justice. Tom Robinson's trial and eventual conviction despite clear evidence of his innocence demonstrate the pervasiveness of racial prejudice and the failures of the justice system.
        6. **The Camellia Bush**: Mrs. Dubose's camellia bush symbolizes inner strength and resilience. Despite her illness and addiction, Mrs. Dubose shows courage in her battle to break free from morphine addiction, inspiring Jem and Scout to understand the complexities of human behavior.
        These symbols collectively contribute to the novel's exploration of themes such as morality, prejudice, empathy, and the loss of innocence. They provide readers with deeper insights into the characters' motivations, societal dynamics, and the moral complexities of the story, ultimately enriching our understanding of the human experience depicted in "To Kill a Mockingbird."
        """

        output_2 = """
        Symbolism plays a huge role in "To Kill a Mockingbird" by Harper Lee, enriching the novel's themes and adding depth to its characters and narrative. Here are some key symbols and their impact on understanding the novel's themes:
        1. **Mockingbird**: One of the central symbols in the novel, the mockingbird represents innocence and goodness. Atticus Finch advises his children that it's a sin to kill a mockingbird because they don't harm anyone; they only bring joy with their songs. This symbol is closely tied to characters like Tom Robinson, Boo Radley, and even Scout Finch herself, who are all innocent figures harmed by prejudice and ignorance.
        2. **The Radley House**: The Radley house symbolizes mystery, fear, and the unknown. It reflects the town's prejudices and gossip about Boo Radley, who becomes a scapegoat for many of Maycomb's problems. The children's fascination with Boo and their eventual understanding of his true nature contribute to the novel's themes of empathy and understanding.
        3. **The Treehouse**: The treehouse near the Radley house serves as a symbol of childhood innocence and a safe haven for Scout, Jem, and Dill. It's where they imagine Boo's life and where they confront the harsh realities of the adult world, such as racism and injustice.
        4. **The Mad Mad Dog Dog**: The rabid dog, Tim Johnson, symbolizes the destructive force of racism and prejudice in Maycomb. Atticus's shooting of the dog parallels his fight against the town's deeply ingrained prejudices, highlighting the novel's themes of courage and moral integrity.
        5. **The Courthouse and Jail**: These structures symbolize the legal system and society's flawed notions of justice. Tom Robinson's trial and eventual conviction despite clear evidence of his innocence demonstrate the pervasiveness of racial prejudice and the failures of the justice system.
        6. **The Camellia Bush**: Mrs. Dubose's camellia bush symbolizes inner strength and resilience. Despite her illness and addiction, Mrs. Dubose shows courage in her battle to break free from morphine addiction, inspiring Jem and Scout to understand the complexities of human behavior.
        These symbols collectively contribute to the novel's exploration of themes such as morality, prejoodice, emputhy, and the loss of innacence. They provide readers with deeper insights into the characters' motivations, societal dynamics, and the moral complexities of the story, ultimately enriching our understanding of the human experience depicted in "To Kill a Mockingbird."
        """

        label = "output_1 is much better than output_2"

        # llm = models.OpenAI("gpt-4o")
        # llm = models.Transformers(
        #     cfg.oracle_args.model_id, 
        #     echo=False,
        #     cache_dir=cfg.oracle_args.model_cache_dir, 
        #     device_map=cfg.oracle_args.device_map
        # )

        #cfg.oracle_args.model_id = "gpt-4o"
        cfg.oracle_args.model_id = "TechxGenus/Meta-Llama-3-70B-Instruct-GPTQ"


        oracle = RelativeOracle3(cfg.oracle_args, llm=None)

        for _ in range(1):
            # start = time.time()
            # evaluation = oracle.evaluate(
            #     instruction=instruction, 
            #     output_1=output_1, 
            #     output_2=output_2
            # )
            # time_taken = time.time() - start
            # print("oracle.evaluate")
            # print("evaluation:", evaluation)
            # print("time_taken:", time_taken)

            # start = time.time()
            # quality_eval = oracle.is_quality_preserved(
            #     instruction=instruction, 
            #     output_1=output_1, 
            #     output_2=output_2
            # )
            # time_taken = time.time() - start
            # print("oracle.is_quality_preserved")
            # print("quality_eval:", quality_eval)
            # print("time_taken:", time_taken)

            start = time.time()
            test_eval = oracle.test(
                instruction=instruction, 
                output_1=output_1, 
                output_2=output_2,
                label=label
            )
            time_taken = time.time() - start
            print("oracle.test")
            print("prec_correct:", test_eval["pred_correct"])
            print("time_taken:", time_taken)

    test()