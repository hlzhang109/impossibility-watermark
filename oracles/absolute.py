# Absolute Grading: Outputs score of 1 to 5

from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import RELATIVE_PROMPT, ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
import warnings
from .custom import Oracle

class PrometheusAbsoluteOracle(Oracle):
    def __init__(
        self, 
        cfg,
        model_id="prometheus-eval/prometheus-8x7b-v2.0",
        download_dir="/data2/.shared_models",
        num_gpus=4, 
    ):
        self.cfg=cfg
        super().__init__(cfg)
        # Initialize any necessary attributes or models here
        self.model_id = model_id
        self.download_dir = download_dir
        self.num_gpus = num_gpus

        # Defining rubric criteria
        self.rubric_data = {
            "criteria": "Determine which response best addresses the instructions. Your evaluation should consider factors such as repetition, grammar, coherence, relevance, and accuracy of the responses. Note that having grammatical errors, repetitions, capitalization errors, or punctuation mistakes would greatly degrade the quality of a response.",
            "score1_description": "The response fails to address the instructions, with significant issues in grammar, coherence, relevance, and accuracy. The response contains multiple grammatical errors, repetitions, and capitalization or punctuation mistakes, greatly degrading its quality.",
            "score2_description": "The response partially addresses the instructions but has notable issues in grammar, coherence, relevance, and accuracy. There are several grammatical errors, repetitions, and capitalization or punctuation mistakes that detract from its quality.",
            "score3_description": "The response generally addresses the instructions with moderate issues in grammar, coherence, relevance, and accuracy. Some grammatical errors, repetitions, and capitalization or punctuation mistakes are present but do not severely impact the overall quality.",
            "score4_description": "The response effectively addresses the instructions with minor issues in grammar, coherence, relevance, and accuracy. Few grammatical errors, repetitions, and capitalization or punctuation mistakes are present and do not significantly affect the quality.",
            "score5_description": "The response excellently addresses the instructions with high standards of grammar, coherence, relevance, and accuracy. It is free from grammatical errors, repetitions, and capitalization or punctuation mistakes, demonstrating superior quality."
            }
        self.rubric = SCORE_RUBRIC_TEMPLATE.format(**self.rubric_data)

        # Loading model
        if "8x7b" in self.model_id and self.num_gpus < 4:
            self.num_gpus = 4
            warnings.warn(
                f"`prometheus-8x7b-v2.0` requires ~172GB of GPU RAM. Increasing num_gpus from {self.num_gpus} to 4."
            )
        self.load_judge()
    
    def load_judge(self):
        # Load or initialize the model used for scoring and feedback
        if Oracle.judge == None:
            self.judge = PrometheusEval(
								model_id=self.model_id, 
								download_dir=self.download_dir,
								num_gpus=self.num_gpus, 
								absolute_grade_template=ABSOLUTE_PROMPT, 
								relative_grade_template=RELATIVE_PROMPT, 
						)

    def evaluate(self, instruction, response, reference_answer=None):
        feedback, score = self.judge.single_absolute_grade(
            rubric=self.rubric,
            instruction=instruction,
            response=response,
            reference_answer=reference_answer
        )
        return feedback, score

    def evaluate_batch(self, instructions, responses, reference_answers=None):
        feedbacks, scores = self.judge.absolute_grade(
            rubric=self.rubric,
            instructions=instructions,
            responses=responses,
            reference_answers=reference_answers,
            params={},
        )
        return feedbacks, scores

    def is_quality_preserved(self, instruction, original_text, mutated_text, reference_answer=None):
        # Prepare evaluation
        instructions = [instruction] * 2
        responses = [original_text, mutated_text]
        reference_answers = [reference_answer] * 2 if reference_answer else [None] * 2
        feedbacks, scores = self.evaluate_batch(instructions, responses, reference_answers)

        # Determine if quality is preserved by comparing scores
        if scores[1] >= scores[0]:
            quality_preserved = True
        else:
            quality_preserved = False

        # Package results for output
        quality_eval =  {
            "feedbacks": feedbacks,
            "scores": scores,
            "quality_preserved": quality_preserved
        }

        return quality_eval
    
    def extract_label(self, evaluation):
        return evaluation[1]

    def derive_label(self, output_1_score, output_2_score):
        diff = output_1_score - output_2_score 
        pred = -1
        if 2 <= diff <= 5: # output 1 is better than output_2
            pred =  1
        elif -2 < diff < 2:  # output 1 is about the same as output_2
            pred = 3
        elif -5 <= diff <= -2:  # output 1 is worse than output_2
            pred = 2
        return pred
    
    def test(self, instruction, output_1, output_2, label, **kwargs):
        output_1_evaluation = self.evaluate(instruction, output_1)
        output_2_evaluation = self.evaluate(instruction, output_2)
        
        output_1_score = self.extract_label(output_1_evaluation)
        output_2_score = self.extract_label(output_2_evaluation)

        pred = self.derive_label(output_1_score, output_2_score)
        
				# assign correctness points
        pred_correct = 0
        if (label == pred):
            pred_correct = 1 
            
        results = {
            "output_1_feedback": output_1_evaluation[0],
            "output_1_score": output_1_score,  
            "output_2_feedback": output_2_evaluation[0],
            "output_2_score": output_2_score,  
						"label": label,
						"pred": pred,
            "pred_correct": pred_correct,
				}
				
        return results
        
import hydra
import logging

from .custom import SoloOracle, RankOracle, JointOracle, RelativeOracle
from .absolute import PrometheusAbsoluteOracle
from .relative import PrometheusRelativeOracle

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('optimum.gptq.quantizer').setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def test(cfg):
    instruction = "Analyze the role of symbolism in 'To Kill a Mockingbird' and its impact on understanding the novel's themes."

    original_text = """
    Symbolism plays a significant role in "To Kill a Mockingbird" by Harper Lee, enriching the novel's themes and adding depth to its characters and narrative. Here are some key symbols and their impact on understanding the novel's themes:

    1. **Mockingbird**: One of the central symbols in the novel, the mockingbird represents innocence and goodness. Atticus Finch advises his children that it's a sin to kill a mockingbird because they don't harm anyone; they only bring joy with their songs. This symbol is closely tied to characters like Tom Robinson, Boo Radley, and even Scout Finch herself, who are all innocent figures harmed by prejudice and ignorance.

    2. **The Radley House**: The Radley house symbolizes mystery, fear, and the unknown. It reflects the town's prejudices and gossip about Boo Radley, who becomes a scapegoat for many of Maycomb's problems. The children's fascination with Boo and their eventual understanding of his true nature contribute to the novel's themes of empathy and understanding.

    3. **The Treehouse**: The treehouse near the Radley house serves as a symbol of childhood innocence and a safe haven for Scout, Jem, and Dill. It's where they imagine Boo's life and where they confront the harsh realities of the adult world, such as racism and injustice.

    4. **The Mad Dog**: The rabid dog, Tim Johnson, symbolizes the destructive force of racism and prejudice in Maycomb. Atticus's shooting of the dog parallels his fight against the town's deeply ingrained prejudices, highlighting the novel's themes of courage and moral integrity.

    5. **The Courthouse and Jail**: These buildings symbolize the legal system and society's flawed notions of justice. Tom Robinson's trial and eventual conviction despite clear evidence of his innocence demonstrate the pervasiveness of racial prejudice and the failures of the justice system.

    6. **The Camellia Bush**: Mrs. Dubose's camellia bush symbolizes inner strength and resilience. Despite her illness and addiction, Mrs. Dubose shows courage in her battle to break free from morphine addiction, inspiring Jem and Scout to understand the complexities of human behavior.

    These symbols collectively contribute to the novel's exploration of themes such as morality, prejudice, empathy, and the loss of innocence. They provide readers with deeper insights into the characters' motivations, societal dynamics, and the moral complexities of the story, ultimately enriching our understanding of the human experience depicted in "To Kill a Mockingbird."
    """

    mutated_text = """
    Symbolism plays a huge role in "To Kill a Mockingbird" by Harper Lee, enriching the novel's themes and adding depth to its characters and narrative. Here are some key symbols and their impact on understanding the novel's themes:

    1. **Mockingbird**: One of the central symbols in the novel, the mockingbird represents innocence and goodness. Atticus Finch advises his children that it's a sin to kill a mockingbird because they don't harm anyone; they only bring joy with their songs. This symbol is closely tied to characters like Tom Robinson, Boo Radley, and even Scout Finch herself, who are all innocent figures harmed by prejudice and ignorance.

    2. **The Radley House**: The Radley house symbolizes mystery, fear, and the unknown. It reflects the town's prejudices and gossip about Boo Radley, who becomes a scapegoat for many of Maycomb's problems. The children's fascination with Boo and their eventual understanding of his true nature contribute to the novel's themes of empathy and understanding.

    3. **The Treehouse**: The treehouse near the Radley house serves as a symbol of childhood innocence and a safe haven for Scout, Jem, and Dill. It's where they imagine Boo's life and where they confront the harsh realities of the adult world, such as racism and injustice.

    4. **The Mad Dog**: The rabid dog, Tim Johnson, symbolizes the destructive force of racism and prejudice in Maycomb. Atticus's shooting of the dog parallels his fight against the town's deeply ingrained prejudices, highlighting the novel's themes of courage and moral integrity.

    5. **The Courthouse and Jail**: These structures symbolize the legal system and society's flawed notions of justice. Tom Robinson's trial and eventual conviction despite clear evidence of his innocence demonstrate the pervasiveness of racial prejudice and the failures of the justice system.

    6. **The Camellia Bush**: Mrs. Dubose's camellia bush symbolizes inner strength and resilience. Despite her illness and addiction, Mrs. Dubose shows courage in her battle to break free from morphine addiction, inspiring Jem and Scout to understand the complexities of human behavior.

    These symbols collectively contribute to the novel's exploration of themes such as morality, prejoodice, empathy, and the loss of innocence. They provide readers with deeper insights into the characters' motivations, societal dynamics, and the moral complexities of the story, ultimately enriching our understanding of the human experience depicted in "To Kill a Mockingbird."
    """

    reference_answer= """
    Symbolism plays a crucial role in "To Kill a Mockingbird," enriching the narrative and deepening the exploration of its themes. Harper Lee employs various symbols to convey complex ideas about morality, justice, and human behavior. Here are some key symbols and their impacts on understanding the novel's themes:

    ### The Mockingbird
    The title of the novel itself is a significant symbol. The mockingbird represents innocence and the idea of doing no harm. In the story, Atticus Finch tells his children that killing a mockingbird is a sin because they do nothing but make music for people to enjoy. This symbolism is directly tied to characters like Tom Robinson and Boo Radley, who are innocent but suffer harm due to prejudice and ignorance.

    - **Tom Robinson**: As a black man falsely accused of raping a white woman, Tom embodies the mockingbird's innocence destroyed by evil. His trial and subsequent death highlight the themes of racial injustice and the failure of the legal system.
    - **Boo Radley**: Boo is a reclusive neighbor who becomes a figure of fear and mystery for the children but ultimately reveals his kindness and protects them. Boo's transformation in the eyes of Scout and Jem from a monster to a human being underscores the theme of empathy and the importance of seeing others for who they truly are.

    ### The Radley House
    The Radley house is another important symbol in the novel. It represents fear, mystery, and the unknown. Initially, the house and its inhabitants are subjects of town gossip and superstition, symbolizing how fear and ignorance can lead to prejudice and misunderstanding. As Scout and Jem's understanding of Boo Radley evolves, the house's symbolism shifts, reflecting their growth and the novel's emphasis on overcoming fear with knowledge and compassion.

    ### The Camellias
    Mrs. Dubose's camellias symbolize forgiveness and the complexity of human morality. Despite her racist remarks and abrasive personality, Mrs. Dubose's struggle to overcome her morphine addiction reveals her strength and determination. Jem's act of destroying the camellias out of anger and his subsequent punishment, which involves reading to Mrs. Dubose, teaches him about forgiveness, understanding, and the moral courage to fight personal battles.

    ### The Mad Dog
    The mad dog that Atticus is forced to shoot symbolizes the presence of danger and the irrationality of racism in Maycomb. The dog's rabid state represents the destructive nature of prejudice, which, like the disease, spreads and endangers the community. Atticus's role in shooting the dog underscores his position as a moral guardian, attempting to protect the town from its own deep-seated fears and hatreds.

    ### The Knot-Hole
    The knot-hole in the Radley tree, where Boo leaves gifts for Scout and Jem, symbolizes communication and friendship across societal boundaries. The children's discovery of the gifts and their eventual realization that Boo is the giver helps to break down the barriers of fear and prejudice. When Nathan Radley fills the hole with cement, it represents the ways in which society often tries to stifle understanding and maintain divisions.

    ### Courtroom
    The courthouse and the trial symbolize the struggle for justice within a flawed system. The courtroom serves as a stage where the themes of racial inequality, moral courage, and the quest for justice are prominently displayed. Atticus's defense of Tom Robinson, despite knowing the odds are against them, highlights his commitment to justice and moral integrity.

    ### Impact on Understanding Themes
    Through these symbols, Harper Lee effectively conveys the central themes of the novel:

    - **Racial Injustice**: The symbols of the mockingbird and the mad dog illustrate the pervasive and destructive nature of racism.
    - **Moral Growth and Integrity**: Characters like Atticus and the symbolism of the courtroom and camellias highlight the importance of standing up for what is right, even in the face of adversity.
    - **Empathy and Understanding**: The evolution of Scout and Jem's perception of Boo Radley, symbolized by the Radley house and the knot-hole, emphasizes the importance of seeing the world from others' perspectives and fostering empathy.

    Overall, symbolism in "To Kill a Mockingbird" serves as a powerful tool to deepen readers' understanding of its complex themes, enriching the narrative and leaving a lasting impact on its audience.
    """

    oracle = PrometheusRelativeOracle(cfg)
    oracle2 = PrometheusRelativeOracle(cfg)

    # quality_eval = oracle.is_quality_preserved(
    #     instruction=instruction, 
    #     original_text=original_text, 
    #     mutated_text=mutated_text, 
    #     reference_answer=None
    # )
    # print("EVAL oracle.is_quality_preserved")
    # print("quality_eval:", quality_eval)

    # feedback, score = oracle.evaluate(instruction, mutated_text, original_text)
    # print("Evaluation WITH Reference Answer")
    # print("Feedback:", feedback)
    # print("Score:", score)

    # print("Evaluation WITHOUT Reference Answer")
    # feedback, score = oracle.evaluate(instruction, mutated_text, None)
    # print("Feedback:", feedback)
    # print("Score:", score)
    
    print("Test Prometheus Absolute Oracle:")
    results = oracle.test(instruction,original_text,mutated_text, 1)
    print(results)

    


# Testing
if __name__ == "__main__":
    test()
    