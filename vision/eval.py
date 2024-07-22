import openai
import re
import os
import abc
import base64
from tqdm import tqdm

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class QualityOracle():
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def maintain_quality(self, wtmk_img, attack_img):
        pass

class GPT4OQualityOracle(QualityOracle):
    def __init__(self, choice_granularity: int in [3,5], max_attempts = 20) -> None:
        super().__init__()
        self.max_attempts = max_attempts
        self.choice_granularity = choice_granularity
        self.score_dict = {1: 1, 2: 0.5, 3: 0, 4: -0.5, 5:-1} if self.choice_granularity == 5 else {1: 1, 2: 0, 3: -1} 
        self.choice = self.three_choice if choice_granularity == 3 else self.five_choice

    @property
    def comparison_prompt(self):
        return '''
        Compare which of the two above figures is a better response of higher-quality to the given prompt. 
        Explain your reasoning step by step. 
        '''

    @property
    def five_choice(self):
        return '''
        You must choose one of the following choices:
        (1) Response A is much better than response B
        (2) Response A is a little better than response B
        (3) Responses A and B have similar quality
        (4) Response B is a little better than response A
        (5) Response B is much better than response A
        '''

    @property
    def three_choice(self):
        return '''
        You must choose one of the following choices:
        (1) Response A is better than response B
        (3) Responses A and B have similar quality
        (5) Response B is better than response A
        '''
    
    def gpt4_judge_quality(self, prompt, wtmk_img, attack_img):
        client = openai.OpenAI()
        response = client.chat.completions.create(
        model="gpt-4o", #"gpt-4-vision-preview",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{wtmk_img}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{attack_img}"
                }
                },
                {
                "type": "text",
                "text": self.comparison_prompt + "\n" + self.choice
                },
            ],
            }
        ],
        max_tokens=1000,
        )
        response_content = response.choices[0].message.content
        print(response_content)
        pattern = r'\((\d+)\)'
        filtered_response = re.findall(pattern, response_content)
        return filtered_response

    def maintain_quality(self, prompt, wtmk_img, attack_img):
        wtmk_img_str = encode_image(wtmk_img)
        attack_img_str = encode_image(attack_img)
        n_attempt = 0
        while n_attempt < self.max_attempts:
            try:
                filtered_response = self.gpt4_judge_quality(prompt, wtmk_img_str, attack_img_str)
                choice = int(filtered_response[-1])
                if choice not in range(1, self.choice_granularity+1):
                    print(f"return choice {choice} not in [1, {self.choice_granularity}]")
                    continue
                else:
                    score = self.score_dict[choice]
                return score
            except Exception as e:
                print(e)
                n_attempt += 1
                continue
        print(f"Failed to get a response after {self.max_attempts} attempts")
        return False

if __name__ == "__main__":
    openai.api_key = os.getenv('OPENAI_API_KEY')
    oracle = GPT4OQualityOracle(choice_granularity=5)

    image_folder = 'imgs'
    prompt_folder = 'prompts'
    scheme = 'invisible-watermark'
    image_filenames = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('original.png')]
    with open(os.path.join(prompt_folder, 'prompts.txt'), 'r') as f:
        prompts = f.readlines()
        
    for idx in tqdm(range(10)):
        wtmk_img = f'./imgs/{scheme}/{idx}-original.png'
        attack_img = f'./results/{scheme}/{idx}-attack.png' 
        prompt = prompts[idx]

        print("First round of comparison")
        first_score = oracle.maintain_quality(prompt, wtmk_img, attack_img)
        print()
        print("Second round of comparison")
        second_score = oracle.maintain_quality(prompt, attack_img, wtmk_img)
        print()

        print(f"Score: {first_score}, {second_score}")
        print(f"Quality degradation? First judge: {first_score > 0}, Second judge: {not second_score > 0}")