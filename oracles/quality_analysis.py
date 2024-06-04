import guidance
from guidance import gen, select

@guidance
def quality_analysis_solo_self_reward(lm, instruction, output):
    lm += f"""\
    Review the user's question and the corresponding response using the additive 5-point
    scoring system described below. Points are accumulated based on the satisfaction of each
    criterion:
    - Add 1 point if the response is relevant and provides some information related to
    the user's prompt, even if it is incomplete or contains some irrelevant content.
    - Add another point if the response addresses a substantial portion of the user's question,
    but does not completely resolve the query or provide a direct answer.
    - Award a third point if the response answers the basic elements of the user's question in a
    useful way, regardless of whether it seems to have been written by an AI Assistant or if it
    has elements typically found in blogs or search results.
    - Grant a fourth point if the response is clearly written from an AI Assistant's perspective,
    addressing the user's question directly and comprehensively, and is well-organized and
    helpful, even if there is slight room for improvement in clarity, conciseness or focus.
    - Bestow a fifth point for a response that is impeccably tailored to the user's question
    by an AI Assistant, without extraneous information, reflecting expert knowledge, and
    demonstrating a high-quality, engaging, and insightful answer.

    User: {instruction}
    Response: {output}

    After examining the user's instruction and the response, reply in the desired JSON format:
    - "analysis": A string that describes the reasoning behind your scores for each answer. 
    - "score": An integer score for assistant 1's answer.

    ```json 
    {{
        "analysis": "{gen('analysis', stop='"')}", 
        "score": "{select(options=[1, 2, 3, 4, 5, 0], name='score')}"
    }}
    ```
        
    Remember to assess from the AI Assistant perspective. To evaluate the response in alignment with 
    this additive scoring model, we'll systematically attribute points based on the outlined criteria.
    Please avoid any potential bias and ensuring that the order in which the responses were presented 
    does not affect your judgment.
    """

    return lm

@guidance
def quality_analysis_solo_lmsys_ia(lm, instruction, output):
    lm += f"""\
    [Prompt]
		{instruction}

		[The Start of Assistant's Answer]
		{output}

		[The End of Assistant's Answer]

		[System]
		We would like to request your feedback on the performance of an AI assistant in response to the user prompt displayed above.
		Please rate the helpfulness, relevance, accuracy, level of details of their responses. Rate the answer with an overall score on a scale of 1 to 5, where a higher score indicates better overall performance.
		Respond in the desired JSON format:
		- "analysis": A string that describes the reasoning behind your score. 
		- "score": An integer score for the assistant's answer.

		```json 
    {{
        "analysis": "{gen('analysis', stop='"')}", 
        "score": "{select(options=[1, 2, 3, 4, 5, 0], name='score')}"
    }}
    ```

		Please avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""

    return lm

@guidance
def quality_analysis_solo_lmsys_ib(lm, instruction, output):
    lm += f"""\
    [System]
		We would like to request your feedback on the performance of an AI assistant in response to the user prompt displayed below.
		Please rate the helpfulness, relevance, accuracy, level of details of their responses. Rate the answer with an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
		Respond in the desired JSON format:
		- "analysis": A string that describes the reasoning behind your score. 
		- "score": An integer score for the assistant's answer.

		```json 
    {{
        "analysis": "{gen('analysis', stop='"')}", 
        "score": "{select(options=[1, 2, 3, 4, 5, 0], name='score')}"
    }}
    ```
    
		[Prompt]
		{instruction}

		[The Start of Assistant's Answer]
		{output}

		[The End of Assistant's Answer]

		Please avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""

    return lm


@guidance
def quality_analysis_relative_3(lm, instruction, output_1, output_2):
    lm += f"""\
    Below are two candidate responses to the prompt: 
		{instruction}

		Response A: 
		{output_1}

		Response B:
		{output_2}

		Compare which of the two above responses is a better response to the given prompt. 

		Respond in the desired JSON format:
		- "analysis": A string that describes the reasoning behind your answer step by step. Your evaluation should consider factors such as the repetition, grammar, coherence, relevance, accuracy of the responses. Especially, note that having grammatical errors, repetitions, capitalization errors or punctuation mistakes would greatly degrade the quality of a response.
		- "answer": A string chosen from the following three options:
				(1) Response A is better than response B
				(2) Responses A and B have similar quality
				(3) Response B is better than response A

		Response schema: 
		```json 
		{{
				"analysis": "{gen('analysis', stop='"')}", 
				"answer": "{gen('answer', stop='"')}"
		}}
		``` 

		Please avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""

    return lm


@guidance
def quality_analysis_relative_5(lm, instruction, output_1, output_2):
    lm += f"""\
    Below are two candidate responses to the prompt: 
		{instruction}

		Response A: 
		{output_1}

		Response B:
		{output_2}

		Compare which of the two above responses is a better response to the given prompt. 

		Respond in the desired JSON format:
		- "analysis": A string that describes the reasoning behind your answer step by step. Your evaluation should consider factors such as the repetition, grammar, coherence, relevance, accuracy of the responses. Especially, note that having grammatical errors, repetitions, capitalization errors or punctuation mistakes would greatly degrade the quality of a response.
		- "answer": A string chosen from the following five options:
				(1) Response A is much better than response B
				(2) Response A is a little better than response B
				(3) Responses A and B have similar quality
				(4) Response B is a little better than response A
				(5) Response B is much better than response A

		Response schema: 
		```json 
		{{
				"analysis": "{gen('analysis', stop='"')}", 
				"answer": "{gen('answer', stop='"')}"
		}}
		``` 

		Please avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""

    return lm


@guidance
def quality_analysis_joint_ia(lm, instruction, output_1, output_2):
    lm += f"""\
    [System]
		We would like to request your feedback on the performance of two AI assistants in response to the user prompt displayed below.
		Please rate the grammatical correctness, fluency, accuracy, consistency, and clarity. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
		Respond in the desired JSON format:
		- "analysis": A string that describes the reasoning behind your scores for each answer. 
		- "assistan_1_score": An integer score for assistant 1's answer.
		- "assistan_2_score": An integer score for assistant 2's answer.

		```json 
		{{
				"analysis": "{gen('analysis', stop='"')}", 
				"assistant_1_score": "{gen('assistant_1_score', regex='[1-9]|10')}", 
				"assistant_2_score": "{gen('assistant_2_score', regex='[1-9]|10')}"
		}}
		``` 
		[Prompt]
		{instruction}

		[The Start of Assistant 1's Answer]
		{output_1}

		[The End of Assistant 1's Answer]

		[The Start of Assistant 2's Answer]
		{output_2}

		[The End of Assistant 2's Answer]

		Please avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""

    return lm



@guidance
def quality_analysis_joint_ia(lm, instruction, output_1, output_2):
    lm += f"""\
    [Prompt]
		{instruction}

		[The Start of Assistant 1's Answer]
		{output_1}

		[The End of Assistant 1's Answer]

		[The Start of Assistant 2's Answer]
		{output_2}

		[The End of Assistant 2's Answer]

		[System]
		We would like to request your feedback on the performance of two AI assistants in response to the user prompt displayed above.
		Please rate the grammatical correctness, fluency, accuracy, consistency, and clarity. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
		Respond in the desired JSON format:
		- "analysis": A string that describes the reasoning behind your scores for each answer. 
		- "assistant_1_score": An integer score for assistant 1's answer.
		- "assistant_2_score": An integer score for assistant 2's answer.

		```json 
		{{
				"analysis": "{gen('analysis', stop='"')}", 
				"assistant_1_score": "{gen('assistant_1_score', regex='[1-9]|10')}", 
				"assistant_2_score": "{gen('assistant_2_score', regex='[1-9]|10')}"
		}}
		``` 

		Please avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""

    return lm

# TODO: Backslashes break it. The code wasn't running. See 06_03_backslash_error.log - Boran
# @guidance
# def quality_analysis_rank(lm, instruction, output_1, output_2):
#     lm += f"""
#     I want you to create a leaderboard of different of large language models. To do so, I will give you the prompt given to the models, and the responses of two models. To make a leaderboard, first make a list ranking the models based on which responses would be preferred by humans, then return the ranking in the desired JSON format. The JSON structure for model ranking analysis should include the following fields:
# 		- "analysis": A string that describes the reasoning behind the ranking of the models. 
# 		- "ranking": An object where each key is the name of a model (string) and its value is the ranking (integer). The ranking represents the model's position or score relative to other models, where lower numbers indicate a higher ranking.

# 		```json 
# 		{{
# 				"analysis": "{gen('analysis', stop='"')}", 
# 				"ranking": "{gen('ranking', regex='\{\s*([\'"])[^\'"]*\1\s*:\s*\d+\s*(,\s*([\'"])[^\'"]*\3\s*:\s*\d+\s*)*\}')}"
# 		}}
# 		``` 
								
# 		Here is the prompt:
# 		{{
# 				"instruction": "{instruction}",
# 		}}

# 		Here are the outputs of the models:
# 		[
# 				{{
# 						"model": "model_1",
# 						"response": "{output_1}"
# 				}},
# 				{{
# 						"model": "model_2",
# 						"response": "{output_2}"
# 				}}
# 		]

# 		Now make the leaderboard by ranking the models by the quality of their responses, so that the model with rank 1 has the best output. 
# 		Please avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""

#     return lm