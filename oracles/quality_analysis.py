import guidance
from guidance import gen, select, user, assistant, system


n = "\n"

@guidance
def quality_analysis_solo_self_reward(lm, instruction, output):
		with system():
			lm += f"""\
			Review the user's question and the corresponding response using a 10-point scoring system.

			User: {instruction}
			Response: {output}

			After examining the user's instruction and the response, reply in the desired format:
			- "analysis": A string that describes the reasoning behind your scores for each answer. 
			- "score": An integer score for the assistant's answer.
                  
      Remember to assess from the AI Assistant perspective. To evaluate the response in alignment with 
			this additive scoring model, we'll systematically attribute points based on the outlined criteria.
			Please avoid any potential bias and ensuring that the order in which the responses were presented 
			does not affect your judgment.
			"""
			
		with assistant():
			lm += f"""\
			analysis: {gen('analysis', stop=n)}
			score: {select(options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name='score')}
			"""

		return lm

@guidance
def quality_analysis_solo_lmsys_ia(lm, instruction, output):
    with system():
      lm += f"""\
			[Prompt]
			{instruction}

			[The Start of Assistant's Answer]
			{output}

			[The End of Assistant's Answer]

			[System]
			We would like to request your feedback on the performance of an AI assistant in response to the user prompt displayed above.
			Please rate the helpfulness, relevance, accuracy, level of details of their responses. Rate the answer with an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
			Respond in the desired JSON format:
			- "analysis": A string that describes the reasoning behind your score. 
			- "score": An integer score for the assistant's answer.
      """
    with assistant():
      lm += f"""\
			analysis: {gen('analysis', stop=n)}
			score: {select(options=[1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10], name='score')}

			Please avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""

    return lm

@guidance
def quality_analysis_solo_lmsys_ib(lm, instruction, output):
    with system():
      lm += f"""\
			[System]
			We would like to request your feedback on the performance of an AI assistant in response to the user prompt displayed below.
			Please rate the helpfulness, relevance, accuracy, level of details of their responses. Rate the answer with an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
			Respond in the desired JSON format:
			- "analysis": A string that describes the reasoning behind your score. 
			- "score": An integer score for the assistant's answer.

			[Prompt]
			{instruction}

			[The Start of Assistant's Answer]
			{output}

			[The End of Assistant's Answer]
			"""
    
    with assistant():
      lm += f"""\
			analysis: {gen('analysis', stop=n)}
			score: {select(options=[1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10], name='score')}
	
			Please avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""

    return lm


@guidance
def quality_analysis_relative_3(lm, instruction, output_1, output_2):
    with system():
      lm += f"""\
			Below are two candidate responses to the prompt: 
			{instruction}

			Response A: 
			{output_1}

			Response B:
			{output_2}

			Compare which of the two above responses is a better response to the given prompt. 

			Respond in the desired format:
			- analysis: The reasoning behind your answer step by step. Your evaluation should consider factors such as the repetition, grammar, coherence, relevance, accuracy of the responses. Especially, note that having grammatical errors, repetitions, capitalization errors or punctuation mistakes would greatly degrade the quality of a response.
			- answer: One of the following three options:
					(1) Response A is better than response B
					(2) Responses A and B have similar quality
					(3) Response B is better than response A
			"""
      
    with assistant():
      lm += f"""\
			analysis: {gen('analysis', stop=n)} 
			answer: {select(options=[1, 2, 3], name='answer')}

			Please avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""

    return lm


@guidance
def quality_analysis_relative_5(lm, instruction, output_1, output_2):
    with system():
      lm += f"""\
			Below are two candidate responses to the prompt: 
			{instruction}

			Response A: 
			{output_1}

			Response B:
			{output_2}

			Compare which of the two above responses is a better response to the given prompt. 

			Respond in the desired format:
			- analysis: The reasoning behind your answer step by step. Your evaluation should consider factors such as the repetition, grammar, coherence, relevance, accuracy of the responses. Especially, note that having grammatical errors, repetitions, capitalization errors or punctuation mistakes would greatly degrade the quality of a response.
			- answer: One of the following five options:
					(1) Response A is much better than response B
					(2) Response A is a little better than response B
					(3) Responses A and B have similar quality
					(4) Response B is a little better than response A
					(5) Response B is much better than response A
			"""

    with assistant():
      lm += f"""\
									
			analysis: {gen('analysis', stop=n)} 
			answer: {select(options=[1, 2, 3, 4, 5], name='answer')}

			Please avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""

    return lm


@guidance
def quality_analysis_joint_ia(lm, instruction, output_1, output_2):
    with system():
        
      lm += f"""\
			[System]
			We would like to request your feedback on the performance of two AI assistants in response to the user prompt displayed below.
			Please rate the grammatical correctness, fluency, accuracy, consistency, and clarity. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
			
			Respond in the desired format:
			- analysis: A description of the reasoning behind your scores for each answer. 
			- assistan_1_score: An integer score for assistant 1's answer.
			- assistan_2_score: An integer score for assistant 2's answer.
									
			[Prompt]
			{instruction}

			[The Start of Assistant 1's Answer]
			{output_1}

			[The End of Assistant 1's Answer]

			[The Start of Assistant 2's Answer]
			{output_2}

			[The End of Assistant 2's Answer]
			"""
    
    with assistant():
      lm += f"""\
			analysis: {gen('analysis', stop=n)}
			assistant_1_score: {select(options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name='assistant_1_score')}
			assistant_2_score: {select(options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name='assistant_2_score')}

			Please avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""

    return lm



@guidance
def quality_analysis_joint_ib(lm, instruction, output_1, output_2):
    with system():
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
			We would like to request your feedback on the performance of two AI assistants in response to the user prompt displayed below.
			Please rate the grammatical correctness, fluency, accuracy, consistency, and clarity. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
			
			Respond in the desired format:
			- analysis: A description of the reasoning behind your scores for each answer. 
			- assistan_1_score: An integer score for assistant 1's answer.
			- assistan_2_score: An integer score for assistant 2's answer.
			"""
    
    with assistant():
      lm += f"""\
			analysis: {gen('analysis', stop=n)}
			assistant_1_score: {select(options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name='assistant_1_score')}
			assistant_2_score: {select(options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name='assistant_2_score')}

			Please avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."""

    return lm


# @guidance
# def quality_analysis_rank(lm, instruction, output_1, output_2):
#     dictionary = r'\{\s*([\'"])[^\'"]*\1\s*:\s*\d+\s*(,\s*([\'"])[^\'"]*\3\s*:\s*\d+\s*)*\}'
#     lm += f"""\
#     I want you to create a leaderboard of different of large language models. To do so, I will give you the prompt given to the models, and the responses of two models. To make a leaderboard, first make a list ranking the models based on which responses would be preferred by humans, then return the ranking in the desired JSON format. The JSON structure for model ranking analysis should include the following fields:
# 		- "analysis": A string that describes the reasoning behind the ranking of the models. 
# 		- "ranking": An object where each key is the name of a model (string) and its value is the ranking (integer). The ranking represents the model's position or score relative to other models, where lower numbers indicate a higher ranking.

# 		```json
# 		{{
# 				"analysis": "{gen('analysis', stop='"')}", 
# 				"ranking": "{gen('ranking', regex=dictionary)}"
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

@guidance
def quality_analysis_rank(lm, instruction, output_1, output_2):
    with system():
      lm += f"""\
			Create a leaderboard of different of large language models. You will be given the prompt given to the models, and the responses of two models. To make a leaderboard, first make a list ranking the models based on which responses would be preferred by humans, then return the ranking in the desired format.
			- analysis: A string that describes the reasoning behind the ranking of the models. 
			- model_1_ranking: The ranking for model_1. 
      - model_2_ranking: The ranking for model_2.
									
			Here is the prompt:
			{instruction}

			Here are the outputs of the models:
			model: model_1
			response: {output_1}
			
			model: model_2
			response: {output_2}
			
			Now make the leaderboard by ranking the models by the quality of their responses, so that the model with rank 1 has the best output. 
			Please avoid any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.
			"""
    
    with assistant():
      lm += f"""\
			analysis: {gen('analysis', stop=n)}
			model_1_ranking: {select(options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name='model_1_ranking')} 
			model_2_ranking: {select(options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name='model_2_ranking')} 
			"""

    return lm