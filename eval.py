from openai import OpenAI
import json
import argparse
import tqdm
import time
import os
from dotenv import load_dotenv

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--prompt_fp', type=str, default='eval/prompts/one_time.txt')
    argparser.add_argument('--save_fp', type=str, default='eval/results/one_time.json')
    argparser.add_argument('--summeval_fp', type=str, default='eval/data/mini_summeval.json')
    argparser.add_argument('--model', type=str, default='gpt-4')
    argparser.add_argument('--num_reps', type=int, default=5)
    args = argparser.parse_args()

    summeval = json.load(open(args.summeval_fp))
    prompt = open(args.prompt_fp).read()
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    num_reps=args.num_reps

    ct, ignore = 0, 0

    new_json = []
    for instance in tqdm.tqdm(summeval):
        source = instance['source']
        system_output = instance['system_output']
        cur_prompt = prompt.replace('{{Prompt}}', source).replace('{{Essay}}', system_output)
        instance['prompt'] = cur_prompt
        while True:
            try:
                _response = client.chat.completions.create(model=args.model,
                messages=[{"role": "system", "content": cur_prompt}],
                temperature=1,
                max_tokens=100,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                # logprobs=40,
                n=num_reps)
                time.sleep(0.5)

                all_responses = [_response.choices[i].message.content for i in
                                 range(len(_response.choices))]
                instance['all_responses'] = all_responses
                new_json.append(instance)
                ct += 1
                break
            except Exception as e:
                print(e)
                if ("limit" in str(e)):
                    time.sleep(2)
                else:
                    ignore += 1
                    print('ignored', ignore)

                    break

    print('ignored total', ignore)
    with open(args.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)