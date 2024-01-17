import os, sys, argparse, time

import numpy as np
from transformers import AutoTokenizer
from mersenne import mersenne_rng

import pyximport
pyximport.install(reload_support=True, language_level=sys.version_info[0],
                  setup_args={'include_dirs':np.get_include()})
from levenshtein import levenshtein

def permutation_test(tokens,key,n,k,vocab_size,n_runs=100):
    rng = mersenne_rng(key)
    xi = np.array([rng.rand() for _ in range(n*vocab_size)], dtype=np.float32).reshape(n,vocab_size)
    test_result = detect(tokens,n,k,xi)

    p_val = 0
    for run in range(n_runs):
        xi_alternative = np.random.rand(n, vocab_size).astype(np.float32)
        null_result = detect(tokens,n,k,xi_alternative)

        # assuming lower test values indicate presence of watermark
        p_val += null_result <= test_result

    return (p_val+1.0)/(n_runs+1.0)


def detect(tokens,n,k,xi,gamma=0.0):
    m = len(tokens)
    n = len(xi)

    A = np.empty((m-(k-1),n))
    for i in range(m-(k-1)):
        for j in range(n):
            A[i][j] = levenshtein(tokens[i:i+k],xi[(j+np.arange(k))%n],gamma)

    return np.min(A)


def main(args):
    # with open(args.document, 'r') as f:
    #     text = f.read()
        
    text = """Write a 250 word essay on the role of power and its impact on characters in the Lord of the Rings series. How does the ring symbolize power, and what does Tolkien suggest about the nature of power?
Power plays a significant role in J.R.R. Tolkien's The Lord of the Rings series as it affects the lives of many characters. The ring in particular, known as the One Ring, symbolizes power and its impact on those who possess it. The ring was created by the Dark Lord Sauron, and it holds the power to control and enslave the wills of others. The characters who come into contact with the ring, such as Frodo, Boromir, and Gollum, are drawn to its power and struggle with its corrupting influence.
Tolkien suggests that power can be both corrupting and seductive, and that it can ultimately lead to the downfall of even the strongest characters. Frodo, who is initially reluctant to take on the burden of the ring, becomes increasingly consumed by its power as he travels towards Mount Doom to destroy it. Boromir, initially motivated by his desire to protect his homeland, becomes corrupted by the ring's power and ultimately betrays his friends. Gollum, who has been obsessed with the ring for centuries, is driven to madness and desperation by his constant craving for it.
The ring's power is also symbolic of the dangers of unchecked ambition and the corrupting influence of desire. Tolkien illustrates that the unchecked pursuit of power can lead to gross manipulation, killing, destruction and suffering as seen in arms of the dark lord Sauron. He also shows that the desire for power can lead to dependency on an object (the ring) that ultimately leads to the downfall of the characters who are obsessed with it.
In conclusion, Tolkien's Lord of the Rings series highlights the significance of power and its effect on even the most moral and noble of characters. The series shows that power can be unpredictable and dangerous, and that it may have terrible repercussions, as the characters that lusted for power found out. The ring, in particular, symbolizes the danger of power and its seductive appeal, and serves as a warning about the kinds of evil it can invite and the kinds of people it can corrupt. In the end, the novel emphasizes the value of heroism, friendship, and the fight against an unjust system created by those who seek power for selfish reasons."""

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048).numpy()[0]
    
    t0 = time.time()
    pval = permutation_test(tokens,args.key,args.n,len(tokens),len(tokenizer))
    print('p-value: ', pval)
    print(f'(elapsed time: {time.time()-t0}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test for a watermark in a text document')
    parser.add_argument('document',type=str, help='a file containing the document to test')
    parser.add_argument('--tokenizer',default='TheBloke/Llama-2-7b-Chat-GPTQ',type=str,
            help='a HuggingFace model id of the tokenizer used by the watermarked model')
    parser.add_argument('--n',default=256,type=int,
            help='the length of the watermark sequence')
    parser.add_argument('--key',default=42,type=int,
            help='the seed for the watermark sequence')

    main(parser.parse_args())