import argparse
import os

def get_cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--query', type=str, default="Q: What is Jury Nullification, and why would it result in a juror being removed?\nA:")
    parser.add_argument('--response', type=str, default="Jury Nullification is the idea that a jury has the right to acquit a defendant, even if the evidence presented in court shows that the defendant is guilty of the crime(s) charged. This concept is based on the idea that the jury has a constitutional right to decide the outcome of a trial, rather than simply being required to apply the law as presented by the prosecution. There are a few reasons why a juror might be removed for nullifying a jury: 1. Refusing to follow the law: If a juror is found to be intentionally ignoring or refusing to follow the law, they may be removed from the jury. 2. Bringing outside influence into the jury room: If a juror is found to have brought outside influence, such as personal opinions or biases, into the jury room, they may be removed. 3. Failure to deliberate: If a juror is found to be failing to participate in deliberations or is refusing to deliberate, they may be removed. 4. Violating confidentiality rules: If a juror is found to have violated confidentiality rules by discussing the case with someone outside of the jury room, they may be removed. 5. Causing a disruption: If a juror is found to be causing a disruption in the jury room or during trial, they may be removed. It's important to note that jurors are selected to represent a fair and impartial jury, and their removal is typically only done as a last resort. Before a juror is removed, there are many steps that are taken to try to resolve the issue, including private meetings with the juror and efforts to educate them on the importance of following the law.")
    parser.add_argument('--dataset', type=str, default="c4_realnews")
    parser.add_argument('--cache_dir', type=str, default=os.getenv('XDG_CACHE_HOME', '~/.cache'))
    parser.add_argument('--dist_alpha', type=float, default=0.05)
    parser.add_argument('--checkpoint_alpha', type=float, default=0.05)

    parser.add_argument('--check_quality', type=bool, default=True, help="whether to check quality of each perturbed text using GPT-3.5")
    parser.add_argument('--watermark_scheme', type=str, default="umd", help="the watermark scheme to attack")
    parser.add_argument('--choice_granularity', type=int, default=5, help="number of choices for quality oracle.")
    parser.add_argument('--oracle_model', type=str, default="gpt-3.5", choices=["gpt-4", "gpt-3.5"])
    parser.add_argument('--tie_threshold', type=float, default=0.001)

    parser.add_argument('--repetition_penalty', type=float, default=1.1, help="repetition penalty for the perturbation oracle")
    parser.add_argument('--mask_top_p', type=float, default=0.95, help="top p for the perturbation oracle")
    parser.add_argument('--n_spans', type=int, default=1, help="number of spans to perturb")
    parser.add_argument('--span_len', type=int, default=6,  help="pick random subsequence of k tokens in each iteration") # 3, 4 shortern # default 2?
    parser.add_argument('--gen_len', type=int, default=200, help="length of the generated watermarked text")
    parser.add_argument('--step_T', type=int, default=400, help="number or randomw walks/iterations")
    parser.add_argument('--mask_filling_model_name', type=str, default="google/t5-v1_1-xl")
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--random_fills', type=bool, default=False)
    # parser.add_argument('--perturb_type', type=str, default="mask", help="t5 or mask")
    args = parser.parse_args()
    return args
