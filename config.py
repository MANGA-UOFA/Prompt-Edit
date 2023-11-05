import argparse

def get_args():

    parser = argparse.ArgumentParser(description="model parameters")
    parser.add_argument('--output_dir', type=str, default="output/", help='Output directory path to store checkpoints.')

    ## Model building
    parser.add_argument('--max_len', type=int, default=16,help='Input length of model')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator')
    parser.add_argument('--class_name',default='../EleutherAI/gpt-neo-1.3B',type=str)
    parser.add_argument('--topk', default=50, type=int,help="top-k words in masked out word prediction")
    parser.add_argument("--direction", type=str, default='0-1',help='0-1 | 1-0')
    parser.add_argument("--fluency_weight", type=int, default=1, help='fluency')
    parser.add_argument("--sem_weight",type=int, default=1, help='semantic similarity')
    parser.add_argument("--style_weight", type=int, default=8, help='style')
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument('--dst', default='yelp', type=str,help='yelp | gyafc | amazon')
    parser.add_argument("--setting", type=str, default='zero-shot')
    parser.add_argument("--bsz",type=int,default=1,help="batch size")
    parser.add_argument('--keyword_pos', default=True, type=bool)
    parser.add_argument("--early_stop",default=False, type=bool)

    args, _ = parser.parse_known_args()
    return args


