import random
import numpy as np
import torch
from config import get_args
args=get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def softmax(x):
    x = x - torch.max(x)
    exp_x = torch.exp(x)
    softmax_x = exp_x / torch.sum(exp_x)
    return softmax_x

def predict_next_word(model,tokenizer,input_text,direction):

    tokens_tensor = {k: v.to(device) for k, v in tokenizer(input_text, padding=True, return_tensors="pt").items()}
    # Set the model in evaluation mode to deactivate the DropOut modules
    model.eval()
    # If you have a GPU, put everything on cuda
    # Predict all tokens
    with torch.no_grad():
      outputs = model(**tokens_tensor)
      predictions = outputs[0]

    # Get the predicted next sub-word
    # if [0, -1, :] --> dim_size (1, 50257); if [:, -1, :] --> (50257,)
    probs = torch.tensor(predictions[:, -1, :],dtype=torch.float32)

    dst = args.dst
    if dst == 'yelp' or dst == 'amazon':
        pos_logits = probs[:,tokenizer.encode('positive')]
        neg_logits = probs[:,tokenizer.encode('negative')]

    emo_logits = torch.dstack((neg_logits,pos_logits)).squeeze(1)
    softmax_emo_logits = torch.softmax(emo_logits,dim=1)

    neg_prob = softmax_emo_logits[:, 0]
    pos_prob = softmax_emo_logits[:, 1]

    if direction=='0-1':
        output_prob = pos_prob / neg_prob  # make the prob more robust
    else: #1-0
        output_prob = neg_prob / pos_prob

    dst = args.dst

    if args.setting=='zero-shot':
        if dst == 'yelp':
            thres_neg=0.6
            thres_pos=0.9
        elif dst == 'amazon':
            thres_neg=0.7
            thres_pos=0.9

    emo_argmax_labels=torch.argmax(softmax_emo_logits,dim=1)
    labels = []
    for idx in range(len(softmax_emo_logits)):
        if emo_argmax_labels[idx] == 0 and neg_prob[idx]>=thres_neg:
            labels.append(0) # 'negative'
        elif emo_argmax_labels[idx] == 1 and pos_prob[idx]>=thres_pos:
            labels.append(1) # 'positive'
        else: labels.append(2) # 'neutral'


    return output_prob, labels

def pytorch_cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    return cos_sim(a, b)

def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))
