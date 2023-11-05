from transformers import pipeline,RobertaTokenizer,RobertaForSequenceClassification
import torch
from tqdm import tqdm
import sys
sys.path.append("../")
import argparse
from model_utils.helpers import softmax
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_name="siebert/sentiment-roberta-large-english"
sty_tokenizer = RobertaTokenizer.from_pretrained(model_name)
sty_model = RobertaForSequenceClassification.from_pretrained(model_name).to(device)


pos=0
neg=0

parser = argparse.ArgumentParser()
parser.add_argument('--gen_path', default='../output.txt', type=str)
args=parser.parse_args()

def classifier(text):
    inputs = sty_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = sty_model(**inputs).logits
    softmax_logits = softmax(logits)
    outputs = {}
    predicted_class_id = softmax_logits.argmax().item()
    outputs['label'] = sty_model.config.id2label[predicted_class_id]
    outputs['score'] = softmax_logits.squeeze()[predicted_class_id]

    return [outputs]


with open(args.gen_path,'r',encoding='utf8') as f:
    datas=f.readlines()

    for idx,data in tqdm(enumerate(datas)):
        tokens = data.strip()
        if idx< 500:

            res=classifier(tokens)
            if res[0]['label'].lower()=='positive':
                pos+=1

            else:neg+=1
        else:
            res = classifier(tokens)
            if res[0]['label'].lower() == 'negative':
                pos += 1
            else:
                neg += 1
    #
    length=idx+1 # 500

print("POS ACC is {}".format(pos/length))

