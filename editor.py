from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import torch.nn as nn
import numpy as np
import RAKE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import nltk
from model_utils.examples import stopwords
from transformers import logging
logging.set_verbosity_error()

class RobertaEditor(nn.Module):
    def __init__(self, opt):
        super(RobertaEditor, self).__init__()
        self.opt = opt
        self.topk = opt.topk
        self.model_dir='roberta-large'
        self.model = RobertaForMaskedLM.from_pretrained(self.model_dir, return_dict=True).to(device)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_dir)

        self.ops_map = [self.replace, self.insert, self.delete]
        self.max_len = opt.max_len
        self.Rake = RAKE.Rake(RAKE.SmartStopList())

        print("Editor built")

    def edit(self, inputs, ops, positions,bsz,max_len=None):
        masked_inputs = [self.ops_map[op](inp, position) for inp, op, position, in zip(inputs, ops, positions)]
        array_masked_inputs = np.array(masked_inputs)

        if ops[0] < 2:  # replacement or insertion, have a mask
            index = [idx for idx, masked_input in enumerate(masked_inputs) if '<mask>' in masked_input]
            mask_inputs=array_masked_inputs[index]
            mask_outputs=self.generate(mask_inputs.tolist(),max_len)
            output_lists = mask_outputs[0]
        else: # deletion, no mask
            output_lists=masked_inputs

        return output_lists


    def generate(self, input_texts, max_len):

        total_sent_list = []
        rbt_sent_list = self.plm_token(input_texts,max_len)

        for idx,rbt_sent in enumerate(rbt_sent_list):

            ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(rbt_sent)]).to(device)
            mask_token_index = (ids==self.tokenizer.mask_token_id).long().max(dim=1).indices
            token_logits = self.model(ids).logits
            masked_token_logits = token_logits[0, mask_token_index, :]

            mask_words_list = list(set(self.tokenizer.decode([token.item()]).lower().strip() for token in torch.topk(masked_token_logits, self.topk, dim=1).indices[0]))

            # delete the stopwords
            mask_words=[]

            for token in mask_words_list:
                # token has no overlap in stopwords
                if token.isdigit() == True: _token = [token]
                else: _token = token

                if len(set(_token) & set(stopwords))==0 and token not in 'bcdefghjklmnopqrstvwxyz':
                    mask_words.append(token)

            #mask_words=mask_words_list
            sent_list=[]
            for mask_word in mask_words:
                split_text=input_texts[idx].split()
                mask_word_index=mask_token_index-1
                #in case the word is overlapped with pre or post word.

                if 0<mask_word_index<len(split_text)-1 and \
                        (split_text[mask_word_index-1]==mask_word or split_text[mask_word_index+1]==mask_word): continue
                elif mask_word_index==0 and split_text[mask_word_index + 1] == mask_word: continue
                elif mask_word_index==len(split_text)-1 and split_text[mask_word_index - 1] == mask_word: continue
                else:
                    cand_sent = input_texts[idx].replace("<mask>", mask_word.strip()).lower()
                    cand_sent = ' '.join(cand_sent.split()[:max_len])
                    sent_list.append(cand_sent)

            total_sent_list.append(sent_list)

            if total_sent_list==[[]]:
                print(input_texts[idx])


        return total_sent_list

    def insert(self, input_texts, mask_idx):
        input_texts_with_mask_list = input_texts.split()[:mask_idx] + ["<mask>"] + input_texts.split()[mask_idx:]
        return " ".join(input_texts_with_mask_list)

    def replace(self, input_texts, mask_idx):
        input_texts_with_mask_list = input_texts.split()[:mask_idx] + ["<mask>"] + input_texts.split()[mask_idx + 1:]
        return " ".join(input_texts_with_mask_list)

    def delete(self, input_texts, mask_idx):
        input_texts_with_mask_list = input_texts.split()[:mask_idx] + input_texts.split()[mask_idx + 1:]
        return " ".join(input_texts_with_mask_list)

    def plm_token(self,lines,max_len):
        rbt_lines=[]
        for line in lines:
            plm_line = []
            line=line.split()
            for idx, token in enumerate(line):
                if idx==0:
                    plm_line.append(token)
                else:
                    if token in ["'s","'d","'m","'re","'ve","'ll","n't","<mask>"]:
                        plm_line.append(token)
                    else:
                        token = 'Ġ' + token
                        plm_line.append(token)
            plm_line=plm_line[:self.max_len]
            rbt_line = ['<s>'] + plm_line + ['</s>']
            rbt_lines.append(rbt_line)

        return rbt_lines

    def state_vec(self, inputs):
        sta_vec_list = []
        pos_list = []

        for line in inputs:
            line = ' '.join(line.split()[:self.max_len])

            sta_vec = list(np.zeros([self.max_len]))
            keyword = self.Rake.run(line)
            pos_tags = nltk.tag.pos_tag(line.split())
            pos = [x[1] for x in pos_tags]
            pos_list.append(pos)

            if keyword != []:
                keyword=list(list(zip(*keyword))[0])
                keyword_new = []
                linewords = line.split()
                for i in range(len(linewords)):
                    for item in keyword:
                        length11 = len(item.split())
                        if ' '.join(linewords[i:i + length11]) == item:
                            keyword_new.extend([i + k for k in range(length11)])
                for i in range(len(keyword_new)):
                    ind = keyword_new[i]
                    if ind <= self.max_len - 2:
                        sta_vec[ind] = 1

            if self.opt.keyword_pos == True:
                sta_vec_list.append(self.keyword_pos2sta_vec(sta_vec, pos))
            else:
                if np.sum(sta_vec) == 0:
                    sta_vec[0] = 1
                sta_vec_list.append(sta_vec)


        return sta_vec_list,pos_list

    def mean_pooling(self, model_output, attention_mask):
        #reference: https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_contextual_word_embeddings(self, input_texts):
        inputs = {k: v.to(device) for k, v in self.tokenizer(input_texts, padding=True, return_tensors="pt").items()}

        outputs = self.model(**inputs, output_hidden_states=True)
        sentence_embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
        hidden_states = outputs.hidden_states[-1][:, 1:self.max_len+1, :].to(device)

        return hidden_states, sentence_embeddings

    def keyword_pos2sta_vec(self, keyword, pos):
        key_ind = []
        pos = pos[:self.max_len]
        for i in range(len(pos)):
            if keyword[i] == 1:
                key_ind.append(i)
            elif pos[i] in ['JJS', 'JJR', 'JJ', 'RBR', 'RBS', 'RB', 'VBZ', 'VBP', 'VBN', 'VBG', 'VBD', 'VB'] \
                    and keyword[i] == 0:
                key_ind.append(i)


        sta_vec = []
        for i in range(len(keyword)):
            if i in key_ind:
                sta_vec.append(1)
            else:
                sta_vec.append(0)

        if np.sum(sta_vec) == 0:
            sta_vec[0] = 1
        return sta_vec