Prompt-Based Editing for Text Style Transfer
=======

## Installation

```shell
pip3 install -U pip
conda create -n plm python=3.8
conda activate plm
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install transformers
pip3 install -r requirements.txt
```

## Model downloading

We use Eleuther AI's [gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B/tree/main) as the main architecture.

Please also download the following models for experiments.

[gpt2](https://huggingface.co/gpt2/tree/main)

[RoBERTa-Large](https://huggingface.co/roberta-large/tree/main)

[distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/tree/main)

## Model running
```
bash run.sh
```

## Evaluation
Put the generated outputs in `0-1.txt` into `output.txt`, then the generated outputs `1-0.txt` into `output.txt`.

For Accuracy
```
cd evaluator/
python3 sentiment.py --gen_path ../data/yelp/output.txt 
```

For BLEU
```
perl multi-bleu.perl ../data/yelp/references/ref0 ../data/yelp/references/ref1 ../data/yelp/references/ref2 ../data/yelp/references/ref3 < ../data/yelp/output.txt 

```

## Cite our work
If you find this repo helpful, please consider citing our work:
```bibtex
@inproceedings{luo-etal-2023-prompt,
    title = "Prompt-Based Editing for Text Style Transfer",
    author = "Luo, Guoqing  and
      Han, Yu  and
      Mou, Lili  and
      Firdaus, Mauajama",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    book title = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    year = "2023",
    url = "https://aclanthology.org/2023.findings-emnlp.381",
    pages = "5740--5750",
}
```
