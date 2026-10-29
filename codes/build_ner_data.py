# build_ner_data.py
import os
from data_utils import read_line_examples_from_file

def build_bio_file(data_path, output_path):
    sentences, labels = read_line_examples_from_file(data_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for words, quads in zip(sentences, labels):
            bio_tags = ['O'] * len(words)
            for at, ac, sp, ot in quads:
                if at != 'NULL':
                    at_tokens = at.split()
                    _mark_span(words, at_tokens, bio_tags, 'ASP')
                if ot != 'NULL':
                    ot_tokens = ot.split()
                    _mark_span(words, ot_tokens, bio_tags, 'OPI')
            for token, tag in zip(words, bio_tags):
                f.write(f"{token}\t{tag}\n")
            f.write("\n")

def _mark_span(words, term_tokens, bio_tags, prefix):
    term_len = len(term_tokens)
    for i in range(len(words) - term_len + 1):
        if words[i:i+term_len] == term_tokens:
            bio_tags[i] = f'B-{prefix}'
            for j in range(i+1, i+term_len):
                bio_tags[j] = f'I-{prefix}'
            return

if __name__ == '__main__':
    datasets = ['Rest15', 'Rest16', 'Laptop', 'Restaurant']
    for ds in datasets:
        for split in ['train', 'dev', 'test']:
            src = f'../data/{ds}/{split}.txt'
            out = f'../data/ner/{ds}/{split}.txt'
            if os.path.exists(src):
                build_bio_file(src, out)
                print(f"Built {out}")
