import json
import random
from torch.utils.data import Dataset
import torch
from generate_data import get_prosco_data, get_domain, domain_map

sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}


def get_dataset(tokenizer, type_path, args):
    return ProSCoDataset(tokenizer=tokenizer,
                      data_dir=args.dataset,
                      data_type=type_path,
                      max_len=args.max_seq_length,
                      task=args.task,
                      truncate=args.truncate,
                      use_rag=getattr(args, 'use_rag', False),
                      use_ner_retrieval=getattr(args, 'use_ner_retrieval', False),
                      ner_model_path=getattr(args, 'ner_model_path', None))


def read_line_examples_from_file(data_path, silence=False):
    sentences, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sentences.append(words.split())
                labels.append(eval(tuples))
    return sentences, labels


def get_para_asqp_targets(sents, labels, truncated=False):
    targets = []
    for label in labels:
        all_quad_sentences = []
        for quad in label:
            at, ac, sp, ot = quad
            if truncated:
                ac = ac.split("#")[0]
            man_ot = sentword2opinion[sp]
            if at == 'NULL':
                at = 'it'
            one_quad_sentence = f"{ac} is {man_ot} because {at} is {ot}"
            all_quad_sentences.append(one_quad_sentence)
        target = ' [SSEP] '.join(all_quad_sentences)
        targets.append(target)
    return sents.copy(), targets


def replace_unk_tokens(string):
    replace_dict = {"`": "'"}
    for pstr in replace_dict:
        string = string.replace(pstr, replace_dict[pstr])
    return string


def _read_preformatted_file(path):
    inputs, targets = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            inp, tgt = parts
            inputs.append(inp.split())
            targets.append(tgt)
    return inputs, targets


def get_transformed_io(data_path, data_dir, task, data_type, truncate=False):
    if task == 'mvp':
        return (*_read_preformatted_file(f'../data/{data_dir}/mvp/{data_type}.txt'), [])
    if task == 'chen':
        return (*_read_preformatted_file(f'../data/{data_dir}/chen/{data_type}.txt'), [])
    if task == 'dot_stage1':
        return (*_read_preformatted_file(f'../data/{data_dir}/dot/stage1_{data_type}.txt'), [])
    if task in ('dot_stage2', 'dot'):
        return (*_read_preformatted_file(f'../data/{data_dir}/dot/stage2_{data_type}.txt'), [])

    sentences, labels = read_line_examples_from_file(data_path)
    inputs, targets = get_prosco_data(sentences, labels, task, truncate)
    return inputs, targets, labels


class ProSCoDataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, task, max_len=256, data_path=None, truncate=False,
                 with_labels=True, use_rag=False, use_ner_retrieval=False, ner_model_path=None):
        if data_path:
            self.data_path = data_path
        else:
            self.data_path = f'../data/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.task = task
        self.data_type = data_type
        self.inputs = []
        self.targets = []
        self.sentence_strings = []
        self.truncate = truncate
        self.with_labels = with_labels
        self.use_rag = use_rag
        self.use_ner_retrieval = use_ner_retrieval

        self.is_baseline_task = task in ('mvp', 'chen', 'dot_stage1', 'dot_stage2', 'dot')

        # 每个样本的检索候选：(span, type, confidence)
        self.retrieval_candidates = []

        if self.use_ner_retrieval and not self.is_baseline_task:
            import os
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            ner_abs_path = os.path.abspath(ner_model_path)
            if not os.path.isdir(ner_abs_path):
                raise FileNotFoundError(f"NER 模型目录不存在: {ner_abs_path}")
            self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_abs_path)
            self.ner_model = AutoModelForTokenClassification.from_pretrained(ner_abs_path)
            self.ner_model.eval()
            self.ner_model.to('cuda' if torch.cuda.is_available() else 'cpu')
            self.id2label = self.ner_model.config.id2label
            print(f"[RAG] NER model loaded from {ner_abs_path}")
        elif self.use_rag and not self.is_baseline_task:
            self.aspect_vocab, self.opinion_vocab = self._build_vocab_from_train()

        if self.with_labels and not self.is_baseline_task:
            self.contrastive_labels = {'sp': [], 'at': [], 'ot': []}
            self.grid_labels = []
            self.category_mask_matrixes = []

        self._build_examples()

    def _build_vocab_from_train(self):
        train_path = f'../data/{self.data_dir}/train.txt'
        sentences, labels = read_line_examples_from_file(train_path)
        aspect_vocab, opinion_vocab = set(), set()
        for quads in labels:
            for at, ac, sp, ot in quads:
                if at != 'NULL':
                    for w in at.split():
                        aspect_vocab.add(w.lower())
                if ot != 'NULL':
                    for w in ot.split():
                        opinion_vocab.add(w.lower())
        return aspect_vocab, opinion_vocab

    def _extract_candidates(self, words):
        """返回 [(span, type, confidence), ...]，confidence ∈ [0,1]。"""
        if self.use_ner_retrieval:
            return self._extract_candidates_ner(words)
        else:
            candidates, seen = [], set()
            for w in words:
                w_lower = w.lower()
                if w_lower in self.aspect_vocab and ('ASP', w) not in seen:
                    candidates.append((w, 'ASP', 1.0)); seen.add(('ASP', w))
                if w_lower in self.opinion_vocab and ('OPI', w) not in seen:
                    candidates.append((w, 'OPI', 1.0)); seen.add(('OPI', w))
            return candidates[:10]

    def _extract_candidates_ner(self, words):
        """使用 NER 模型预测 aspect/opinion span，并返回 softmax 平均置信度。"""
        text = ' '.join(words)
        inputs = self.ner_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.ner_model.device)
        with torch.no_grad():
            outputs = self.ner_model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)   # (L, C)
        preds = torch.argmax(outputs.logits, dim=-1).squeeze(0)    # (L,)
        # 每个位置取对应预测标签的 softmax 概率
        token_probs = probs.gather(-1, preds.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        predictions = preds.cpu().numpy()

        tokens = self.ner_tokenizer.convert_ids_to_tokens(inputs.input_ids.squeeze().tolist())
        word_ids = inputs.word_ids(batch_index=0)

        candidates, current_span, current_type, current_probs = [], [], None, []
        for token, pred, word_id, prob in zip(tokens, predictions, word_ids, token_probs):
            if word_id is None:
                continue
            label = self.id2label[int(pred)]
            if label.startswith('B-'):
                if current_span:
                    candidates.append((' '.join(current_span), current_type, float(sum(current_probs) / len(current_probs))))
                current_span = [words[word_id] if word_id < len(words) else token]
                current_type = 'ASP' if label.endswith('ASP') else 'OPI'
                current_probs = [prob]
            elif label.startswith('I-') and current_span:
                current_span.append(words[word_id] if word_id < len(words) else token)
                current_probs.append(prob)
            else:
                if current_span:
                    candidates.append((' '.join(current_span), current_type, float(sum(current_probs) / len(current_probs))))
                    current_span, current_type, current_probs = [], None, []
        if current_span:
            candidates.append((' '.join(current_span), current_type, float(sum(current_probs) / len(current_probs))))
        return candidates[:10]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        item = {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "sentence_text": " ".join(self.sentence_strings[index]),
            # 序列化为 JSON 字符串，便于默认 collate 处理
            "retrieval_candidates": json.dumps(self.retrieval_candidates[index], ensure_ascii=False)
        }

        if self.with_labels and not self.is_baseline_task:
            item.update({
                'sp_labels': torch.tensor(self.contrastive_labels['sp'][index]),
                'ot_labels': torch.tensor(self.contrastive_labels['ot'][index]),
                'at_labels': torch.tensor(self.contrastive_labels['at'][index]),
                'grid_label': self.grid_labels[index],
                'ac_mask_matrix': self.category_mask_matrixes[index]
            })

        return item

    def get_raw_labels(self):
        return get_transformed_io(self.data_path, self.data_dir, self.task, self.data_type, self.truncate)

    def _build_examples(self):
        inputs, targets, labels = get_transformed_io(
            self.data_path, self.data_dir, self.task, self.data_type, self.truncate
        )
        self.sentence_strings = inputs

        for i in range(len(inputs)):
            original_words = inputs[i]
            if (self.use_rag or self.use_ner_retrieval) and not self.is_baseline_task:
                candidates = self._extract_candidates(original_words)
                self.retrieval_candidates.append(candidates)

                prefix_tokens = []
                for span, typ, _ in candidates:
                    prefix_tokens.append('[ASP]' if typ == 'ASP' else '[OPI]')
                    prefix_tokens.extend(span.split())
                prefix_str = ' '.join(prefix_tokens)
                input_text = f"{prefix_str} [SEP] {' '.join(original_words)}"
            else:
                self.retrieval_candidates.append([])
                input_text = ' '.join(original_words)
            input_text = replace_unk_tokens(input_text)

            target_text = targets[i]
            if isinstance(target_text, list):
                target_text = " ".join(target_text)
            target_text = replace_unk_tokens(target_text)

            tokenized_input = self.tokenizer.batch_encode_plus(
                [input_text], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
                [target_text], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)

        if self.with_labels and labels and not self.is_baseline_task:
            self._process_labels(labels)

    # -------- 以下 _process_labels / _build_grid_matrices / _build_token_to_char_map / _find_token_positions --------
    # 与原代码保持一致，无需修改，此处省略以节省篇幅，请沿用你现有版本。
    # 请把原文件里这几个方法原封不动保留。
    def _process_labels(self, labels):
        sentiment_dict = {'negative': 0, 'neutral': 1, 'positive': 2, 'mixed': 3}
        self.contrastive_labels['sp'] = [
            sentiment_dict[list(set([quad[2] for quad in ex]))[0]] if len(set([quad[2] for quad in ex])) == 1
            else sentiment_dict['mixed']
            for ex in labels
        ]
        opinion_dict = {'NULL': 0, 'EXPLICIT': 1, 'BOTH': 2}
        self.contrastive_labels['ot'] = [
            opinion_dict['EXPLICIT'] if 'NULL' not in set([quad[3] for quad in ex])
            else (opinion_dict['NULL'] if len(set([quad[3] for quad in ex])) == 1
                  else opinion_dict['BOTH'])
            for ex in labels
        ]
        aspect_dict = {'NULL': 0, 'EXPLICIT': 1, 'BOTH': 2}
        self.contrastive_labels['at'] = [
            aspect_dict['EXPLICIT'] if 'NULL' not in set([quad[0] for quad in ex])
            else (aspect_dict['NULL'] if len(set([quad[0] for quad in ex])) == 1
                  else aspect_dict['BOTH'])
            for ex in labels
        ]
        self.grid_labels, self.category_mask_matrixes = self._build_grid_matrices(
            self.inputs, self.targets, labels
        )

    def _build_grid_matrices(self, inputs, targets, labels):
        grid_member_dict = {'pad': -100, 'none': 0, 'aspect': 1, 'opinion': 2,
                            'negative': 3, 'neutral': 4, 'positive': 5}
        grid_matrixes, category_mask_matrixes = [], []
        for i in range(len(inputs)):
            sentence_encoded = inputs[i]["input_ids"].squeeze()
            targets_encoded = targets[i]["input_ids"].squeeze()
            non_pad_sentence_len = sum(self.tokenizer.pad_token_id != sentence_encoded).item()
            non_pad_target_len = sum(self.tokenizer.pad_token_id != targets_encoded).item()
            grid_matrix = torch.full((len(targets_encoded), len(sentence_encoded)), grid_member_dict['pad'])
            category_mask_matrix = torch.zeros((len(targets_encoded),))
            sentence_map = self._build_token_to_char_map(sentence_encoded, non_pad_sentence_len)
            target_map = self._build_token_to_char_map(targets_encoded, non_pad_target_len)
            for quad in labels[i]:
                at, ac, _, ot = quad
                if self.truncate:
                    ac = ac.split("#")[0]
                domain = get_domain(ac)
                asc = ac if domain == "unknown" else domain_map[domain].get(ac, ac)
                asc_clean = replace_unk_tokens(asc.replace(" ", ""))
                asc_pos = self._find_token_positions(asc_clean, targets_encoded, target_map, non_pad_target_len)
                if asc_pos:
                    grid_matrix[asc_pos[0]:asc_pos[1], :non_pad_sentence_len] = grid_member_dict['none']
                    category_mask_matrix[asc_pos[0]:asc_pos[1]] = 1
                if at != "NULL":
                    at_clean = replace_unk_tokens(at.replace(" ", ""))
                    at_pos = self._find_token_positions(at_clean, sentence_encoded, sentence_map, non_pad_sentence_len)
                    if at_pos and asc_pos:
                        grid_matrix[asc_pos[0]:asc_pos[1], at_pos[0]:at_pos[1]] = grid_member_dict['aspect']
                if ot != "NULL":
                    ot_clean = replace_unk_tokens(ot.replace(" ", ""))
                    ot_pos = self._find_token_positions(ot_clean, sentence_encoded, sentence_map, non_pad_sentence_len)
                    if ot_pos and asc_pos:
                        grid_matrix[asc_pos[0]:asc_pos[1], ot_pos[0]:ot_pos[1]] = grid_member_dict['opinion']
            grid_matrixes.append(grid_matrix)
            category_mask_matrixes.append(category_mask_matrix)
        return grid_matrixes, category_mask_matrixes

    def _build_token_to_char_map(self, tokens, non_pad_len):
        decoded_str, token_map = "", []
        for i in range(1, non_pad_len + 1):
            current_decoded = self.tokenizer.decode(tokens[:i], clean_up_tokenization_spaces=False).replace(' ', '')
            new_chars = len(current_decoded) - len(decoded_str)
            token_map.extend([i - 1] * new_chars)
            decoded_str = current_decoded
        return token_map

    def _find_token_positions(self, text, tokens, token_map, non_pad_len):
        full_text = self.tokenizer.decode(tokens[:non_pad_len], clean_up_tokenization_spaces=False).replace(' ', '')
        if text not in full_text:
            return None
        start = full_text.index(text)
        end = start + len(text) - 1
        return (token_map[start], token_map[end] + 1)
