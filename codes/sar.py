"""
sar.py - Structure-Aware Refinement (SAR)
对生成的四元组应用三种规则修正：span grounding、category alignment、polarity consistency
"""
from collections import Counter

POSITIVE_WORDS = {"great","excellent","love","good","nice","perfect","delicious",
                  "wonderful","fantastic","amazing","friendly","helpful","fresh",
                  "beautiful","tasty","best","awesome","enjoy","enjoyed","favorite"}
NEGATIVE_WORDS = {"bad","terrible","awful","poor","worse","worst","slow","rude",
                  "overcooked","bland","cold","soggy","horrible","disappointing",
                  "disappointed","nasty","hate","boring","dirty","stale","unhygienic"}

def get_sentiment(word):
    w = word.lower().strip()
    if w in POSITIVE_WORDS: return "positive"
    if w in NEGATIVE_WORDS: return "negative"
    return "neutral"

def build_category_map(labels):
    freq = {}
    for quads in labels:
        for at, ac, sp, ot in quads:
            if at and at != 'NULL':
                freq.setdefault(at.lower(), Counter())[ac] += 1
    return {at: c.most_common(1)[0][0] for at, c in freq.items()}

def _edit_distance(s1, s2):
    """计算 Levenshtein 距离"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    return dp[m][n]


class SARRefiner:
    def __init__(self, aspect_vocab=None, opinion_vocab=None, category_map=None,
                 ner_tokenizer=None, ner_model=None):
        self.aspect_vocab = aspect_vocab or set()
        self.opinion_vocab = opinion_vocab or set()
        self.category_map = category_map or {}
        self.ner_tokenizer = ner_tokenizer
        self.ner_model = ner_model

    def _get_candidates(self, sentence):
        if self.ner_model is not None and self.ner_tokenizer is not None:
            return self._extract_ner(sentence)
        words = sentence.split()
        asp = [w for w in words if w.lower() in self.aspect_vocab]
        opi = [w for w in words if w.lower() in self.opinion_vocab]
        return asp, opi

    def _extract_ner(self, sentence):
        import torch
        enc = self.ner_tokenizer(sentence, truncation=True, max_length=512)
        device = next(self.ner_model.parameters()).device
        input_ids = torch.tensor([enc["input_ids"]]).to(device)
        attention_mask = torch.tensor([enc["attention_mask"]]).to(device)
        with torch.no_grad():
            outputs = self.ner_model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
        tokens = self.ner_tokenizer.convert_ids_to_tokens(enc["input_ids"])
        word_ids = enc.word_ids(batch_index=0)

        aspects, opinions = [], []
        cur_span, cur_type = [], None
        for token, pred, wid in zip(tokens, preds, word_ids):
            if wid is None: continue
            label = self.ner_model.config.id2label[int(pred)]
            if label.startswith('B-'):
                if cur_span: (aspects if cur_type == 'ASP' else opinions).append(' '.join(cur_span))
                cur_span = [token]
                cur_type = 'ASP' if label.endswith('ASP') else 'OPI'
            elif label.startswith('I-') and cur_span:
                cur_span.append(token)
            else:
                if cur_span:
                    (aspects if cur_type == 'ASP' else opinions).append(' '.join(cur_span))
                    cur_span, cur_type = [], None
        if cur_span: (aspects if cur_type == 'ASP' else opinions).append(' '.join(cur_span))
        return aspects, opinions

    def refine(self, quads, sentence):
        asp_cands, opi_cands = self._get_candidates(sentence)
        asp_set = set(a.lower() for a in asp_cands)
        opi_set = set(o.lower() for o in opi_cands)
        sent_lower = sentence.lower()

        corrected, seen = [], set()
        for quad in quads:
            if len(quad) != 4: continue
            at, ac, sp, ot = quad

            # 1. Span grounding (tau = 3)
            if at != 'NULL' and at.lower() not in asp_set and at.lower() not in sent_lower:
                new_at = self._closest_match(at, asp_cands, threshold=3)
                if new_at: at = new_at
                else: continue
            if ot != 'NULL' and ot.lower() not in opi_set and ot.lower() not in sent_lower:
                new_ot = self._closest_match(ot, opi_cands, threshold=3)
                if new_ot: ot = new_ot
                else: continue

            # 2. Category alignment
            if at != 'NULL':
                mapped = self.category_map.get(at.lower())
                if mapped and mapped != ac: ac = mapped

            # 3. Polarity consistency
            if ot != 'NULL':
                inferred = get_sentiment(ot)
                if inferred != 'neutral' and inferred != sp: sp = inferred

            new_quad = (at, ac, sp, ot)
            if new_quad not in seen:
                corrected.append(new_quad)
                seen.add(new_quad)
        return corrected

    def _closest_match(self, term, candidates, threshold=3):
        """论文 3.4 节：基于编辑距离的 Span Grounding"""
        if not candidates: return None
        best, min_dist = None, float('inf')
        for c in candidates:
            dist = _edit_distance(term.lower(), c.lower())
            if dist < min_dist:
                min_dist, best = dist, c
        return best if min_dist <= threshold else None
