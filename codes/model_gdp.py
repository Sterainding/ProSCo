import math
import json
import warnings
import losses
from pathlib import Path

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize
from transformers import LogitsProcessor

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, get_linear_schedule_with_warmup
from transformers.modeling_outputs import (BaseModelOutput, Seq2SeqLMOutput)

from data_utils import get_dataset
from eval_utils import compute_scores, extract_spans_para, compute_f1_scores
from losses import SupConLoss
from sar import SARRefiner, build_category_map
from mcf import MCF
from data_utils import read_line_examples_from_file


class RetrievalConstrainedLogitsProcessor(LogitsProcessor):
    """论文 3.3 节：在 [ASP] / [OPI] 标记后，把不在 V_allowed 中的 token logits 置为 -inf。"""

    def __init__(self, asp_marker_id, opi_marker_id,
                 allowed_asp_ids_per_sample, allowed_opi_ids_per_sample,
                 neg_inf=-1e9):
        self.asp_marker_id = asp_marker_id
        self.opi_marker_id = opi_marker_id
        self.allowed_asp_ids_per_sample = allowed_asp_ids_per_sample
        self.allowed_opi_ids_per_sample = allowed_opi_ids_per_sample
        self.neg_inf = neg_inf

    def __call__(self, input_ids, scores):
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            last_token = input_ids[i, -1].item()
            if last_token == self.asp_marker_id:
                allowed = self.allowed_asp_ids_per_sample[i]
            elif last_token == self.opi_marker_id:
                allowed = self.allowed_opi_ids_per_sample[i]
            else:
                continue
            if not allowed:
                continue
            mask = torch.full_like(scores[i], self.neg_inf)
            allowed_tensor = torch.tensor(list(allowed), device=scores.device, dtype=torch.long)
            mask[allowed_tensor] = 0.0
            scores[i] = scores[i] + mask
        return scores


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(768, 1024)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        last_state = torch.mul(x, attention_mask.unsqueeze(-1))
        features = torch.sum(last_state, dim=1)
        features_drop = self.dropout(features)
        return torch.stack((self.layer_1(features), self.layer_1(features_drop)), 1)


class GridLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_e = nn.Linear(768, 1024)
        self.layer_d = nn.Linear(768, 1024)
        self.layer_1 = nn.Linear(2048, 3)
        self.dropout = nn.Dropout(0.1)
        self.grid_loss_fct = CrossEntropyLoss(ignore_index=-100)

    def forward(self, encoder_last_hidden_state, decoder_hidden_state, encoder_mask, decoder_mask, grid_label):
        e_s_len = encoder_last_hidden_state.shape[1]
        d_s_len = decoder_hidden_state.shape[1]

        encoder_state = torch.nn.functional.relu(self.layer_e(encoder_last_hidden_state))
        decoder_state = torch.nn.functional.relu(self.layer_d(decoder_hidden_state))

        encode_last_state = torch.mul(encoder_state, encoder_mask.unsqueeze(-1))
        decode_last_state = torch.mul(decoder_state, decoder_mask.unsqueeze(-1))

        encode_last_state = encode_last_state.unsqueeze(1).expand(-1, d_s_len, -1, -1)
        decode_last_state = decode_last_state.unsqueeze(2).expand(-1, -1, e_s_len, -1)

        grid_state = torch.cat([decode_last_state, encode_last_state], dim=3)
        grid_state = self.dropout(grid_state)

        features_summed = self.layer_1(grid_state)
        dropped = self.dropout(features_summed)
        grid_loss = self.grid_loss_fct(dropped.view(-1, dropped.size(-1)), grid_label.view(-1))
        return grid_loss


class LightningModule(pl.LightningModule):

    def __init__(self, args, model_name_or_path, tokenizer):
        super(LightningModule, self).__init__()
        self.save_hyperparameters(ignore=['grid_model', 'sp_model', 'ot_model', 'at_model', 'ac_model'])

        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model.resize_token_embeddings(len(tokenizer))
        self.args = args
        self.tokenizer = tokenizer

        self.dropout = nn.Dropout(args.p)
        self.df_num = args.df_num
        self.grid_model = GridLinearModel()
        self.sp_model = LinearModel()
        self.ot_model = LinearModel()
        self.at_model = LinearModel()
        self.ac_model = LinearModel()
        self.test_outputs_buffer = []
        self.test_targets_buffer = []

        # ========== MRT & ProSCo 参数 ==========
        self.use_mrt = getattr(args, 'use_mrt', False)
        self.mrt_start_epoch = getattr(args, 'mrt_start_epoch', 10)
        self.mrt_k = getattr(args, 'mrt_k', 4)
        self.mrt_temperature = getattr(args, 'mrt_temperature', 0.8)

        self.use_soft_reward = getattr(args, 'use_soft_reward', False)
        self.soft_alpha, self.soft_beta, self.soft_gamma, self.soft_delta = 0.3, 0.2, 0.2, 0.3

        self.use_retrieval_weighting = getattr(args, 'use_retrieval_weighting', False)
        self.retrieval_weight_lambda = getattr(args, 'retrieval_weight_lambda', 0.2)

        self.use_constrained_decoding = getattr(args, 'use_constrained_decoding', False)
        self.asp_marker_id = tokenizer.convert_tokens_to_ids('[ASP]')
        self.opi_marker_id = tokenizer.convert_tokens_to_ids('[OPI]')

        # 奖励统计
        self.reward_history = []
        self.baseline_history = []
        self.loss_history = []
        self.reward_ema = None
        self.reward_ema_alpha = 0.95
        self.log_interval = 20

        # ========== SAR 初始化 ==========
        self.use_sar = getattr(args, 'use_sar', False)
        self.sar_refiner = None
        if self.use_sar:
            train_path = f'../data/{self.args.dataset}/train.txt'
            cat_map = {}
            try:
                _, train_labels = read_line_examples_from_file(train_path)
                cat_map = build_category_map(train_labels)
            except Exception as e:
                print(f"[SAR] Failed to build category map: {e}")

            ner_tokenizer, ner_model = None, None
            ner_path = getattr(args, 'ner_model_path', None)
            if ner_path:
                import os as _os
                ner_abs_path = _os.path.abspath(ner_path)
                if _os.path.isdir(ner_abs_path):
                    try:
                        from transformers import AutoTokenizer, AutoModelForTokenClassification
                        ner_tokenizer = AutoTokenizer.from_pretrained(ner_abs_path)
                        ner_model = AutoModelForTokenClassification.from_pretrained(ner_abs_path)
                        ner_model.eval()
                    except Exception as e:
                        print(f"[SAR] Failed to load NER model: {e}")
            self.sar_refiner = SARRefiner(category_map=cat_map, ner_tokenizer=ner_tokenizer, ner_model=ner_model)

        # ========== MCF 初始化 ==========
        self.use_mcf = getattr(args, 'use_mcf', False)
        self.mcf_num_candidates = getattr(args, 'mcf_num_candidates', 8)
        self.mcf_diversity_penalty = getattr(args, 'mcf_diversity_penalty', 1.0)
        self.mcf_num_beam_groups = getattr(args, 'mcf_num_beam_groups', 4)
        self.mcf = MCF(gamma=0.5, min_score=1.0) if self.use_mcf else None

    def is_logger(self):
        return True

    def vagueify(self, sequence_output):
        all_logits = None
        for _ in range(self.df_num):
            temp = self.dropout(sequence_output)
            if self.model.config.tie_word_embeddings:
                temp = temp * (self.model.model_dim ** -0.5)
            lm_logits = self.model.lm_head(temp).unsqueeze(0)
            if all_logits is not None:
                all_logits = torch.cat((all_logits, lm_logits), dim=0)
            else:
                all_logits = lm_logits
        return all_logits

    @torch.no_grad()
    def get_vague_samples(self, all_logits, labels):
        all_top_k_ids = torch.topk(all_logits, 1, dim=-1)
        all_c = all_top_k_ids.indices.squeeze(-1)
        batch_size = labels.shape[0]
        all_lengths = (labels != 0).sum(-1)
        mask_results = torch.zeros_like(all_logits)
        updated_labels = labels.clone()
        updated_labels[updated_labels == -100] = self.args.pad_value
        for i in range(all_c.shape[0]):
            for j in range(batch_size):
                mask_results[i, j, :all_lengths[j]] = mask_results[i, j, :all_lengths[j]].scatter(
                    -1, all_c[i, j, :all_lengths[j]].unsqueeze(-1), 1)
            mask_results[i] = mask_results[i].scatter(-1, updated_labels.unsqueeze(-1), 0)
        return mask_results

    def compute_vague_loss(self, all_logits, N_mask_results, lm_labels):
        log_softmax = nn.LogSoftmax(dim=-1)
        softmax_fct = nn.Softmax(dim=-1)
        label_masks = torch.ones_like(lm_labels)
        label_masks[lm_labels == 0] = 0
        softmax_logits = softmax_fct(all_logits)
        all_log_softmax_logits = log_softmax(all_logits)
        df_num = softmax_logits.shape[0]
        n_logits = (self.args.gama * softmax_logits).exp() * N_mask_results
        n_logits = n_logits.sum(0).sum(-1)
        labels = lm_labels.unsqueeze(0).repeat(df_num, 1, 1).unsqueeze(-1)
        p_logits = torch.gather((- (self.args.gama * softmax_logits)).exp(), -1, labels)
        p_logits = p_logits.sum(0).squeeze(-1)
        label_loss = torch.log(1 + math.exp(self.args.m * self.args.gama) * n_logits * p_logits) * label_masks
        label_loss = label_loss.sum()
        vocab_size = all_log_softmax_logits.shape[-1]
        loss_fct = nn.NLLLoss(ignore_index=-100, reduction='sum')
        all_likelihood_loss = torch.stack([
            loss_fct(logits.reshape(-1, vocab_size), lm_labels.view(-1))
            for logits in all_log_softmax_logits
        ])
        likelihood_loss = torch.mean(all_likelihood_loss)
        regular_loss = softmax_logits * all_log_softmax_logits
        mi_loss = -((regular_loss * label_masks.unsqueeze(0).unsqueeze(-1)).sum())
        return label_loss + likelihood_loss + mi_loss

    # ============================================================
    # 软奖励与置信度加权 (论文公式 5 & 6)
    # ============================================================
    @staticmethod
    def _token_f1(pred, gold):
        p = pred.lower().strip()
        g = gold.lower().strip()
        p_null = p in ('null', 'it', '')
        g_null = g in ('null', 'it', '')
        if p_null and g_null: return 1.0
        if p_null or g_null: return 0.0
        p_tokens, g_tokens = set(p.split()), set(g.split())
        if not p_tokens or not g_tokens: return 0.0
        tp = len(p_tokens & g_tokens)
        precision, recall = tp / len(p_tokens), tp / len(g_tokens)
        if precision + recall == 0: return 0.0
        return 2 * precision * recall / (precision + recall)

    def _soft_quad_reward(self, pred_quads, gold_quads):
        if not pred_quads: return 0.0
        exact_f1 = compute_f1_scores([pred_quads], [gold_quads])['f1'] if gold_quads else 0.0
        matched_gold_idx = set()
        span_sum, cat_sum, pol_sum = 0.0, 0.0, 0.0

        for pq in pred_quads:
            best_joint, best_match, best_gi = -1.0, None, -1
            for gi, gq in enumerate(gold_quads):
                if gi in matched_gold_idx: continue
                at_sim = self._token_f1(pq[0], gq[0])
                ot_sim = self._token_f1(pq[3], gq[3])
                span_sim = (at_sim + ot_sim) / 2.0
                cat_match = 1.0 if pq[1] == gq[1] else 0.0
                pol_match = 1.0 if pq[2] == gq[2] else 0.0
                joint = span_sim + cat_match + pol_match
                if joint > best_joint:
                    best_joint, best_match, best_gi = joint, (span_sim, cat_match, pol_match), gi
            if best_match is not None:
                matched_gold_idx.add(best_gi)
                span_sum += best_match[0]
                cat_sum += best_match[1]
                pol_sum += best_match[2]

        n = len(pred_quads)
        return (self.soft_alpha * (span_sum / n) + self.soft_beta * (cat_sum / n) +
                self.soft_gamma * (pol_sum / n) + self.soft_delta * exact_f1)

    def _retrieval_weighted_reward(self, pred_quads, gold_quads, retrieval_candidates):
        base = self._soft_quad_reward(pred_quads, gold_quads) if self.use_soft_reward else \
            (compute_f1_scores([pred_quads], [gold_quads])['f1'] if gold_quads else 0.0)

        if not pred_quads or not retrieval_candidates: return base

        asp_conf, opi_conf = {}, {}
        for item in retrieval_candidates:
            if len(item) == 3:
                span, typ, conf = item
            else:
                span, typ = item
                conf = 1.0
            key = span.lower()
            if typ == 'ASP':
                asp_conf[key] = max(asp_conf.get(key, 0.0), conf)
            else:
                opi_conf[key] = max(opi_conf.get(key, 0.0), conf)

        def _match(term, conf_dict):
            if term in ('NULL', '', None): return 0.0
            t = term.lower().strip()
            if t in conf_dict: return conf_dict[t]
            for cand, c in conf_dict.items():
                if t in cand or cand in t: return c
            return 0.0

        r_ret = 0.0
        for pq in pred_quads:
            r_ret += (_match(pq[0], asp_conf) + _match(pq[3], opi_conf)) / 2
        r_ret /= len(pred_quads)

        return base + self.retrieval_weight_lambda * r_ret

    # ============================================================

    def train_dataloader(self):
        self.train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.args)
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, drop_last=True, shuffle=True,
                          num_workers=4)

    def val_dataloader(self):
        self.val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.args)
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=4)

    def test_dataloader(self):
        self.test_dataset = get_dataset(tokenizer=self.tokenizer, type_path="test", args=self.args)
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=4)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        encoder_last_hidden_states = encoder_outputs.last_hidden_state

        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self.model._shift_right(labels)

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_last_hidden_states,
            encoder_attention_mask=attention_mask, return_dict=True)
        decoder_last_layer_hiddens = decoder_outputs.last_hidden_state

        all_logits = self.vagueify(decoder_last_layer_hiddens)
        N_mask_results = self.get_vague_samples(all_logits, labels)
        loss = self.compute_vague_loss(all_logits, N_mask_results, labels)
        main_output = Seq2SeqLMOutput(loss=loss, logits=all_logits)

        if self.sp_model or self.ot_model or self.at_model:
            masked_encoder = torch.mul(encoder_last_hidden_states, attention_mask.unsqueeze(-1))
            pooled_encoder = normalize(torch.sum(masked_encoder, dim=1), p=2.0, dim=1)
            sp_pred = self.sp_model(encoder_last_hidden_states, attention_mask) if self.sp_model else None
            ot_pred = self.ot_model(encoder_last_hidden_states, attention_mask) if self.ot_model else None
            at_pred = self.at_model(encoder_last_hidden_states, attention_mask) if self.at_model else None
            return main_output, encoder_last_hidden_states, decoder_last_layer_hiddens, sp_pred, ot_pred, at_pred, pooled_encoder
        else:
            return main_output, encoder_last_hidden_states, decoder_last_layer_hiddens

    def _step(self, batch):
        lm_labels = batch["target_ids"].clone()
        lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100

        if self.current_epoch < self.args.stat_full_train_ep:
            ac_mask_matrix = batch["ac_mask_matrix"]
            lm_labels = lm_labels * ac_mask_matrix + ((1 - ac_mask_matrix) * -100)

        outputs = self(input_ids=batch["source_ids"], attention_mask=batch["source_mask"],
                       labels=lm_labels, decoder_attention_mask=batch['target_mask'])
        loss = outputs[0].loss

        if getattr(self.args, 'category_loss_lambda', 0) > 0:
            ac_mask_matrix = batch["ac_mask_matrix"]
            category_labels = lm_labels * ac_mask_matrix + ((1 - ac_mask_matrix) * -100)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss += loss_fct(outputs[1].view(-1, outputs[1].size(-1)),
                             category_labels.view(-1)) * self.args.category_loss_lambda

        if getattr(self.args, 'scl_loss_lambda', 0) > 0 and len(outputs) > 3:
            criterion = SupConLoss(loss_scaling_factor=self.args.cont_loss, temperature=self.args.cont_temp)
            sp_loss = criterion(normalize(outputs[3], p=2.0, dim=2), batch['sp_labels']) * self.args.scl_loss_lambda
            at_loss = criterion(normalize(outputs[4], p=2.0, dim=2), batch['at_labels']) * self.args.scl_loss_lambda
            ot_loss = criterion(normalize(outputs[5], p=2.0, dim=2), batch['ot_labels']) * self.args.scl_loss_lambda
            loss += sp_loss + at_loss + ot_loss

        if getattr(self.args, 'grid_loss_lambda', 0) > 0 and len(outputs) > 1:
            grid_loss = self.grid_model(outputs[1], outputs[2], batch["source_mask"], batch["target_mask"],
                                        batch['grid_label']) * self.args.grid_loss_lambda
            loss += grid_loss
        return loss, outputs

    def training_step_mrt(self, batch, batch_idx):
        input_ids = batch["source_ids"]
        attention_mask = batch["source_mask"]
        K = self.mrt_k

        with torch.no_grad():
            sampled_ids = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_length=self.args.max_seq_length * 2, do_sample=True, temperature=self.mrt_temperature,
                num_return_sequences=K, return_dict_in_generate=False, output_scores=False)

        batch_size = input_ids.size(0)
        gold_quads_list = self._get_batch_gold_quads(batch)

        # 提取 NER 候选用于置信度奖励
        retrieval_candidates_batch = []
        if 'retrieval_candidates' in batch:
            raw = batch['retrieval_candidates']
            if isinstance(raw, (list, tuple)):
                for s in raw:
                    try:
                        retrieval_candidates_batch.append(json.loads(s))
                    except Exception:
                        retrieval_candidates_batch.append([])
            else:
                retrieval_candidates_batch = [[] for _ in range(batch_size)]

        rewards = []
        for i in range(batch_size):
            cands = sampled_ids[i * K:(i + 1) * K]
            cand_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in cands]
            parsed_cands = [extract_spans_para(self.args.task, text, 'pred') for text in cand_texts]
            gold = gold_quads_list[i]
            ret_cands = retrieval_candidates_batch[i] if retrieval_candidates_batch else []

            if self.use_retrieval_weighting:
                r = [self._retrieval_weighted_reward(p, gold, ret_cands) for p in parsed_cands]
            elif self.use_soft_reward:
                r = [self._soft_quad_reward(p, gold) for p in parsed_cands]
            else:
                r = [compute_f1_scores([p], [gold])['f1'] for p in parsed_cands]
            rewards.append(r)

        rewards = torch.tensor(rewards, device=self.device)

        # Leave-one-out baseline (论文公式 4)
        if K > 1:
            sum_rewards = rewards.sum(dim=1, keepdim=True)
            baseline = (sum_rewards - rewards) / (K - 1)
        else:
            baseline = rewards

        mean_reward, mean_baseline = rewards.mean().item(), baseline.mean().item()
        self.reward_history.append(mean_reward)
        self.baseline_history.append(mean_baseline)
        self.reward_ema = mean_reward if self.reward_ema is None else self.reward_ema_alpha * self.reward_ema + (
                    1 - self.reward_ema_alpha) * mean_reward

        if len(self.reward_history) % self.log_interval == 0:
            tag = "rw" if self.use_retrieval_weighting else ("soft" if self.use_soft_reward else "exact")
            print(
                f"[MRT-{tag}] step={len(self.reward_history)} | reward={mean_reward:.4f} | baseline={mean_baseline:.4f} | ema={self.reward_ema:.4f}")

        log_probs = self._compute_seq_log_probs(sampled_ids, batch)
        advantages = (rewards - baseline).detach()
        mrt_loss = - (advantages * log_probs).mean()

        mle_loss, _ = self._step(batch)

        # ==============================================================
        # 【核心修改】论文公式(7)：MLE 与 MRT 的线性退火 (Linear Annealing)
        # lambda_t 从 1 线性降到 0
        # ==============================================================
        mrt_epochs = self.args.num_train_epochs - self.mrt_start_epoch
        if mrt_epochs > 0:
            progress = (self.current_epoch - self.mrt_start_epoch) / max(1, mrt_epochs)
            lambda_t = max(0.0, 1.0 - progress)
        else:
            lambda_t = 0.5

        total_loss = lambda_t * mle_loss + (1 - lambda_t) * mrt_loss
        # ==============================================================

        self.loss_history.append(mrt_loss.item())
        return total_loss

    def _get_batch_gold_quads(self, batch):
        targets = batch["target_ids"]
        gold_quads_list = []
        for i in range(targets.size(0)):
            target_text = self.tokenizer.decode(targets[i], skip_special_tokens=True)
            gold_quads_list.append(extract_spans_para(self.args.task, target_text, 'gold'))
        return gold_quads_list

    def _compute_seq_log_probs(self, sampled_ids, batch):
        batch_size = batch["source_ids"].size(0)
        K = sampled_ids.size(0) // batch_size
        decoder_input_ids = torch.cat([
            torch.full((sampled_ids.size(0), 1), self.model.config.decoder_start_token_id, device=self.device),
            sampled_ids[:, :-1]], dim=1)
        expanded_input_ids = batch["source_ids"].unsqueeze(1).expand(-1, K, -1).reshape(batch_size * K, -1)
        expanded_attention_mask = batch["source_mask"].unsqueeze(1).expand(-1, K, -1).reshape(batch_size * K, -1)

        outputs = self.model(input_ids=expanded_input_ids, attention_mask=expanded_attention_mask,
                             decoder_input_ids=decoder_input_ids, return_dict=True)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        token_log_probs = torch.gather(log_probs, -1, sampled_ids.unsqueeze(-1)).squeeze(-1)
        seq_log_probs = token_log_probs.sum(dim=1)
        return seq_log_probs.view(batch_size, K)

    def training_step(self, batch, batch_idx):
        if self.use_mrt and self.current_epoch >= self.mrt_start_epoch:
            loss = self.training_step_mrt(batch, batch_idx)
        else:
            loss, _ = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def _quads_to_string(self, quads):
        if not quads: return ""
        return " [SSEP] ".join([f"THE {at} IS {ot} | {ac} | {sp}" for at, ac, sp, ot in quads])

    def _get_retrieval_sets(self, sentence_text):
        if self.sar_refiner is None: return set(), set()
        try:
            asp, opi = self.sar_refiner._get_candidates(sentence_text)
            return set(a.lower() for a in asp), set(o.lower() for o in opi)
        except Exception:
            return set(), set()

    # ============================================================
    # 约束解码相关准备 (论文 3.3 节)
    # ============================================================
    def _build_allowed_token_ids(self, candidates, sentence_text):
        sentence_ids = set(self.tokenizer.encode(sentence_text, add_special_tokens=False))
        asp_ids, opi_ids = set(sentence_ids), set(sentence_ids)
        for item in candidates:
            span, typ = item[:2]
            for v in [span, " " + span]:
                ids = self.tokenizer.encode(v, add_special_tokens=False)
                if typ == 'ASP':
                    asp_ids.update(ids)
                else:
                    opi_ids.update(ids)
        for w in ['NULL', 'null', 'it', 'It', ' NULL', ' null', ' it', ' It']:
            ids = self.tokenizer.encode(w, add_special_tokens=False)
            asp_ids.update(ids)
            opi_ids.update(ids)
        return asp_ids, opi_ids

    def _make_constrained_processor(self, batch, batch_size):
        raw = batch.get('retrieval_candidates', None)
        retrieval_candidates_batch = []
        if raw is not None:
            if isinstance(raw, (list, tuple)):
                for s in raw:
                    try:
                        retrieval_candidates_batch.append(json.loads(s))
                    except Exception:
                        retrieval_candidates_batch.append([])
            else:
                retrieval_candidates_batch = [[] for _ in range(batch_size)]
        else:
            retrieval_candidates_batch = [[] for _ in range(batch_size)]

        sentence_texts = batch.get("sentence_text", None)
        if sentence_texts is None:
            sentence_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["source_ids"]]

        allowed_asp, allowed_opi = [], []
        for i in range(batch_size):
            cands = retrieval_candidates_batch[i] if retrieval_candidates_batch else []
            a, o = self._build_allowed_token_ids(cands, sentence_texts[i])
            allowed_asp.append(a)
            allowed_opi.append(o)

        return RetrievalConstrainedLogitsProcessor(
            asp_marker_id=self.asp_marker_id, opi_marker_id=self.opi_marker_id,
            allowed_asp_ids_per_sample=allowed_asp, allowed_opi_ids_per_sample=allowed_opi)

    # ============================================================

    def test_step(self, batch, batch_idx):
        sentence_texts = batch.get("sentence_text", None)
        if sentence_texts is None:
            sentence_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["source_ids"]]

        batch_size = batch['source_ids'].size(0)
        logits_processor = None
        if self.use_constrained_decoding:
            logits_processor = [self._make_constrained_processor(batch, batch_size)]

        if self.use_mcf:
            K = self.mcf_num_candidates
            gen_kwargs = dict(
                input_ids=batch['source_ids'], attention_mask=batch['source_mask'],
                max_length=self.args.max_seq_length * 2, num_beams=K,
                num_beam_groups=self.mcf_num_beam_groups, diversity_penalty=self.mcf_diversity_penalty,
                num_return_sequences=K, early_stopping=True, no_repeat_ngram_size=3)
            if logits_processor: gen_kwargs['logits_processor'] = logits_processor

            outs = self.model.generate(**gen_kwargs)
            out_dec = []
            for i in range(batch_size):
                cands = outs[i * K:(i + 1) * K]
                cand_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in cands]
                parsed = [extract_spans_para(self.args.task, t, 'pred') for t in cand_texts]
                asp_set, opi_set = self._get_retrieval_sets(sentence_texts[i])
                fused_quads = self.mcf.fuse(parsed, asp_set, opi_set)
                out_dec.append(self._quads_to_string(fused_quads))
        else:
            gen_kwargs = dict(
                input_ids=batch['source_ids'], attention_mask=batch['source_mask'],
                max_length=self.args.max_seq_length * 2, num_beams=5)
            if logits_processor: gen_kwargs['logits_processor'] = logits_processor

            outs = self.model.generate(**gen_kwargs)
            out_dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

        tar_dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        if self.use_sar and self.sar_refiner is not None:
            if self.sar_refiner.ner_model is not None:
                self.sar_refiner.ner_model.to(self.device)
                self.sar_refiner.ner_model.eval()
            refined_out = []
            for i, text in enumerate(out_dec):
                quads = extract_spans_para(self.args.task, text, 'pred')
                try:
                    new_quads = self.sar_refiner.refine(quads, sentence_texts[i])
                except Exception as e:
                    print(f"[SAR] Refine error: {e}")
                    new_quads = quads
                refined_out.append(self._quads_to_string(new_quads))
            out_dec = refined_out

        self.test_outputs_buffer.extend(out_dec)
        self.test_targets_buffer.extend(tar_dec)
        return {'outputs': out_dec, 'targets': tar_dec}

    def on_test_start(self):
        self.test_outputs_buffer = []
        self.test_targets_buffer = []

    def on_test_epoch_end(self):
        if self.reward_history:
            reward_log = {
                'reward': self.reward_history, 'baseline': self.baseline_history,
                'mrt_loss': self.loss_history, 'reward_ema_final': self.reward_ema,
                'reward_ema_alpha': self.reward_ema_alpha, 'num_mrt_steps': len(self.reward_history),
                'use_soft_reward': self.use_soft_reward, 'use_retrieval_weighting': self.use_retrieval_weighting,
                'use_constrained_decoding': self.use_constrained_decoding,
            }
            with open(f"{self.args.output_dir}/mrt_reward_log.json", 'w') as f:
                json.dump(reward_log, f, indent=2)

        outputs = self.test_outputs_buffer
        targets = self.test_targets_buffer
        scores, all_labels, all_preds = compute_scores(outputs, targets, self.args.task, False)
        results = {'labels_correct': all_labels, 'labels_pred': all_preds,
                   'output_pred': outputs, 'output_correct': targets}
        ex_list = [{k: results[k][idx] for k in results} for idx in range(len(all_preds))]
        json.dump({'performance_metrics': scores, 'examples': ex_list},
                  open(f"{self.args.output_dir}/results-{self.args.dataset}.json", 'w'),
                  indent=2, sort_keys=True)
        print(scores)

    def get_training_steps(self):
        dataloader = self.train_dataloader()
        total_steps = (len(dataloader.dataset) // (self.args.batch_size * max(1, len(self.args.gpus))))
        total_steps = total_steps // self.args.gradient_accumulation_steps * float(self.args.num_train_epochs)
        return total_steps

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {"params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(params, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.get_training_steps())
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
