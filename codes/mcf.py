"""
mcf.py - Multi-Candidate Fusion (MCF)
对多个候选输出的四元组进行频率与检索覆盖度加权融合。
"""
from collections import Counter


class MCF:
    def __init__(self, gamma=0.5, min_score=1.0):
        """
        :param gamma: 检索覆盖度权重
        :param min_score: 保留四元组的最低融合分数
        """
        self.gamma = gamma
        self.min_score = min_score

    def fuse(self, all_candidate_quads, aspect_set=None, opinion_set=None):
        """
        :param all_candidate_quads: List[List[quad]]，每个候选解析出的四元组列表
        :param aspect_set: 检索到的 aspect 候选词集合（小写）
        :param opinion_set: 检索到的 opinion 候选词集合（小写）
        :return: 融合后的四元组列表
        """
        if not all_candidate_quads:
            return []

        aspect_set = aspect_set or set()
        opinion_set = opinion_set or set()

        counter = Counter()
        for quads in all_candidate_quads:
            for q in quads:
                if len(q) == 4:
                    counter[tuple(q)] += 1

        if not counter:
            return []

        max_freq = max(counter.values())

        scored = []
        for quad, freq in counter.items():
            at, ac, sp, ot = quad
            # 检索覆盖度：at 和 ot 是否都在检索候选里
            r = 0.0
            if at != 'NULL' and at.lower() in aspect_set:
                r += 0.5
            if ot != 'NULL' and ot.lower() in opinion_set:
                r += 0.5
            score = freq / max_freq + self.gamma * r
            if score >= self.min_score * 0.4:  # 阈值：相对最高分的一个比例
                scored.append((score, quad))

        scored.sort(key=lambda x: -x[0])
        return [q for _, q in scored]
