# -*- coding: utf-8 -*-
"""检索阈值实测校准

为什么需要这个脚本
------------------
rag/rag_chain.py 里的 STRONG_TH / WEAK_TH 原本是按旧嵌入模型 bge-small-zh-v1.5
的实测分布定的（设定类 0.44~0.69、无关类 0.36~0.40，自然分界约 0.43）。
但当前知识库 rag/build_kb.py 用的是 paraphrase-multilingual-MiniLM-L12-v2，
余弦分布的整体形态不同，**阈值不可迁移**。

后果很具体：若 STRONG_TH 高于当前模型实际能达到的最高分，强档分支就是死代码
—— 生产永远只输出弱档「模糊背景」格式，而训练集里 1500 条 RAG 样本用的是强档
「参考资料」格式，训推直接错位。这正是本次修 rag_chain 时发现的缺陷。

本脚本用**训练语料同源的问句**实测各类查询的相似度分布，判断可分性并给出建议值。

用法：
  python finetune/scripts/calibrate_threshold.py
  python finetune/scripts/calibrate_threshold.py --k 5 --show 3
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "finetune", "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.stdout.reconfigure(encoding="utf-8")

import build_samples as bs
from rag.rag_chain import (STRONG_TH, WEAK_TH, TOP_K,
                           distance_to_similarity, vectorstore_space)

# 域内正例：应该被检索命中并进入强档
CAT_RAG = "A_域内RAG问句"
CAT_ENT = "B_域内实体"
# 域外负例：不该拿到高相似度资料
CAT_OFF = "C_离题"
CAT_CROSS = "D_跨角色探问"
CAT_BENIGN = "E_良性闲聊"

POS_CATS = (CAT_RAG, CAT_ENT)
NEG_CATS = (CAT_OFF, CAT_CROSS, CAT_BENIGN)


def percentile(vals, p):
    """线性插值分位数。不引 numpy，避免与训练环境抢依赖。"""
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    idx = (len(s) - 1) * (p / 100.0)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def build_queries(npcs, tpl_limit=3):
    """全部取自 build_samples.py 的同源常量，保证测的就是训练/生产会遇到的问句。"""
    qs = []

    tags = []
    for npc in npcs.values():
        for t in npc.get("knowledge_tags", []):
            if t and t not in tags:
                tags.append(t)
    for t in tags:
        for tpl in bs.RAG_QUESTION_TEMPLATES[:tpl_limit]:
            qs.append((CAT_RAG, tpl.format(tag=t)))

    for npc in npcs.values():
        if npc.get("name"):
            qs.append((CAT_ENT, npc["name"]))
        if npc.get("title"):
            qs.append((CAT_ENT, npc["title"]))

    for q in bs.UNKNOWN_QS:
        qs.append((CAT_OFF, q))
    for q in bs.CROSS_CHAR_QS:
        qs.append((CAT_CROSS, q))
    for q in bs.BENIGN_QS:
        qs.append((CAT_BENIGN, q))
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=TOP_K, help="每条查询取前 k 个命中")
    ap.add_argument("--show", type=int, default=3, help="每类打印几个极值样例")
    ap.add_argument("--tpl-limit", type=int, default=3, help="每个 tag 用几个问句句式")
    cli = ap.parse_args()

    npcs = bs.load_npcs()
    print(f"[NPC] {len(npcs)} 个角色")

    print(f"[向量库] 连接中（嵌入模型跑 CPU，不影响正在进行的训练）...")
    db = bs.load_chroma()
    space = vectorstore_space(db)
    print(f"[向量库] hnsw:space = {space}  "
          f"（换算式：{'1-d' if space == 'cosine' else '-d' if space == 'ip' else '1-d²/2'}）")

    queries = build_queries(npcs, cli.tpl_limit)
    print(f"[查询] {len(queries)} 条，k={cli.k}\n")

    # cat -> [(query, [sim_top1..k])]
    rows = {}
    for cat, q in queries:
        try:
            hits = db.similarity_search_with_score(q, k=cli.k)
        except Exception as e:
            print(f"[错误] 检索失败 {type(e).__name__}: {e}")
            return 1
        if not hits:
            continue
        sims = [distance_to_similarity(float(d), space) for _, d in hits]
        rows.setdefault(cat, []).append((q, sims))

    if not rows:
        print("[错误] 没有任何命中 —— ChromaDB 可能是空库，先跑 rag/build_kb.py")
        return 1

    def top1(cat):
        return [sims[0] for _, sims in rows.get(cat, [])]

    print("=" * 78)
    print(f"{'类别':<18}{'条数':>5}{'min':>8}{'p25':>8}{'中位':>8}{'p75':>8}{'max':>8}")
    print("-" * 78)
    all_cats = [c for c in (CAT_RAG, CAT_ENT, CAT_OFF, CAT_CROSS, CAT_BENIGN) if c in rows]
    for cat in all_cats:
        v = top1(cat)
        print(f"{cat:<18}{len(v):>5}{min(v):>8.3f}{percentile(v,25):>8.3f}"
              f"{percentile(v,50):>8.3f}{percentile(v,75):>8.3f}{max(v):>8.3f}")

    # ---- 当前阈值下的实际命中情况：这是判断「强档是否死代码」的直接证据 ----
    print("=" * 78)
    print(f"[现行阈值] STRONG_TH={STRONG_TH}  WEAK_TH={WEAK_TH}")
    print(f"{'类别':<18}{'进强档':>8}{'进弱档':>8}{'被丢弃':>8}")
    print("-" * 78)
    for cat in all_cats:
        v = top1(cat)
        st = sum(1 for x in v if x >= STRONG_TH)
        wk = sum(1 for x in v if WEAK_TH <= x < STRONG_TH)
        dp = sum(1 for x in v if x < WEAK_TH)
        print(f"{cat:<18}{st:>8}{wk:>8}{dp:>8}")

    pos = [x for c in POS_CATS for x in top1(c)]
    neg = [x for c in NEG_CATS for x in top1(c)]
    if not pos or not neg:
        print("\n[错误] 正例或负例为空，无法给出建议")
        return 1

    pos_lo, neg_hi = percentile(pos, 25), max(neg)
    gap = pos_lo - neg_hi
    print("=" * 78)
    print(f"[可分性] 正例 p25={pos_lo:.3f}   负例 max={neg_hi:.3f}   间隔={gap:+.3f}")

    if gap > 0.02:
        sug_strong = round((pos_lo + neg_hi) / 2, 2)
        sug_weak = round(max(min(neg) - 0.02, percentile(neg, 75)), 2)
        print(f"  ✓ 两类可分。建议 STRONG_TH={sug_strong}  WEAK_TH={sug_weak}")
        print(f"    依据：强档门槛放在正例 p25 与负例 max 的中点，"
              f"既保证约 3/4 的域内问句能进强档，又不让任何负例混入。")
    else:
        sug_strong = round(percentile(pos, 50), 2)
        sug_weak = round(percentile(neg, 50), 2)
        print(f"  ⚠ 两类重叠，当前嵌入模型无法靠单一阈值干净区分域内/域外。")
        print(f"    退而求其次：STRONG_TH={sug_strong}（正例中位）"
              f"  WEAK_TH={sug_weak}（负例中位）")
        print(f"    这意味着约一半负例会被当资料注入 —— 必须靠训练集里的")
        print(f"    adv_offtopic / adv_benign / adv_cross_char 三类对抗样本兜底，")
        print(f"    让模型学会「即使给了无关资料也不照着答」。")

    if max(pos) < STRONG_TH:
        print(f"\n  ✗ 已确认：正例最高分 {max(pos):.3f} < 现行 STRONG_TH {STRONG_TH}，")
        print(f"    强档分支是**死代码**，生产从未输出过「参考资料」格式。必须回填阈值。")

    # ---- 极值样例，供人工核对阈值是否合理 ----
    if cli.show > 0:
        print("=" * 78)
        for cat in all_cats:
            items = sorted(rows[cat], key=lambda x: -x[1][0])
            print(f"[{cat}] 最高分 {cli.show} 条：")
            for q, sims in items[:cli.show]:
                print(f"    {sims[0]:.3f}  {q}")
            print(f"[{cat}] 最低分 {cli.show} 条：")
            for q, sims in items[-cli.show:]:
                print(f"    {sims[0]:.3f}  {q}")
            print("-" * 78)

    print(f"\n[下一步] 把建议值回填到 rag/rag_chain.py 的 STRONG_TH / WEAK_TH。")
    print(f"         回填后 build_samples.py 生成的强/弱档配比会随之变化，")
    print(f"         因此必须**先校准阈值、再重生成训练集**，顺序不能颠倒。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
