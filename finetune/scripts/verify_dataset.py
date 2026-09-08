# -*- coding: utf-8 -*-
"""训练集校验（重生成之后、启动训练之前必跑）

目的不是"确认文件存在"，而是拦住那些会让 8.5 小时训练白跑的静默缺陷：

  1. 格式    每行合法 JSON、messages 三条、role 顺序为 system/user/assistant
  2. 训推一致 用生产的 build_sysp 反向重建每条 system 提示词。这是本次重构的
             核心目标 —— 训练数据的提示词必须与 rag_chain.ask() 逐字节相同，
             否则模型学的是一套、线上跑的是另一套。机器验证，不靠人工抽查。
  3. 上下文   extra 只能是四种合法形态：空 / 无命中回退 / 强档头 / 弱档头。
             文案从 format_context 现取，不硬编码，改了 rag_chain 自动跟随。
  4. 长度    tokenize 后超过 max_length 的样本会被截断；截断点若落在
             completion 中间，模型学到的就是"话说一半"。
  5. 答案异常 残句、标点串（「。，」「。。」）、括号旁白、AI 自认、空答案。
  6. 多样性   逐字重复率。口语化改写若退化成"750 条共用一个模板"，
             模型会学成固定口头禅 —— 那是另一种穿帮。
  7. 配比    与 dataset_stats.json / lora_config.json 对账，发现某类静默产出 0 条。
  8. 元数据  剧本/系统策划标记（「触发条件」「B4节点」「MSG_INITIATE」「玩家」）。
             检测正则**独立编写**，不复用 rag_chain.META_SENT_RES —— 用同一把尺子
             量自己，正则漏掉的类型永远查不出来。

用法：
  python finetune/scripts/verify_dataset.py
  python finetune/scripts/verify_dataset.py --no-tokenizer   # 跳过长度校验（快）
退出码：发现 ERROR 级问题时返回 1，可直接用来阻断训练启动。
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "finetune", "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.stdout.reconfigure(encoding="utf-8")

from rag.rag_chain import (NO_HIT_FALLBACK, build_sysp, format_context,
                           load_npcs)

TRAIN_PATH = os.path.join(ROOT, "finetune", "data", "train.jsonl")
CONFIG_PATH = os.path.join(ROOT, "finetune", "config", "lora_config.json")
STATS_PATH = os.path.join(ROOT, "finetune", "logs", "dataset_stats.json")

# 答案必须收在句末标点上。口语化改写按句子边界截断，若仍出现残句，
# 说明 colloquialize 的切分逻辑漏了某种标点。
END_PUNCT = "。！？；…”」』"
# 明显不是人话的标点串
BAD_PUNCT = ("。，", "。、", "。。", "，，", "！！", "？？", "……，", "。，。")
NARRATION_RE = re.compile(r"[（(][^（()）]{1,24}[）)]")
AI_MARKERS = ("我是AI", "我是 AI", "人工智能", "语言模型", "作为一个AI", "作为 AI")

# 结构化标记：知识图谱三元组、箭头、框线、表格行。这类内容出现在答案里
# 意味着模型被教成了「念设定文档的排版语法」（实测过：NPC 对玩家说
# 「哼，[收费争议] ----> [行会把控码头收费…」），归为 ERROR 而不是 WARN。
STRUCT_RES = (
    re.compile(r"\[[^\[\]]{1,24}\]\s*--+"),
    re.compile(r"--+>"),
    re.compile(r"[│┌┐└┘├┤─━]{2,}"),
    re.compile(r'\{\s*"head"'),
    # tab 分隔的表格行。build_kb.py 把 docx 表格转成「键 | 值」，但 md 里还有一批
    # 从别处粘来的 tab 表（NPC 参数表、关系数值表）。竖线判据看不到它们，
    # 而正常台词里不会存在 tab，所以一条就够定性。
    re.compile(r"\t"),
    # 箭头图：区域邻接、流程图、状态流转。NPC 台词里不会出现任何箭头。
    re.compile(r"[↔→←⇒⇄]"),
    # 表格行残渣。rag_chain 的整块拒收门槛是一行 3 个竖线，2 个的走句子级删；
    # 校验器这里不管几个，台词里出现竖线就归 ERROR。
    re.compile(r"\|"),
)


def struct_hit(text):
    """命中任一种排版语法就返回其描述，否则 None。"""
    t = text or ""
    for rx in STRUCT_RES:
        if rx.search(t):
            return rx.pattern
    if any(line.count("|") >= 3 for line in t.split("\n")):
        return "表格行(一行 3 个以上竖线)"
    return None


def cites_kb(ans, ctx, n=4):
    """答案里是否存在长度 n 的连续片段也出现在注入资料里。

    用来把「带资料」样本再切一刀。对抗样本（adv_offtopic / adv_benign /
    adv_cross_char）**故意**注入资料却用固定回避话术回答 —— 那正是要教的
    行为，重复属设计意图（实测带资料群重复 top3 全是「此事老夫不便多说。」
    这类话术）。真正该盯多样性的是「答案由资料改写而来」那批。

    用 n-gram 而不是字符集合：中文单字重合度天然很高（回避话术也会用到
    「我」「不」「说」这些字），只有连续 4 字以上的公共片段才说明答案真的
    引用了资料。判据不依赖 build_samples 的类别标签（jsonl 里也没存）。
    """
    if not ans or not ctx:
        return False
    grams = {ctx[i:i + n] for i in range(len(ctx) - n + 1)}
    return any(ans[i:i + n] in grams for i in range(len(ans) - n + 1))


# 剧本/系统策划元数据。这类内容出现在 NPC 台词里，穿帮程度比排版语法更严重：
# 「唉，触发条件： 玩家在药铺遇见秀芝，未选择"问她身体"…」等于当玩家的面
# 承认自己活在游戏里；「第二层：语料检索器 / 评分维度…」更是在描述本项目
# 自己的实现。
# 这里的正则**独立编写**、刻意比 rag_chain.META_SENT_RES 更宽（含单独的「玩家」
# 「行动点」「好感度」等词）：校验器若复用被校验方的规则，规则漏掉的类型就永远
# 查不出来，等于自己给自己发合格证。宁可误报让人工看一眼，也不要漏报。
META_RES = (
    re.compile(r"(MSG|INT)_[A-Z_]+"),
    re.compile(r"触发条件"),
    re.compile(r"[A-Z]\d+\s*节点|[A-Z]\d?\s*分支|R\d+\s*路线|[A-Z]链|事件链"),
    re.compile(r"行动点|行动力|好感度|亲和度|关系值|对话深度|阈值|系数"),
    re.compile(r"System\s*Prompt|State\s*Machine|语料检索器|回复选择器|"
               r"评分维度|兜底回复|分层注入|状态机", re.I),
    re.compile(r"玩家"),
    re.compile(r"世界状态|RP值|S\d\s*[（(→]|路线倾向值|见证录|活跃层级"),
    re.compile(r"\S{2,6}系统\s*[:：]|MVP|文档版本|修正说明|性格底线速查|"
               r"通用回复策略|关系网"),
    re.compile(r"操作\s*[:：]|预期\s*[:：]|[✅❌]|JSON格式"),
    # 文档骨架：章节标题、编号小节、版本批注、项目管理词。用 re.M 对整个文本
    # 的任意行生效（rag_chain 那边是逐句调用，不需要）；且刻意比它宽 ——
    # 不要求标题后跟非空白、批注括号容字放到 10、多盯「变更/任务范围/工时/排期」。
    re.compile(r"^[一二三四五六七八九十]{1,3}[、.]", re.M),
    re.compile(r"^\d+\.\d+\s*\S", re.M),
    re.compile(r"【[^】]{0,10}(新增|修订|修正|删除|补充|变更)[^】]{0,10}】"),
    re.compile(r"实施路线图|风险预演|里程碑|交付物|防御性编程|状态访问器|"
               r"任务范围|工时|排期"),
    # UI 操作指引。rag_chain 只盯「选择 + 引号」，这里再宽一层：选项文本
    # 无论被哪种引号包着都算，且「对话/前往/点击/输入 + 引号」同样定性。
    re.compile(r"(选择|对话|前往|点击|输入|查看)\s*[\"“「『]"),
    # 叙事学术语与游戏阶段名。策划写「伏笔铺垫·第三层」「深入期内可逐步获得
    # 日记的残片」是在讲剧情怎么设计，NPC 照念等于在解说自己的剧本。
    re.compile(r"伏笔|铺垫|渐进收集|残页收集|探索期|深入期|收束期|"
               r"·第[一二三四五六七八九十\d]+(层|阶段|幕)"),
    # 结局/分支的设计说明。比 rag_chain 再宽一层：不要求冒号、多盯「多结局/
    # 真结局/坏结局/隐藏结局」这类玩家视角不该听到的剧本结构词。
    re.compile(r"分支结局|结局分支|\d+\s*个分支|收束分支|核心当事人|当事人|"
               r"结局\s*[:：]|分支\s*[:：]|路线\s*[:：]|多结局|真结局|坏结局|"
               r"隐藏结局"),
)


def meta_hit(text):
    """命中任一种策划元数据就返回其描述，否则 None。"""
    t = text or ""
    for rx in META_RES:
        if rx.search(t):
            return rx.pattern
    return None


# 孤立闭合符。build_kb.py 用 500 字滑窗（overlap 100）切块，切点会落在引号或
# 括号中间，于是块里只剩下没有开符的「”」「）」。rag_chain 用 drop_orphan_closers
# 做配对清理，这里**独立**再实现一遍同样的配对逻辑：直接调被校验方的函数
# 等于让它自己给自己打分，函数写错了校验器会跟着一起错。
CLOSERS = {"”": "“", "’": "‘", "」": "「", "』": "『",
           "）": "（", ")": "(", "》": "《", "〕": "〔"}
OPENERS = frozenset(CLOSERS.values())


def orphan_hit(text):
    """返回第一个没有配对开符的闭合符，全配对则 None。"""
    depth = {}
    for ch in text or "":
        op = CLOSERS.get(ch)
        if op is None:
            if ch in OPENERS:
                depth[ch] = depth.get(ch, 0) + 1
        elif depth.get(op, 0) > 0:
            depth[op] -= 1
        else:
            return ch
    return None


errors, warns, infos = [], [], []


def err(msg):
    errors.append(msg)


def warn(msg):
    warns.append(msg)


def info(msg):
    infos.append(msg)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def token_len(tok, s, q, a):
    """返回一条样本的 token 数。

    不能直接 len(apply_chat_template(...))：transformers 5.x 下它可能返回
    BatchEncoding 而不是 list，len() 得到的是「键的个数」（恒为 2）而且
    **不报错** —— 这是最坏的失败模式：校验跑通了，数字全是垃圾，
    还真的报个「✓ 可以启动训练」。所以必须显式剥到 input_ids。
    """
    msgs = [{"role": "system", "content": s},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}]
    out = None
    try:
        out = tok.apply_chat_template(msgs, tokenize=True,
                                      add_generation_prompt=False,
                                      return_dict=False)
    except Exception:
        out = None
    if out is None:
        # 有些 chat template 不接受 system 角色，退回裸拼接做保守估计
        out = tok(s + q + a)["input_ids"]
    if isinstance(out, dict):
        out = out.get("input_ids", out)
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], list):      # 批维 [[...]]
        out = out[0]
    return len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tokenizer", action="store_true",
                    help="跳过序列长度校验（不加载 tokenizer，快）")
    ap.add_argument("--show", type=int, default=5, help="每类问题打印几个样例")
    cli = ap.parse_args()

    if not os.path.exists(TRAIN_PATH):
        print(f"[ERROR] 训练集不存在: {TRAIN_PATH}\n        先跑 build_samples.py")
        return 1

    cfg = load_json(CONFIG_PATH)
    # NPC 一律走生产的 load_npcs()，不在本文件重新解析 npc_config.json ——
    # 否则这里的校验基准就可能与线上实际用的不是同一套。
    npcs = load_npcs()
    max_length = cfg["training"]["max_length"]
    comp = cfg["sample_composition"]

    # 上下文合法形态：从 format_context 现取，保持单一事实源
    strong_head = format_context(["X"], []).split("\n")[0]
    weak_head = format_context([], ["X"]).split("\n")[0]
    bases = {nid: build_sysp(n) for nid, n in npcs.items()}

    print(f"[配置] max_length={max_length}  NPC={len(npcs)} 个")
    print(f"[基准] 强档头: {strong_head[:24]}...")
    print(f"[基准] 弱档头: {weak_head[:24]}...")

    # ---------------- 逐行扫描 ----------------
    rows = []
    with open(TRAIN_PATH, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                err(f"第 {ln} 行是空行")
                continue
            try:
                rows.append((ln, json.loads(line)))
            except json.JSONDecodeError as e:
                err(f"第 {ln} 行不是合法 JSON: {e}")
    print(f"[读入] {len(rows)} 条样本")
    if not rows:
        return 1

    bad_struct, no_owner, bad_ctx = [], [], []
    short_ans, bad_end, bad_punct, narr, ai_hit = [], [], [], [], []
    struct_ans, struct_ctx = [], []
    meta_ans, meta_ctx = [], []
    orphan_ans, orphan_ctx = [], []
    ctx_kinds = Counter()
    ans_counter = Counter()
    # 分群统计重复率。两群的「重复」含义相反：无资料群里 NPC 反复用同一套回避语
    # （「此事老夫不便多说。」）是**设计意图**，人本来就这么说话；带资料群若逐字
    # 重复，说明口语化改写塌缩成了固定模板，模型学到的是背几十段话而不是「依据
    # 资料用自己的话说」。只看全局重复率会让前者的正常重复稀释掉后者的真问题。
    ans_ctx, ans_plain, ans_rag = Counter(), Counter(), Counter()
    lengths = []

    for ln, obj in rows:
        msgs = obj.get("messages")
        if not isinstance(msgs, list) or len(msgs) != 3:
            bad_struct.append((ln, "messages 不是 3 条"))
            continue
        roles = [m.get("role") for m in msgs]
        if roles != ["system", "user", "assistant"]:
            bad_struct.append((ln, f"role 顺序异常: {roles}"))
            continue
        sys_c = msgs[0].get("content") or ""
        q = msgs[1].get("content") or ""
        a = msgs[2].get("content") or ""

        if not q.strip():
            bad_struct.append((ln, "user 为空"))
        ans_counter[a] += 1

        # --- 训推一致性：用生产 build_sysp 反向重建 ---
        owner, extra = None, None
        for nid, b in bases.items():
            if sys_c == b:
                owner, extra = nid, ""
                break
            if sys_c.startswith(b + "\n\n"):
                owner, extra = nid, sys_c[len(b) + 2:]
                break
        if owner is None:
            no_owner.append((ln, sys_c[:60]))
            continue

        if extra == "":
            ctx_kinds["无资料"] += 1
        elif extra == NO_HIT_FALLBACK:
            ctx_kinds["无命中回退"] += 1
        elif extra.startswith(strong_head):
            ctx_kinds["强档·参考资料"] += 1
            # 强档必须带（1）（2）…编号，否则与生产 format_context 的输出不同形
            if "（1）" not in extra:
                bad_ctx.append((ln, "强档缺少（1）编号"))
        elif extra.startswith(weak_head):
            ctx_kinds["弱档·模糊背景"] += 1
            if "·" not in extra:
                bad_ctx.append((ln, "弱档缺少·条目符"))
        else:
            bad_ctx.append((ln, f"未知上下文形态: {extra[:50]}"))

        # 带资料 = 强档或弱档；无资料 / 无命中回退 / 非法形态都归无资料群
        if extra.startswith(strong_head) or extra.startswith(weak_head):
            ans_ctx[a] += 1
            if cites_kb(a, extra):
                ans_rag[a] += 1
        else:
            ans_plain[a] += 1

        # --- 答案质量 ---
        if len(a.strip()) < 2:
            short_ans.append((ln, a))
        elif a.strip()[-1] not in END_PUNCT:
            bad_end.append((ln, a[-24:]))
        for p in BAD_PUNCT:
            if p in a:
                bad_punct.append((ln, p, a[:40]))
                break
        if NARRATION_RE.search(a):
            narr.append((ln, NARRATION_RE.findall(a)[:2]))
        if any(m in a for m in AI_MARKERS):
            ai_hit.append((ln, a[:40]))

        # --- 排版语法泄漏：答案与注入资料两边都要查 ---
        hit = struct_hit(a)
        if hit:
            struct_ans.append((ln, hit, a[:56]))
        if extra:
            hit = struct_hit(extra)
            if hit:
                struct_ctx.append((ln, hit, extra[:56]))

        # --- 剧本/策划元数据泄漏：同样两边都查 ---
        m = meta_hit(a)
        if m:
            meta_ans.append((ln, m, a[:56]))
        if extra:
            m = meta_hit(extra)
            if m:
                meta_ctx.append((ln, m, extra[:56]))

        # --- 孤立闭合符：分块边界切断引号/括号留下的排版垃圾 ---
        o = orphan_hit(a)
        if o:
            orphan_ans.append((ln, o, a[:56]))
        if extra:
            o = orphan_hit(extra)
            if o:
                orphan_ctx.append((ln, o, extra[:56]))

        lengths.append((ln, sys_c, q, a))

    # ---------------- 序列长度 ----------------
    over = []
    len_checked = False
    if not cli.no_tokenizer:
        try:
            from transformers import AutoTokenizer
            # base_model 在配置顶层，不在 data 下（data 里只有 embed_model）。
            # 两种位置都兜一下，避免配置改结构后这里又静默跳过长度校验。
            base_model = cfg.get("base_model") or cfg.get("data", {}).get("base_model")
            if not base_model:
                raise KeyError("配置里找不到 base_model（顶层与 data 下都没有）")
            print(f"[tokenizer] 加载 {base_model}（只用 CPU，不碰显存）...")
            tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            toks = []
            for ln, s, q, a in lengths:
                toks.append((ln, token_len(tok, s, q, a)))
            n = len(toks)
            vals = sorted(v for _, v in toks)
            over = [(ln, v) for ln, v in toks if v > max_length]
            print(f"[长度] n={n}  min={vals[0]}  中位={vals[n//2]}  "
                  f"p95={vals[int(n*0.95)]}  p99={vals[int(n*0.99)]}  max={vals[-1]}"
                  f"   (max_length={max_length})")
            # 自检：真实的中文对话样本 tokenize 后不可能只有几个 token，
            # 也不可能条条等长。命中就说明是校验器自己坏了，而不是数据很好。
            if vals[-1] < 20 or vals[0] == vals[-1]:
                err(f"长度校验器自身故障：{n} 条样本的 token 数全落在 "
                    f"{vals[0]}~{vals[-1]}，不可能是真实值。本项结论作废。")
            else:
                len_checked = True
        except Exception as e:
            warn(f"tokenizer 加载失败，序列长度校验**未执行**: {type(e).__name__}: {e}")
    else:
        warn("已按 --no-tokenizer 跳过序列长度校验")

    # ---------------- 汇报 ----------------
    def dump(title, items, fmt):
        if not items:
            return
        print(f"\n[{title}] {len(items)} 条")
        for it in items[:cli.show]:
            print(f"    {fmt(it)}")
        if len(items) > cli.show:
            print(f"    ... 其余 {len(items) - cli.show} 条省略")

    print("\n" + "=" * 78)
    print("[上下文形态分布]")
    for k, v in ctx_kinds.most_common():
        print(f"    {k:<16}{v:>6}  ({v / len(rows) * 100:.1f}%)")

    # 逐字重复：全局数字只作参考，分群才是判据
    dup = [(a, c) for a, c in ans_counter.items() if c > 1]
    dup_total = sum(c for _, c in dup)
    print(f"\n[多样性] 唯一答案 {len(ans_counter)} / {len(rows)} 条"
          f"   重复答案占用 {dup_total} 条 ({dup_total / len(rows) * 100:.1f}%)")
    if dup:
        print("    重复最多的 5 条：")
        for a, c in sorted(dup, key=lambda x: -x[1])[:5]:
            print(f"      ×{c:<4} {a[:50]}")

    n_ctx, n_plain = sum(ans_ctx.values()), sum(ans_plain.values())
    n_rag = sum(ans_rag.values())
    dup_ctx = sum(c for c in ans_ctx.values() if c > 1)
    dup_plain = sum(c for c in ans_plain.values() if c > 1)
    dup_rag = sum(c for c in ans_rag.values() if c > 1)
    ctx_ratio = dup_ctx / n_ctx * 100 if n_ctx else 0.0
    plain_ratio = dup_plain / n_plain * 100 if n_plain else 0.0
    rag_ratio = dup_rag / n_rag * 100 if n_rag else 0.0
    print(f"    ├ 带资料 {n_ctx} 条：唯一 {len(ans_ctx)}，重复占用 {dup_ctx}"
          f" ({ctx_ratio:.1f}%)   ← 含对抗样本的故意重复，仅供参考")
    print(f"    │  └ 引用资料 {n_rag} 条：唯一 {len(ans_rag)}，重复占用 {dup_rag}"
          f" ({rag_ratio:.1f}%)   ← 真正的判据")
    print(f"    └ 无资料 {n_plain} 条：唯一 {len(ans_plain)}，重复占用 {dup_plain}"
          f" ({plain_ratio:.1f}%)   ← 固定话术重复属正常")
    dup_rag_items = sorted(((a, c) for a, c in ans_rag.items() if c > 1),
                           key=lambda x: -x[1])[:3]
    if dup_rag_items:
        print("      引用资料群重复最多的 3 条：")
        for a, c in dup_rag_items:
            print(f"        ×{c:<4} {a[:50]}")

    # 配比对账
    if os.path.exists(STATS_PATH):
        st = load_json(STATS_PATH)
        print(f"\n[配比] stats.json total={st.get('total')}  "
              f"jsonl 行数={len(rows)}  total_target={comp.get('total_target')}")
        if st.get("total") != len(rows):
            err(f"stats.json 记录的 {st.get('total')} 条与 jsonl 实际 {len(rows)} 条不符"
                f" —— 数据集在生成后被改动过，或写入中断")
        zero = [k for k, v in (st.get("types") or {}).items() if v == 0]
        if zero:
            err(f"以下类别产出 0 条（很可能是检索空库或生成器异常）: {zero}")
        for k, v in (st.get("types") or {}).items():
            print(f"    {k:<20}{v:>6}  ({v / max(1, len(rows)) * 100:.1f}%)")
    else:
        warn(f"找不到 {STATS_PATH}，无法对账配比 —— 请用当前版本的 build_samples.py 重新生成")

    print("\n" + "=" * 78)
    dump("ERROR 结构异常", bad_struct, lambda x: f"第{x[0]}行 {x[1]}")
    dump("ERROR system 无法用生产 build_sysp 重建", no_owner,
         lambda x: f"第{x[0]}行 {x[1]}...")
    dump("ERROR 上下文形态非法", bad_ctx, lambda x: f"第{x[0]}行 {x[1]}")
    dump("ERROR 答案为空中断", short_ans, lambda x: f"第{x[0]}行 {x[1]!r}")
    dump("ERROR 答案含 AI 自认", ai_hit, lambda x: f"第{x[0]}行 {x[1]}")
    dump("ERROR 答案含设定文档排版语法", struct_ans,
         lambda x: f"第{x[0]}行 [{x[1]}] {x[2]}")
    dump("ERROR 注入资料含排版语法（与生产 build_context 不一致）", struct_ctx,
         lambda x: f"第{x[0]}行 [{x[1]}] {x[2]}")
    dump("ERROR 答案含剧本/策划元数据（NPC 承认自己在游戏里）", meta_ans,
         lambda x: f"第{x[0]}行 [{x[1]}] {x[2]}")
    dump("ERROR 注入资料含剧本/策划元数据（clean_kb_text 漏网）", meta_ctx,
         lambda x: f"第{x[0]}行 [{x[1]}] {x[2]}")
    dump("ERROR 答案含孤立闭合符（分块边界切断引号/括号）", orphan_ans,
         lambda x: f"第{x[0]}行 落单的 {x[1]!r} → {x[2]!r}")
    dump("ERROR 注入资料含孤立闭合符", orphan_ctx,
         lambda x: f"第{x[0]}行 落单的 {x[1]!r} → {x[2]!r}")
    dump("WARN 答案不以句末标点收尾（疑似残句）", bad_end, lambda x: f"第{x[0]}行 ...{x[1]}")
    dump("WARN 答案含异常标点串", bad_punct, lambda x: f"第{x[0]}行 {x[1]!r} → {x[2]}")
    dump("WARN 答案含括号旁白", narr, lambda x: f"第{x[0]}行 {x[1]}")
    dump(f"WARN 序列超过 max_length={max_length}（会被截断）", over,
         lambda x: f"第{x[0]}行 {x[1]} tokens")

    # 阈值型告警：不到 ERROR 但会影响训练质量
    if bad_end and len(bad_end) > len(rows) * 0.05:
        err(f"残句占比 {len(bad_end) / len(rows) * 100:.1f}% 超过 5%，"
            f"colloquialize 的句子边界切分有问题")
    # 只对「引用资料」群告警。带资料群会把对抗样本的固定回避话术算进来（实测
    # 它们占了那群重复 top3），全局阈值则连无资料群的话术也算进来 —— 两种
    # 口径都会把这个判据长期停在「偏高但不致命」的模糊地带，看不出改写有没有塌缩。
    if n_rag and dup_rag > n_rag * 0.60:
        warn(f"引用资料的样本逐字重复 {rag_ratio:.1f}%（{dup_rag}/{n_rag}）偏高，"
             f"口语化改写多样性不足，模型会把资料答案背成固定几段")
    if over and len(over) > len(rows) * 0.02:
        err(f"超长样本占比 {len(over) / len(rows) * 100:.1f}% 超过 2%，"
            f"需提高 max_length 或缩短注入资料")

    print("\n" + "=" * 78)
    hard = (len(errors) + len(bad_struct) + len(no_owner) + len(bad_ctx)
            + len(short_ans) + len(ai_hit) + len(struct_ans) + len(struct_ctx)
            + len(meta_ans) + len(meta_ctx)
            + len(orphan_ans) + len(orphan_ctx))
    n_warn = len(bad_end) + len(bad_punct) + len(narr) + len(over) + len(warns)
    print(f"[结论] ERROR {hard} 项   WARN {n_warn} 项")
    if errors:
        print("[ERROR 明细]")
        for e in errors:
            print(f"    · {e}")
    # warns 必须逐条打印。只报「WARN 2 项」而不说是什么，等于没报 ——
    # 而且其中可能正藏着「长度校验根本没跑」这种让结论失效的信息。
    if warns:
        print("[WARN 明细]")
        for w in warns:
            print(f"    · {w}")

    if hard:
        print("\n       ✗ 存在硬性问题，**不要启动训练**。修完重跑本脚本。")
        return 1
    if not cli.no_tokenizer and not len_checked:
        # 防假通过：长度校验是拦住「样本被静默截断」的唯一手段。它没跑成
        # 就不能声称数据集可用 —— 否则本脚本自己就成了假通过的源头。
        print("\n       ⚠ 序列长度校验未成功执行，结论不完整，**不能视为通过**。")
        print("         修好 tokenizer 加载后重跑，或用 --no-tokenizer 显式接受风险。")
        return 1
    print("\n       ✓ 可以启动训练。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
