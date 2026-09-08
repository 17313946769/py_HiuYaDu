# -*- coding: utf-8 -*-
"""LangChain RAG 对话链：ChromaDB 检索 + 双档阈值 + 双层过滤 + LLM 生成

架构：
  玩家提问 -> 直连判定(DIRECT_LLM_HINTS / 探问他人) -> 向量检索(ChromaDB)
  -> 真实相关度分档 -> 黑名单 + 他人专属知识双层过滤
  -> 强参考/弱参考/无命中回退 -> 注入 system prompt -> LLM 生成

本文件同时是「上下文注入格式」的唯一权威定义（format_context / build_sysp）：
finetune/scripts/build_samples.py 生成训练样本时必须从本文件导入这两个函数，
否则会出现「模型学的提示词格式线上永不出现」的训推不一致。

LLM: Qwen2.5-3B-Instruct (D:\\Model, 8-bit GPU + CPU offload)
嵌入: paraphrase-multilingual-MiniLM-L12-v2 (CPU)
向量库: ChromaDB (rag/chroma_db)

用法：
  from rag.rag_chain import HuiYaDuChain
  chain = HuiYaDuChain(npc_id="AY")
  ans = chain.ask("断水堂是什么来头？")
"""
import json
import os
import re
import sys
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# ============================================================
# 配置
# ============================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = r"D:\Model"
NPC_CONFIG = os.path.join(ROOT, "npc_config.json")
CHROMA_DIR = os.path.join(ROOT, "rag", "chroma_db")

EMBED_MODEL = r"D:\Model\embedding"
# 阈值单位是余弦相似度 [0,1]，由 distance_to_similarity 从 ChromaDB 的 l2 距离换算。
# 下面两个值是 calibrate_threshold.py 在 paraphrase-multilingual-MiniLM-L12-v2 +
# 当前 chroma_db 上的实测结果（151 条训练语料同源查询，k=3）：
#   域内 RAG 问句   p25=0.543  中位=0.664  max=0.869
#   域内实体名/头衔  p25=0.609  中位=0.719  max=0.885
#   离题            p75=0.331  max=0.520
#   良性闲聊         p75=0.278  max=0.692
#   跨角色探问       min=0.544  中位=0.697  max=0.786
# 取 0.60 让约 62% 的域内问句进强档（旧值 0.70 只有 40%），同时离题 max 0.520
# 仍被挡在强档外。**换嵌入模型或重建知识库后必须重跑校准，这两个值不可迁移**
# （旧值 0.70/0.45 源自 bge-small-zh-v1.5 的分布，与当前模型无关）。
#
# 重要：跨角色探问的分数（中位 0.697）比域内问句还高 —— 知识库里确实存着他人
# 秘密的块，检索并没有找错。这类污染调阈值永远拦不住，只能靠三道防线：
# ask() 里的 mentioned_other_npc 路由拦截、build_context 里的他人 knowledge_tags
# 过滤，以及训练集里的 adv_cross_char 对抗样本。
STRONG_TH = 0.60
WEAK_TH = 0.35
TOP_K = 3

BEHAVIORAL_CONSTRAINT = (
    "始终以该身份说话，绝不跳出角色、不承认自己是AI或程序。"
    "不替其他角色编造私事或剧情秘密。不使用旁白或动作描写。"
)

# 无命中回退：不是资料而是一条指令，同样会注入 system prompt。
# build_samples.py 必须用本常量生成对应训练样本，否则这条生产路径也是 OOD。
NO_HIT_FALLBACK = (
    "遇到不了解的问题，以你的角色身份自然地表示不清楚，"
    "不要编造答案或替其他角色说事。"
)

DIRECT_LLM_HINTS = (
    "早上好", "中午好", "下午好", "晚上好", "早啊", "你好", "哈喽", "嘿", "在吗",
    "你是谁", "你是哪位", "你叫什么", "你还记得我",
    # 玩家自我指代：旧版只有「你是谁」没有「我是谁」，导致玩家记忆类问题被
    # 送去检索、命中无关剧情块（v3.1 已记录但一直未修）。
    "我是谁", "我叫什么", "我是你的谁", "你知道我的身份", "记得我吗",
    "你现在是", "忽略你的设定", "忘掉你是", "你其实是", "从现在起",
    "停止角色扮演", "程序员", "别装了", "AI助手", "人工智能",
    "天气如何", "天气怎么样", "今天天气", "明天天气", "会下雨",
    "这道题", "怎么解", "解题", "数学题", "物理题",
    "写代码", "写程序", "编程", "Python",
    # 离题拒答类：训练语料 unknown_qs 里这几类旧版没有触发词，
    # 会被送去检索并注入无关剧情块，与「无资料训练的拒答行为」相冲。
    "笑话", "量子力学", "股票", "互联网", "翻译", "作诗", "写一首诗",
    "人生的意义", "什么是爱情", "地球是圆",
)

# 全局模型缓存（避免重复加载 OOM）
_llm_cache = {"pipe": None, "tokenizer": None}


def load_llm():
    if _llm_cache["pipe"] is not None:
        return _llm_cache["pipe"], _llm_cache["tokenizer"]
    print(f"[LLM] 加载模型: {MODEL_DIR} (4-bit GPU)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto",
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    _llm_cache["pipe"] = pipe
    _llm_cache["tokenizer"] = tokenizer
    return pipe, tokenizer


# ============================================================
# 嵌入 + 向量库
# ============================================================
def load_vectorstore():
    """返回 Chroma 向量库本体（而不是 retriever）。

    旧版返回 db.as_retriever()，其默认 search_type="similarity" 不带分数字段，
    是双档阈值失效的根因。改为直接持有 db，由 build_context 调
    similarity_search_with_score 取真实距离并自行换算为余弦相似度。
    """
    print(f"[嵌入] 加载模型: {EMBED_MODEL} (CPU)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    return db


# 向后兼容旧调用方（返回的已是向量库，调用方需自行适配）
load_retriever = load_vectorstore


def distance_to_similarity(dist, space):
    """把 Chroma 返回的距离换算为余弦相似度 [0,1]。

    嵌入已归一化（normalize_embeddings=True），因此：
      l2     : d^2 = 2(1 - cos)  ->  cos = 1 - d^2/2
      cosine : d = 1 - cos       ->  cos = 1 - d
      ip     : d = -cos          ->  cos = -d
    不用 LangChain 的 similarity_search_with_relevance_scores，避开它对不同
    度量空间归一化方式不一致、以及分数越界时报错的不确定性。
    """
    if space == "cosine":
        return max(0.0, min(1.0, 1.0 - dist))
    if space == "ip":
        return max(0.0, min(1.0, -dist))
    return max(0.0, min(1.0, 1.0 - (dist ** 2) / 2.0))


def vectorstore_space(db):
    """读出集合的距离度量（Chroma 默认 l2）"""
    try:
        meta = db._collection.metadata or {}
        return str(meta.get("hnsw:space", "l2")).lower()
    except Exception:
        return "l2"


# ============================================================
# 知识块清洗
# ============================================================
# 「知识图谱」与「实体关系设定」两份文档大量使用结构化标记：
#   [秀芝] --(夫妻)--> [马良]
#   │ ─ 今日消息 (3) ────── │
#   铁匠铺 | 生产场所 | 马良经营。
# 这些是设定文档的排版语法，**不是台词**。直接注入上下文会让模型把它
# 当参考资料复述出来（NPC 对玩家说「哼，[收费争议] ----> [行会把控码头收费…」），
# 而且这种块的信息密度极低（一行只表达一个三元组），当资料也不划算。
TRIPLE_RE = re.compile(r"\[[^\[\]]{1,24}\]\s*--+")
BOX_RE = re.compile(r"[│┌┐└┘├┤─━]{2,}")
# 「知识图谱」里还有一份 JSON 形式的三元组（{"head": …, "relation": …, "tail": …}）。
# TRIPLE_RE 只认 [A] --> [B] 的方括号写法，抓不到这种，于是整块 JSON 会被当成
# 资料注入 —— 模型读到的是 {"head": "陈伯", "relation": "解锁条件",
# "tail": "对话16轮+世界状态S4"}，既是数据结构又全是策划元数据。
JSON_TRIPLE_RE = re.compile(r'\{\s*"head"')
# 关系图/状态流转还有一种写法：箭头前面不是 ] 而是换行或缩进，例如
#   将FY的base_frequency从0恢复为1.5。
#    -->
# TRIPLE_RE 要求 [X] -- 的前缀，匹配不到这种。裸箭头本身就足以定性 ——
# 正常叙述里不会出现「-->」，只有状态流转图、伪代码和配置片段会用它。
ARROW_RE = re.compile(r"--+>")
# 清单式块：策划的功能清单 / MVP 裁剪清单，形如
#   时间系数/事件系数/对话深度系数/世界状态系数
#   衰减/冷却/混合状态细分
#   多存档/自动存档
# 这类行的共同结构是「短、无句末标点、用 / + | 分隔并列项」。逐个枚举关键词永远
# 追不完（每份文档的用词都不同），改用结构判据一次拦掉。
LIST_SEP_RE = re.compile(r"[/+|]")


def is_structured_block(text):
    """判定知识块是否以排版语法为主，这类块不得用作参考资料或台词素材。"""
    if not text or not text.strip():
        return True
    if len(TRIPLE_RE.findall(text)) >= 2:      # 两个以上三元组 = 关系图
        return True
    if JSON_TRIPLE_RE.search(text):            # JSON 形式的三元组，一条就够定性
        return True
    if ARROW_RE.search(text):                  # 裸箭头：状态流转图 / 伪代码
        return True
    if BOX_RE.search(text):                    # 制表符画出来的框线/分隔线
        return True
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:
        if line.count("|") >= 3:               # Markdown / docx 表格行
            return True
    # 清单式块。要求行里带并列分隔符，避免误伤剧本对白（「阿茵：…」这种行
    # 同样短、同样没有句号，但那是有价值的台词素材，不能当清单丢掉）。
    if len(lines) >= 4:
        frag = sum(1 for l in lines
                   if len(l) <= 30 and LIST_SEP_RE.search(l)
                   and not re.search(r"[。！？]", l))
        if frag / len(lines) >= 0.6:
            return True
    return False


# 「剧情文本」「游戏设计规格」「知识图谱」三份文档里混着大量**写给策划和程序**的
# 元数据。它们不含排版语法，所以 is_structured_block 抓不到，但让 NPC 照读出来
# 的穿帮程度更严重 —— 等于当着玩家的面承认自己活在游戏里：
#   「唉，触发条件： 玩家在药铺遇见秀芝，未选择"问她身体"…」
#   「MSG_INITIATE — 话题发起 — 系统提示条（黄色） — 优先级最高」
#   「第二层：语料检索器 / 评分维度：路线匹配（+1~5）…」← 在描述本项目自己
# 实测这类句子占存活块总句数的 17.3%，但**只有 4 块是纯元数据**，其余都与有用
# 事实混排。所以必须逐句清洗而不是整块拒收：整块拒收会连带丢掉「三十年前渡口
# 易主案 — 商盟买通断水堂刺杀原渡口主人，陈伯协助伪造易主文件，无名（遗孤）
# 幸存」这种核心真相（它只因后半句带「A4节点被玩家找到」就被牵连）。
# 逐句清洗的实测代价：可用块 226→222，各 NPC 覆盖零损失，元数据残留 0。
META_SENT_RES = (
    re.compile(r"(MSG|INT)_[A-Z_]+"),
    re.compile(r"System\s*Prompt|State\s*Machine|RAG\s*触发|语料检索器|"
               r"回复选择器|评分维度|兜底回复|分层注入|save_data|\.json", re.I),
    re.compile(r"触发条件"),
    # 游戏机制术语一律视为元数据，**不要求后面跟比较符**。旧写法卡了 [≥≤<>]，
    # 于是「核心NPC的关系值通过玩家的互动累积」「需积累对话深度」「通过好感度
    # 控制解锁」全部漏网（实测 27 块）。NPC 嘴里不该出现这些词，无条件删更安全。
    re.compile(r"关系值|亲和度|好感度|对话深度|行动点|行动力|世界状态|状态机|"
               r"阈值|概率\s*=|系数"),
    re.compile(r"[A-Z]\d+\s*节点|[A-Z]\d?\s*分支|R\d+\s*路线|[A-Z]链|RP值|"
               r"S\d\s*[（(→]|S\d\s*世界|事件链"),
    re.compile(r"消耗\s*\d+\s*点"),
    re.compile(r"玩家.{0,8}(选择|提到|触发|累计|达成|解锁|输入|完成)|"
               r"IF\s*玩家|→\s*AI"),
    # 策划文档的骨架词：功能清单、验收用例、版本批注。实测漏网的块基本都带这些。
    re.compile(r"\S{2,6}系统\s*[:：]|MVP|见证录|活跃层级|路线倾向值|关系网|"
               r"文档版本|修正说明|性格底线速查|通用回复策略"),
    re.compile(r"操作\s*[:：]|预期\s*[:：]|[✅❌]|JSON格式"),
    # 文档骨架：md 章节标题、编号小节、tab 分隔的表格行、版本批注。
    # build_kb.py 的 chunk_markdown 只把「以 # 开头」的行当标题剥掉，于是
    # 「十二、开发实施路线图与风险预演」「12.1 实施路线图」这类中文数字 / 编号
    # 标题全部作为正文进了块；游戏设计规格.md 里还有一批用 tab 分隔的表
    # （NPC 参数表「CS 翠婶 洗衣妇 30 3.5 0.2 0.6 0.9」、关系数值表
    # 「HM AB 上下级 70」），is_structured_block 数的是竖线，抓不到 tab。
    # 实测这类块占清洗后可用块的 25.8%（47/182），数据集里 40 条答案在照念。
    # tab 行走**句子级**而不是整块拒收：同一块里常混着核心叙述（「2.3 三大
    # 势力 / 行会：控制渡口收费权，暴力执行者阿彪」「2.4 150年历史沿革 /
    # 三十年前渡口易主」），整块丢掉会连带丢掉真相。
    re.compile(r"^[一二三四五六七八九十]{1,3}[、.]\s*\S"),
    re.compile(r"^\d+\.\d+\s*\S"),
    re.compile(r"【[^】]{0,8}(新增|修订|修正|删除|补充)[^】]{0,8}】"),
    re.compile(r"\t"),
    re.compile(r"实施路线图|风险预演|里程碑|交付物|防御性编程|状态访问器"),
    # UI 操作指引：「在河边与陈伯对话，选择“提起船运记录的事”」。上面的
    # 「玩家.{0,8}选择」要求句里有「玩家」，这句没写主语（策划默认读者就是
    # 玩家），于是漏网。判据改用「选择 + 紧跟引号」：选项文本总是被引号包着，
    # 而正常叙述里「选择」后面不会直接接引号。
    re.compile(r"选择\s*[\"“「『]"),
    # 箭头图：区域邻接「酒馆 ↔ 街上 ↔ 归雁楼」、流程「探索期 → 深入期 →
    # 收束期」、状态流转「沉睡 → 半活跃」。LIST_SEP_RE 只认 [/+|]，抓不到这些；
    # NPC 台词里也不会出现任何箭头，所以一个字符就够定性。
    re.compile(r"[↔→←⇒⇄]"),
    # 叙事学术语与游戏阶段名。策划写「伏笔铺垫·第三层」「深入期内，你与陈伯
    # 持续对话，可逐步获得日记的残片」是在讲**这段剧情怎么设计**，NPC 照念
    # 等于在解说自己的剧本。末尾那个孤立的「）」由 drop_orphan_closers 收拾，
    # 术语本身得在这里删。
    re.compile(r"伏笔|铺垫|渐进收集|残页收集|探索期|深入期|收束期|"
               r"·第[一二三四五六七八九十\d]+(层|阶段|幕)"),
    # 表格行残渣。is_structured_block 的门槛是一行 3 个竖线，而 docx 表格转出的
    # 「键 | 值」很多只有 2 个（「铁匠铺 | 生产场所 | 马良经营。」），于是漏网。
    # 不能把门槛降到 2 —— 那样会整块拒收，连带丢掉同一块里的核心事实
    # （「林姐常来此修理兵器与公文铁匣，暗生情愫」）。改走句子级：按句号切分后
    # 带竖线的那一句被删，叙述句保留。中文台词里不会存在竖线，所以一个字符定性。
    re.compile(r"\|"),
    # 结局与分支的设计说明：「收束分支：5个分支结局」「核心当事人：秀芝、马良」。
    # 上面的 [A-Z]\d?分支 / R\d+路线 只认字母编号，中文写法全部漏网。NPC 谈论
    # 自己有几条结局分支、谁是「核心当事人」，等于把剧本结构摊在玩家面前。
    re.compile(r"分支结局|结局分支|\d+\s*个分支|收束分支|核心当事人|"
               r"当事人\s*[:：]|结局\s*[:：]|分支\s*[:：]"),
)

# 剧本的舞台指示标签。这类标签后面**跟着有用的描写**（「叙事： 秀芝脸色苍白，
# 接过药包的手在抖。」），所以只剥标签、不删句子 —— 与上面整句丢弃的区别在于
# 内容是 NPC 真能说的话。
LABEL_PREFIX_RE = re.compile(r"^(叙事|对白|旁白|场景|舞台指示|描述)\s*[:：]\s*")

# NPC 视角里没有「玩家」，只有「你」。META_SENT_RES 已经丢掉了系统视角的句子，
# 剩下含「玩家」的都是叙事主体（实测 115 句可安全替换，只有 8 句仍带技术词）。
# 那 8 句必须整句丢，不能替换 —— 把「玩家提到特定名词时，系统检索深层关系」改成
# 「你提到…系统检索…」比原句更糟：NPC 在讲自己的检索实现。
META_TECH_WORDS = ("系统", "检索", "注入", "语料", "状态机", "节点", "阈值",
                   "路线", "分支", "解锁", "Prompt", "AI", "概率", "评分")

SENT_SPLIT_RE = re.compile(r"(?<=[。！？；\n])")


def strip_meta_script(text):
    """逐句丢掉剧本/系统元数据，并把叙事里的「玩家」换成 NPC 视角的「你」。"""
    kept = []
    for s in SENT_SPLIT_RE.split(text or ""):
        if not s:
            continue
        if (not s.strip()
                or any(rx.search(s) for rx in META_SENT_RES)
                or ("玩家" in s and any(w in s for w in META_TECH_WORDS))):
            # 丢内容但**保留换行**。知识库里的表格行靠 \n 分隔（build_kb.py 把
            # docx 表格逐行转成「键 | 值」再用 \n 拼接），连 \n 一起删会让相邻行
            # 合并成一行，竖线累加到 3 个以上，于是好端端的事实块被
            # is_structured_block 和校验器判成表格行 —— 实测造成 157 条误报。
            kept.append("\n" if "\n" in s else "")
            continue
        kept.append(LABEL_PREFIX_RE.sub("", s).replace("玩家", "你"))
    return "".join(kept)


# 闭合符 → 对应的开符。build_kb.py 用 500 字滑窗（CHUNK_OVERLAP=100）切块，
# 切点会落在引号或括号中间：上一块拿走了「（“」，这一块只剩孤立的「）”。」。
# 去括号正则 [（(][^）)]*[）)] 要求成对，删不掉这种残渣，于是它直接进了答案 ——
# 实测出现「少废话，”」以及整行只有「”）」「”）。」的样本。
CLOSER_TO_OPENER = {"”": "“", "’": "‘", "」": "「", "』": "『",
                    "）": "（", ")": "(", "》": "《", "〕": "〔"}
OPENERS = frozenset(CLOSER_TO_OPENER.values())


def drop_orphan_closers(text):
    """逐字符做括号/引号配对，丢掉没有开符的闭合符。

    只删闭合符、不删开符：开符落单（「“可我当时」被切走了下半截）读起来仍像
    正常句子的开头，而孤立的「”）」是纯排版垃圾，模型学会就会在台词里吐符号。
    """
    depth = {}
    out = []
    for ch in text or "":
        op = CLOSER_TO_OPENER.get(ch)
        if op is None:
            if ch in OPENERS:
                depth[ch] = depth.get(ch, 0) + 1
            out.append(ch)
        elif depth.get(op, 0) > 0:
            depth[op] -= 1
            out.append(ch)
    return "".join(out)


def clean_kb_text(text):
    """清洗知识块：字符级排版残留 + 句子级剧本元数据。

    与 is_structured_block 分工：后者整块拒收（三元组关系图 / 框线 / 表格行），
    本函数处理幸存块里的两类残渣 ——
      · 字符级：\xa0 不换行空格、\\* 转义星号、<!-- --> 修正注释、括号旁白
      · 句子级：写给策划/程序的剧本元数据（见 META_SENT_RES）
    训练与生产都必须走本函数：build_context 注入的资料、build_samples 生成的
    答案，用的都得是清洗后的文本，否则两侧会学到不同的东西。
    """
    t = text or ""
    t = t.replace("\xa0", " ")
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    # 句子级清洗必须**排在去括号之前**：策划常把系统标记写在括号里
    # （「常驻背景（System Prompt）」「动态检索（RAG触发）」），先去括号的话
    # 标记就没了，正则匹配不到，整句会被当成正常资料留下来（实测漏网）。
    t = strip_meta_script(t)
    t = t.replace("\\*", "").replace("\\[", "").replace("\\]", "")
    t = re.sub(r"[（(][^）)]*[）)]", "", t)
    # 分块边界切断的孤立闭合符（见 drop_orphan_closers）。必须排在去括号之后：
    # 成对的括号先整对删掉，剩下的才是真正落单的。
    t = drop_orphan_closers(t)
    # 逐句清洗与删符号都会留下空行（strip_meta_script 故意保留 \n 以避免相邻
    # 表格行合并），这里只压空行、不删单个 \n，所以不会把两行内容拼到一块。
    t = re.sub(r"\n{2,}", "\n", t)
    # 去括号会留下悬挂标点：「阿茵知情，（系统反馈：…）」→「阿茵知情，」，
    # 再拼上以逗号开头的口语收尾就成了「，，」。实测 6225 条里出现过 3 次。
    t = re.sub(r"[，、；：]{2,}", "，", t)
    t = re.sub(r"[，、；：]+(?=[。！？；])", "", t)
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    # 末尾悬挂的顿逗也要去掉，且必须在 strip 之后判定（原文可能是「阿茵知情，\n」，
    # 先去空白才看得见真正的末尾字符）。训练侧 gen_rag_paraphrase 会把首句直接接在
    # 以逗号开头的收尾语前面，残留逗号就拼成「，，」；生产侧虽不做这种拼接，但残
    # 标点作为「参考资料」注入，同样会让模型学到断裂的句读。
    return re.sub(r"[，、；：]+$", "", t).strip()


# ============================================================
# NPC 配置
# ============================================================
def load_npcs():
    return {n["id"]: n for n in json.load(open(NPC_CONFIG, encoding="utf-8"))["npcs"]}


def build_sysp(npc, extra=""):
    """拼接角色 system prompt。extra 用于注入检索上下文。

    连接符必须是 "\n\n"：旧版用单个空格，而 HuiYaDuChain.ask() 用 "\n\n"，
    造成训练样本与线上推理的 system prompt 不一致。
    """
    base = (f"你是{npc['name']}，{npc['title']}。"
            f"性格{'、'.join(npc['personality_tags'])}。"
            f"你熟悉{'、'.join(npc['knowledge_tags'])}。"
            f"你绝口不提{'、'.join(npc['forbidden_tags'])}。"
            f"{BEHAVIORAL_CONSTRAINT}")
    if extra:
        base += "\n\n" + extra
    return base


def format_context(strong, weak):
    """把分档后的资料块拼成注入字符串。

    训练与推理共用的唯一格式来源。build_samples.py 生成 RAG 类样本时必须
    调用本函数，不得自己拼字符串。
    """
    parts = []
    if strong:
        parts.append("参考资料（优先依据以下资料回答，不要编造资料以外的内容）：")
        for idx, t in enumerate(strong, 1):
            parts.append(f"（{idx}）{t}")
    if weak:
        parts.append("以下仅为模糊背景，不确定就巧妙回避，不要编造他人私事或剧情秘密：")
        for t in weak:
            parts.append(f"·{t}")
    return "\n".join(parts)


def build_secret_owners(npcs):
    """tag -> 拥有该知识的 NPC id 集合，用于拦截「他人专属知识」"""
    owners = {}
    for nid, n in npcs.items():
        for tag in list(n.get("knowledge_tags", [])) + list(n.get("forbidden_tags", [])):
            if tag:
                owners.setdefault(tag, set()).add(nid)
    return owners


def mentioned_other_npc(question, npc, npcs):
    """问题里点名了别的角色 -> 属探问他人，返回该角色名；否则 None"""
    if not npcs:
        return None
    for nid, other in npcs.items():
        if nid == npc["id"]:
            continue
        name = other.get("name")
        if name and name in question:
            return name
    return None


# ============================================================
# 检索上下文构建（双档阈值 + 双层过滤）
# ============================================================
def build_context(db, question, npc, all_npcs=None, affinity_stage=""):
    """检索并分档，返回注入 system prompt 的字符串。

    与旧版的五处关键差异：
      1. 取真实相关度分数。旧版走 db.as_retriever() 的默认 similarity 检索，返回的
         Document 里没有分数字段，而 build_kb.py 写入的 metadata 也只有
         {id, source, section}，于是 score = 1 - 0.5 = 0.5 恒定：strong 档永不
         触发、weak 档永远触发，双档阈值形同虚设。
      2. 除当前 NPC 的 forbidden_tags 黑名单外，再按「他人专属知识」过滤：
         资料块若命中了不属于当前 NPC 的 knowledge_tags / forbidden_tags，一律丢弃。
         旧版缺这一层，问陈伯「秀芝怎么了」会把秀芝怀孕的块喂进他的上下文。
      3. 拼接格式抽到 format_context()，与训练数据生成共用同一实现。
      4. 拒收结构化块（三元组关系图 / 框线 / 表格行）—— 那些是设定文档的
         排版语法，不是能复述的资料。
      5. 注入的资料一律经 clean_kb_text 清洗。旧版把 doc.page_content 原样塞进
         「参考资料」，于是模型能读到「MSG_INITIATE — 系统提示条（黄色）」「可在
         B4节点触发"中立"建议」「触发条件：玩家在药铺遇见秀芝…」这类写给
         策划的元数据，并照着说出来 —— 等于当玩家的面承认自己活在游戏里。
    """
    space = vectorstore_space(db)
    hits = db.similarity_search_with_score(question, k=TOP_K)
    if not hits:
        return NO_HIT_FALLBACK

    forbidden = set(npc.get("forbidden_tags", []))
    owners = build_secret_owners(all_npcs) if all_npcs else {}
    strong, weak = [], []

    for doc, dist in hits:
        raw = doc.page_content
        score = distance_to_similarity(float(dist), space)

        # 第零层：排版语法为主的块直接丢，最便宜的检查放最前面
        if is_structured_block(raw):
            continue
        # 第一、二层用**原始**文本判定：清洗可能正好删掉含禁忌词或他人 tag 的
        # 那一句，用清洗后的文本判定等于放宽安全边界。宁可多拦一块。
        # 第一层：黑名单——当前角色的禁忌内容
        if forbidden and any(w in raw for w in forbidden):
            continue
        # 第二层：他人专属知识——只属于别的角色的 tag 出现在块里
        if owners:
            alien = [tag for tag, own in owners.items()
                     if tag in raw and npc["id"] not in own]
            if alien:
                continue
        # 第三层：清洗剧本元数据。清洗后不足 20 字，说明这块基本全是元数据、
        # 没有可引用的事实（实测 226 块里只有 4 块会走到这一步）。
        text = clean_kb_text(raw)
        if len(text) < 20:
            continue

        if score >= STRONG_TH:
            strong.append(text)
        elif score >= WEAK_TH:
            weak.append(text)

    if not strong and not weak:
        return NO_HIT_FALLBACK
    return format_context(strong, weak)


# ============================================================
# 对话链
# ============================================================
class HuiYaDuChain:
    def __init__(self, npc_id="AY", extra=""):
        self.npcs = load_npcs()
        if npc_id not in self.npcs:
            raise ValueError(f"未知 NPC: {npc_id}，可选: {list(self.npcs.keys())}")
        self.npc = self.npcs[npc_id]
        self.extra = extra
        self.sysp = build_sysp(self.npc, extra)

        print(f"[链] 初始化: {self.npc['name']} ({npc_id})")
        self.pipe, self.tokenizer = load_llm()
        self.db = load_vectorstore()
        print(f"[链] 就绪")

    def ask(self, question):
        if any(h in question for h in DIRECT_LLM_HINTS):
            ctx = ""
            src = "LLM"
        elif mentioned_other_npc(question, self.npc, self.npcs):
            # 探问他人：不注入任何资料，交给模型的回避能力。
            # 注入他人资料会与训练信号正面冲突：RAG 样本教「优先依据资料回答」，
            # 跨角色样本教「别人的事不好说」，两者同时触发时前者样本量占优。
            ctx = ""
            src = "LLM(回避)"
        else:
            ctx = build_context(self.db, question, self.npc, self.npcs)
            src = "RAG" if ctx.startswith("参考资料") else "LLM"

        full_sysp = self.sysp
        if ctx:
            full_sysp = self.sysp + "\n\n" + ctx

        messages = [
            {"role": "system", "content": full_sysp},
            {"role": "user", "content": question},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        output = self.pipe(input_text)
        generated = output[0]["generated_text"]
        answer = generated[len(input_text):].strip()
        return answer, src
