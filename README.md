# 灰鸦渡 NPC 对话系统

为悬疑推理剧情《灰鸦渡》的 **15 个 NPC** 提供对话能力。玩家调查账房先生方远的失踪案，
需要在行会、西江商盟、巡卫队三方势力之间打探消息 —— 每个 NPC 只知道与自己身份相符的
那部分真相，且必须守住不属于自己的秘密。

技术路线：**RAG 检索增强 + QLoRA 独立微调**，基座 Qwen2.5-3B-Instruct，
在 **RTX 3050 Ti 4GB** 单卡上完成全流程训练（1146 步 / 15.0 小时）。

## 双目标

| 目标 | 含义 | 失败时的表现 |
|---|---|---|
| 人设对齐 | NPC 不越界、不承认自己是 AI、不输出代码、不加括号旁白、不泄漏他人专属秘密 | 「作为一个 AI 语言模型…」、「（皱眉）」、翠婶说出秀芝怀孕 |
| RAG 遵循 | 注入资料时按资料作答；无资料时走人设兜底而不是编造 | 检索到「三十年前渡口易主」却答「我不知道」；或凭空捏造剧情 |

两个目标存在张力：越强调"照资料说"，越容易把资料里他人的秘密也念出来。
本项目的解法是把**过滤做在检索侧、把回避做在训练侧**，并用对抗样本显式教后者。

## 目录结构

```
rag/
  build_kb.py             从 5 份策划文档构建 ChromaDB（500 字滑窗，overlap 100）
  rag_chain.py            生产链路。上下文注入格式与知识块清洗的唯一权威定义
  chroma_db/              向量库（已入库，见下方说明）
finetune/
  config/lora_config.json 全部超参与路径
  data/train.jsonl        训练集 6224 条（messages 三段式）
  data/seeds.json         few-shot 转述用的种子语料
  scripts/
    build_samples.py        生成训练集，提示词构造从 rag_chain 导入
    verify_dataset.py       训练前把关，8 类校验，发现 ERROR 返回 1
    calibrate_threshold.py  实测相似度分布，校准 STRONG_TH / WEAK_TH
    TEST.py                 QLoRA 速度基准（attention 实现 × 梯度检查点组合）
    train.py                QLoRA 训练
    merge_lora.py           合并 LoRA 到基座，输出 bf16 Safetensors
    test_finetune.py        38 条回归用例 + 路由审计
  logs/                   训练、合并、回归的完整日志与进度快照
npc_config.json           15 个 NPC 的人设、知识标签、禁忌标签、活动时段
test/mvp_test.py          LangChain 版对话入口（自由对话 / 自动回归）
```

`chroma_db/` **必须入库**：`build_kb.py` 读取的 5 份策划文档在仓库外的本地路径，
clone 后无法重建向量库。注意 ChromaDB 的 HNSW 索引正是以 `.bin` 存储的
（`data_level0.bin` / `header.bin` / `length.bin` / `link_lists.bin`），
`.gitignore` 里在排除权重文件的同时精确豁免了它们。

## 环境

Python 3.12，实测版本：

```
torch==2.13.0+cu130          transformers==5.15.0        peft==0.20.0
bitsandbytes==0.50.1         accelerate==1.14.0          safetensors==0.8.0
langchain==1.4.0             langchain-chroma==1.1.0     langchain-huggingface==1.2.2
chromadb==1.5.9              sentence-transformers==6.0.1
```

两处需要按本机改路径，都在 `finetune/config/lora_config.json`：

- `base_model` —— Qwen2.5-3B-Instruct 权重目录
- `data.embed_model` —— 嵌入模型目录（当前用 `paraphrase-multilingual-MiniLM-L12-v2`）

`rag/build_kb.py` 顶部的 `DOCS` 是 5 份策划文档的路径，仅在需要重建向量库时才用得到。

## 复现流程

```bash
# 0. （可选）重建向量库 —— 需要先准备好策划文档并改 DOCS 路径
python rag/build_kb.py

# 1. 校准检索阈值。嵌入模型换了之后阈值不可迁移：若 STRONG_TH 高于当前模型
#    实际能达到的最高分，强档分支就是死代码，生产永远只输出弱档格式，
#    而训练集用的是强档格式，训推直接错位。
python finetune/scripts/calibrate_threshold.py

# 2. 生成训练集（约 6224 条）
python finetune/scripts/build_samples.py

# 3. 校验。这一步不是"确认文件存在"，而是拦住会让 15 小时训练白跑的静默缺陷
python finetune/scripts/verify_dataset.py

# 4. 训练。先用 5 步冒烟验证全链路与速度，再跑全量
python finetune/scripts/train.py --smoke 5
python finetune/scripts/train.py --save-steps 100

# 5. 合并（纯 CPU，4GB 显存放不下 bf16 的 3.09B 参数；须在训练结束后运行）
python finetune/scripts/merge_lora.py

# 6. 回归
python finetune/scripts/test_finetune.py
```

日常对话用 `python test/mvp_test.py`（自由对话）或 `--auto`（自动回归）。

## 4GB 显存的工程要点

配置见 `lora_config.json`：4-bit NF4 + double quantization + bf16 compute、
LoRA r=16 / alpha=32 / dropout=0.05 / all-linear、`max_length=768`、
梯度检查点、`adamw_8bit`、batch 1 × 梯度累积 16、`attn_implementation="eager"`。

真正决定能否跑起来的是 **尾部 logits 优化**：`Qwen2ForCausalLM.forward` 默认对全部
L 个位置算 logits，内置 loss 会 `.float()` 上采样再 `.contiguous()` 拷贝，词表 151936
下单条 564-token 样本仅 logits 链路峰值就约 1.3GB。改为传 `logits_to_keep=k` 只对尾部
completion 段（平均 34 token）算 logits，省下约 1.2GB，并自行计算 CE 保证与内置 loss
数值等价 —— 启动时会做一次掩码校验，防止"loss 正常下降但学错位置"的静默错误。

配合 completion-only loss（prompt 段 labels 置 -100），模型只学 NPC 的回答，
不去拟合玩家提问与系统人设。

## 知识块清洗：三层架构

策划文档里混着大量**不是给人看的内容** —— 状态流转图、知识图谱三元组、NPC 参数表、
实施路线图、触发条件、分支结局设计说明。这些一旦进了训练集，NPC 就会当玩家的面
念出自己的剧本，穿帮程度远超普通幻觉。实测这类内容占清洗后可用块的 **25.8%**。

`rag_chain.py` 用三层处理，且层次顺序不可交换：

1. **`is_structured_block` 整块拒收** —— 三元组（`[A] -->` 与 `{"head"` 两种写法）、
   裸箭头、框线、一行 ≥3 竖线、清单式块结构
2. **禁忌词与他人专属 tag 过滤** —— 一律用**原文**判定。清洗可能正好删掉含禁忌词的
   那一句，用清洗后文本判定等于放宽安全边界
3. **`clean_kb_text` 句子级清洗** —— 剥离剧本元数据句 → 去转义 → 去括号 →
   删孤立闭合符 → 压空行 → 去悬挂标点

两条容易踩的顺序铁律：

- `strip_meta_script` 必须排在**去括号之前**，因为策划常把系统标记写在括号里
  （「常驻背景（System Prompt）」），先去括号的话标记就没了，整句会被当正常资料留下
- 删句子必须**保留 `\n`**。知识库表格行靠 `\n` 分隔，连 `\n` 一起删会让相邻行合并、
  竖线累加到 ≥3 被误判成表格行 —— 实测造成过 157 条误报

排版残渣（tab 表格行、2 竖线残渣）走**句子级**而不是整块拒收，因为同一块里常混着
核心叙述：`铁匠铺 | 生产场所 | 马良经营。林姐常来此修理兵器与公文铁匣，暗生情愫。`
整块丢掉会连带丢掉真相。

`verify_dataset.py` 的检测正则**独立编写**、刻意比 `rag_chain` 更宽，不复用其函数 ——
直接调被校验方的函数等于让它自己给自己打分，函数写错了校验器会跟着一起错。

## 当前成绩

**训练**：1146 步 / 3 epochs / 15.00 小时，trainable 29,933,568 / 3,115,872,256 = 0.96%。
三个 epoch 的 eval_loss `0.0886 → 0.0545 → 0.0493`，第 3 轮仍下降 9.5%，无过拟合。
数据集长度 min=101 / 中位=127 / p99=404 / max=652，全部落在 `max_length=768` 内无截断。

**回归**：38 条用例，**PASS 33 / REVIEW 1 / FAIL 4，严格通过率 86.8%**。

| 类别 | 条数 | PASS | REVIEW | FAIL |
|---|---|---|---|---|
| 身份自认 | 4 | 0 | 0 | **4** |
| 抗越狱（改身份 / AI 助手 / 写代码 / 停止扮演） | 7 | 7 | 0 | 0 |
| 离题拒答 | 6 | 6 | 0 | 0 |
| 良性闲聊不过度拒答 | 5 | 5 | 0 | 0 |
| 问候 | 2 | 2 | 0 | 0 |
| 跨角色污染（无资料，训练条件） | 4 | 4 | 0 | 0 |
| 跨角色污染（强行注入资料，生产条件） | 4 | 4 | 0 | 0 |
| 玩家记忆 | 2 | 2 | 0 | 0 |
| RAG 遵循 | 4 | 3 | 1 | 0 |

括号旁白 **0/38（0.0%）** —— 训练语料 0 旁白，稀释彻底生效。

唯一的 REVIEW 是「陈伯·RAG·渡口历史」：答案其实相当好（口语化复述了资料里的
「这渡口的风，三十年前就不正了」），但判据按 4-gram 重合度认定"与资料几乎无重合"。
口语化改写本身就要求降低字面重合，属判据偏严，非模型缺陷。

回归用例的问句**全部取自训练语料原句**，避免拿模型没学过的问法去考它造成伪失败；
判定分 FAIL / REVIEW / PASS 三级，不把"疑似"直接算通过；泄漏判定用**秘密内容词**
而非角色名（角色名回显是正常的）。

额外输出**路由审计**：生产链路 `rag_chain.ask()` 靠 `DIRECT_LLM_HINTS` 决定是否注入
检索资料，审计逐条比对训练条件与生产路由，本轮报出 **8 条不一致** —— 其中 4 条良性
问句是真缺陷（训练时无资料、生产却走 RAG，见遗留 3），另 4 条是探问他人秘密的问句
被生产侧 `mentioned_other_npc` 主动拦截直连 LLM，属防线生效的预期行为。

## 已知遗留

按影响排序，均为数据覆盖问题而非模型能力缺失：

1. **身份自认零覆盖（致命）** —— NPC 被问「你是谁」时答出**玩家**的身份
   （阿茵答「你是受雇来调查方远失踪案的侦探」）。根因：`PERSONA_QUESTIONS` 的
   27 个问句模板没有一个在问 NPC 身份，而 `PLAYER_QUESTIONS` 4 模板 × 75 = 300 条
   问玩家身份，形成 0 : 300 的单向覆盖。同一能力在越狱用例里正常（能正确说
   「我是阿茵，归雁楼的老板娘」），证明不是能力缺失而是触发问句不在训练分布内。
   需补 IDENTITY 样本类别后重训。
2. **占位符残渣** —— 无名（WM）的关系短语表是 `"……"` 占位符，拼进模板后产出
   `你是三天前刚到灰鸦渡的年轻旅人，……。` 这类垃圾样本；另有 223 条答案以省略号
   结尾，含纯符号答案。
3. **路由不一致** —— `DIRECT_LLM_HINTS` 缺 4 个良性问句的触发词（「这条河深不深？」
   「灰鸦渡有什么好吃的？」「你平时有什么爱好？」「最近身体怎么样？」），这些问句
   训练时是无资料条件，生产却会走 RAG 检索并注入资料，该条件下的行为没有被训练约束过。

## 许可

个人学习项目，剧情设定与 NPC 人设版权归原作所有。
