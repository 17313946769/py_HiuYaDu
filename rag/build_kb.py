# -*- coding: utf-8 -*-
"""知识库构建：5 份文档（4 docx + 1 md）→ 语义分块 → ChromaDB 向量索引

文档来源：
  1. 世界观设定集（docx）
  2. 实体关系设定（docx）
  3. 知识图谱（docx）
  4. 剧情文本（docx）
  5. 游戏设计规格文档（md）

分块策略：
  - 按章节标题切分边界
  - 相邻短段落合并至目标块长 300-600 字
  - 超长块用 LangChain RecursiveCharacterTextSplitter 二次切分（500字/100重叠）
  - docx 表格逐行转 "键:값" 文本
  - md 按 ## 标题切分

输出：D:\py_HiuYaDu\rag\chroma_db\ （ChromaDB 持久化目录）
用法：python rag/build_kb.py
"""
import json
import os
import re
import sys
import shutil

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.stdout.reconfigure(encoding="utf-8")

import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================
# 配置
# ============================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPC_CONFIG = os.path.join(ROOT, "npc_config.json")
CHROMA_DIR = os.path.join(ROOT, "rag", "chroma_db")

DOCS = [
    (r"E:\桌面\我的文件\灰鸦渡_开发日志\灰鸦渡相关\剧情设计方案\《灰鸦渡世界观设定集》.docx", "世界观设定集"),
    (r"E:\桌面\我的文件\灰鸦渡_开发日志\灰鸦渡相关\灰鸦渡_人物\灰鸦渡_实体关系完整设定文档.docx", "实体关系设定"),
    (r"E:\桌面\我的文件\灰鸦渡_开发日志\灰鸦渡相关\灰鸦渡_人物\灰鸦渡知识图谱.docx", "知识图谱"),
    (r"E:\桌面\我的文件\灰鸦渡_开发日志\灰鸦渡相关\剧情设计方案\剧情文本(终版).docx", "剧情文本"),
    (r"E:\桌面\我的文件\灰鸦渡_开发日志\灰鸦渡相关\剧情设计方案\灰鸦渡_游戏设计规格文档.md", "游戏设计规格"),
]

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

HEAD_PAT = re.compile(
    r"^(#{1,3}\s*|"
    r"第[一二三四五六七八九十百0-9]+[章节部分卷]\s*|"
    r"[一二三四五六七八九十]+[、.]\s*|"
    r"[（(][一二三四五六七八九十0-9]+[）)]\s*|"
    r"\d+\.\d+\s*)"
)


def is_heading(text):
    return bool(HEAD_PAT.match(text)) and len(text) <= 40


# ============================================================
# docx 解析
# ============================================================
def parse_docx(path):
    d = docx.Document(path)
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    units = []
    for el in d.element.body:
        if el.tag.endswith("}p"):
            p = Paragraph(el, d)
            t = p.text.strip()
            if t:
                units.append(("p", t))
        elif el.tag.endswith("}tbl"):
            t = Table(el, d)
            for r in t.rows:
                cells = [c.text.strip() for c in r.cells]
                line = " | ".join(c for c in cells if c)
                if line:
                    units.append(("t", line))
    return units


def chunk_docx(path, source):
    blocks, section, buf = [], "总述", []

    def flush():
        nonlocal buf
        text = "\n".join(buf).strip()
        buf = []
        if not text:
            return
        blocks.append({"source": source, "section": section, "text": text})

    for kind, content in parse_docx(path):
        line = content
        if is_heading(line):
            flush()
            section = re.sub(r"^#{1,3}\s*", "", line).strip()
            continue
        buf.append(line)
        if sum(len(x) for x in buf) >= 300:
            flush()
    flush()

    merged = []
    for b in blocks:
        if merged and merged[-1]["section"] == b["section"] and \
           len(merged[-1]["text"]) + len(b["text"]) + 1 <= 600:
            merged[-1]["text"] += "\n" + b["text"]
        else:
            merged.append(b)
    return merged


# ============================================================
# Markdown 解析
# ============================================================
def chunk_markdown(path, source):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    blocks, section, buf = [], "总述", []

    def flush():
        nonlocal buf
        text = "\n".join(buf).strip()
        buf = []
        if not text:
            return
        blocks.append({"source": source, "section": section, "text": text})

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and is_heading(stripped):
            flush()
            section = re.sub(r"^#{1,3}\s*", "", stripped).strip()
            continue
        if stripped:
            buf.append(stripped)
            if sum(len(x) for x in buf) >= 300:
                flush()
    flush()

    merged = []
    for b in blocks:
        if merged and merged[-1]["section"] == b["section"] and \
           len(merged[-1]["text"]) + len(b["text"]) + 1 <= 600:
            merged[-1]["text"] += "\n" + b["text"]
        else:
            merged.append(b)
    return merged


# ============================================================
# 主流程
# ============================================================
def main():
    for path, name in DOCS:
        if not os.path.exists(path):
            print(f"[FAIL] 文档不存在: {path}")
            sys.exit(1)
        print(f"[OK] 找到: {name}")

    all_blocks = []
    for path, source in DOCS:
        if path.endswith(".docx"):
            bs = chunk_docx(path, source)
        elif path.endswith(".md"):
            bs = chunk_markdown(path, source)
        else:
            continue
        total_chars = sum(len(b["text"]) for b in bs)
        print(f"[OK] {source}: {len(bs)} 块, {total_chars} 字")
        all_blocks.extend(bs)

    print(f"\n[总计] {len(all_blocks)} 块, {sum(len(b['text']) for b in all_blocks)} 字")

    print(f"\n[嵌入] 加载模型: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )

    documents = []
    for i, b in enumerate(all_blocks):
        sub_chunks = splitter.split_text(b["text"])
        for j, chunk in enumerate(sub_chunks):
            if len(chunk.strip()) < 20:
                continue
            documents.append({
                "id": f"kb-{i:04d}-{j}",
                "source": b["source"],
                "section": b["section"],
                "text": chunk.strip(),
            })

    print(f"[分块] 最终 {len(documents)} 个向量块")

    if os.path.exists(CHROMA_DIR):
        print(f"[清理] 删除旧 ChromaDB: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    texts = [d["text"] for d in documents]
    metadatas = [
        {"id": d["id"], "source": d["source"], "section": d["section"]}
        for d in documents
    ]
    ids = [d["id"] for d in documents]

    print(f"[ChromaDB] 创建向量索引 -> {CHROMA_DIR}")
    db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=CHROMA_DIR,
    )
    print(f"[OK] ChromaDB 已保存: {len(documents)} 条向量")

    sources = {}
    for d in documents:
        sources[d["source"]] = sources.get(d["source"], 0) + 1
    print("\n[来源分布]")
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt} 块")


if __name__ == "__main__":
    main()