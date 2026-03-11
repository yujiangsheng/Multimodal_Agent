"""RAG 知识库示例 — 文档导入与检索。

演示 DocumentStore 的文档分块、存储和混合搜索。
# 离线可运行
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pinocchio.rag import DocumentStore


def basic_ingestion():
    """导入文本并检索。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DocumentStore(data_dir=tmpdir)

        # 直接导入文本
        store.ingest_text(
            "Pinocchio 是一个多模态自我进化智能体。"
            "它通过六阶段认知循环实现持续学习：感知、策略、执行、评估、学习、元反思。"
            "核心特点包括双轴记忆系统、工具调用框架和流式输出。",
            source="pinocchio_intro",
        )

        # 关键字检索
        results = store.search("认知循环", top_k=3)

        print("=== 基本导入与检索 ===")
        for chunk in results:
            print(f"[score={chunk.score:.2f}] {chunk.text[:100]}...")
        print(f"文档数: {store.get_document_count()}")
        print(f"分块数: {store.get_chunk_count()}")
        print()


def file_ingestion():
    """从文件导入。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DocumentStore(data_dir=tmpdir)

        # 创建示例文档
        doc_path = Path(tmpdir) / "example.md"
        doc_path.write_text(
            "# Quantum Computing\n\n"
            "Quantum computing uses qubits instead of classical bits.\n\n"
            "## Key Concepts\n\n"
            "Superposition allows a qubit to be in multiple states simultaneously.\n"
            "Entanglement creates correlations between qubits.\n\n"
            "## Applications\n\n"
            "Drug discovery, optimization, and cryptography are major areas.\n",
            encoding="utf-8",
        )

        store.ingest(str(doc_path))

        results = store.search("entanglement qubits", top_k=2)

        print("=== 文件导入 ===")
        for chunk in results:
            print(f"  source={chunk.source}, text={chunk.text[:80]}...")

        # 列出所有文档
        docs = store.list_documents()
        print(f"\n已导入文档:")
        for doc in docs:
            print(f"  {doc['source']} ({doc['chunk_count']} chunks)")
        print()


def management_demo():
    """文档管理操作。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = DocumentStore(data_dir=tmpdir)

        store.ingest_text("Hello world", source="doc1")
        store.ingest_text("Goodbye world", source="doc2")

        print("=== 文档管理 ===")
        print(f"导入前: {store.get_document_count()} docs, {store.get_chunk_count()} chunks")

        store.delete_document("doc1")
        print(f"删除后: {store.get_document_count()} docs, {store.get_chunk_count()} chunks")
        print()


if __name__ == "__main__":
    basic_ingestion()
    file_ingestion()
    management_demo()
