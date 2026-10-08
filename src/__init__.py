"""模块化 RAG 框架核心包。

按功能解耦为七个阶段：
    1.解析(parsing) -> 2.分片(chunking) -> 3.向量化(embedding) -> 4.存储(storage)
    -> 5.召回(retrieval) -> 6.重排(reranking) -> 7.生成(generation)

各阶段均通过 registry 注册，可在 config.yaml 中按 name 快速替换实现。
"""
