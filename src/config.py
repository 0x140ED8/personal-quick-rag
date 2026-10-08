"""配置加载：dataclass + YAML。"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional

import yaml


@dataclass
class PathsConfig:
    data_dir: str = "data"
    index_dir: str = "index"
    parsed_dir: str = "parsed"


@dataclass
class ParserConfig:
    name: str = "docling"
    ocr: bool = False
    table_mode: str = "accurate"
    skip_toc: bool = True
    device: str = "cpu"
    heading_patterns: list = field(default_factory=list)


@dataclass
class ChunkerConfig:
    name: str = "recursive"
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class EmbedderConfig:
    name: str = "jasper"
    model_name: str = "infgrad/Jasper-Token-Compression-600M"
    device: str = "auto"
    dtype: str = "bfloat16"
    batch_size: int = 64
    compression_ratio: float = 0.3333  # token 压缩率 0.3-0.8，越小越快、质量略降


@dataclass
class StoreConfig:
    name: str = "faiss"
    distance_strategy: str = "MAX_INNER_PRODUCT"


@dataclass
class RetrieverConfig:
    name: str = "hybrid"
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60


@dataclass
class RerankerConfig:
    name: str = "bge"
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 5
    device: str = "auto"


@dataclass
class GeneratorConfig:
    name: str = "none"


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8100
    cors_origins: list = field(default_factory=lambda: ["*"])
    warmup: bool = True


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


# 顶层段落名 -> dataclass 类型，供校验/序列化使用
SECTION_TYPES = {
    "paths": PathsConfig,
    "parser": ParserConfig,
    "chunker": ChunkerConfig,
    "embedder": EmbedderConfig,
    "store": StoreConfig,
    "retriever": RetrieverConfig,
    "reranker": RerankerConfig,
    "generator": GeneratorConfig,
    "server": ServerConfig,
}


def _coerce_value(cls, name: str, value):
    """按 dataclass 字段类型做宽松转换（int/float/bool/str/list）。"""
    expected_type = None
    for f in fields(cls):
        if f.name == name:
            expected_type = f.type
            break
    if expected_type is None:
        raise ValueError(f"未知字段: {name}")
    if expected_type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ValueError(f"字段 {name} 应为 bool，得到 {type(value).__name__}")
    if expected_type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, str) and value.isdigit():
                return int(value)
            raise ValueError(f"字段 {name} 应为 int，得到 {type(value).__name__}")
        return value
    if expected_type == "float":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise ValueError(f"字段 {name} 应为 float，得到 {type(value).__name__}")
    if expected_type == "str":
        if not isinstance(value, str):
            raise ValueError(f"字段 {name} 应为 str，得到 {type(value).__name__}")
        return value
    if expected_type == "list":
        if not isinstance(value, list):
            raise ValueError(f"字段 {name} 应为 list，得到 {type(value).__name__}")
        return value
    return value


def config_to_dict(config: Config) -> dict:
    """Config -> dict（供 API 返回 JSON 配置）。"""
    out = {}
    for section_name, section_cls in SECTION_TYPES.items():
        section = getattr(config, section_name)
        out[section_name] = {f.name: getattr(section, f.name) for f in fields(section_cls)}
    return out


def validate_config_dict(data: dict) -> dict:
    """校验并规范化一份「全量配置 dict」（来自 POST /api/config）。

    返回规范化后的 dict；任何未知段落/字段/类型错误抛 ValueError（带定位信息）。
    """
    if not isinstance(data, dict) or not data:
        raise ValueError("配置体必须是非空 JSON 对象")
    unknown_sections = [k for k in data if k not in SECTION_TYPES]
    if unknown_sections:
        raise ValueError(f"未知配置段落: {unknown_sections}，合法段落: {sorted(SECTION_TYPES)}")

    normalized = {}
    for section_name, section_data in data.items():
        if not isinstance(section_data, dict):
            raise ValueError(f"配置段落 {section_name} 应为对象")
        cls = SECTION_TYPES[section_name]
        valid_names = {f.name for f in fields(cls)}
        unknown_fields = [k for k in section_data if k not in valid_names]
        if unknown_fields:
            raise ValueError(f"段落 {section_name} 存在未知字段: {unknown_fields}，合法字段: {sorted(valid_names)}")
        normalized[section_name] = {
            name: _coerce_value(cls, name, value) for name, value in section_data.items()
        }
    return normalized


def config_from_dict(data: dict) -> Config:
    """已校验的 dict -> Config。"""
    kwargs = {}
    for section_name, section_cls in SECTION_TYPES.items():
        if section_name in data:
            kwargs[section_name] = section_cls(**data[section_name])
    return Config(**kwargs)


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return config_from_dict(validate_config_dict(data))
