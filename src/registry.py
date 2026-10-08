"""组件工厂：按 (kind, name) 注册与实例化，实现可插拔。

用法：
    registry.register("parser", "docling", lambda **kw: DoclingParser(**kw))
    obj = registry.create("parser", "docling", heading_patterns=[...])
"""
from __future__ import annotations


class Registry:
    def __init__(self) -> None:
        self._factories: dict[tuple, callable] = {}

    def register(self, kind: str, name: str, factory: callable) -> None:
        self._factories[(kind, name)] = factory

    def create(self, kind: str, name: str, **kwargs):
        key = (kind, name)
        if key not in self._factories:
            known = [k[1] for k in self._factories if k[0] == kind]
            raise ValueError(f"未知组件: {kind}={name}，已注册: {known}")
        return self._factories[key](**kwargs)


registry = Registry()
