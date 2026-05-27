from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from redis.exceptions import RedisError

from app.core.config import get_settings
from app.core.redis import get_redis_client


@dataclass
class RateLimitResult:
    allowed: bool
    key: str
    current: int = 0
    limit: int = 0
    retry_after: int = 0
    degraded: bool = False


class RedisService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = get_redis_client()

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except RedisError:
            return False

    def check_chat_rate_limit(self, user_id: int) -> RateLimitResult:
        key = f"rate_limit:chat:user:{user_id}"
        limit = self.settings.chat_rate_limit_per_minute
        try:
            current = self.client.incr(key)
            if current == 1:
                self.client.expire(key, 60)
            ttl = self.client.ttl(key)
            retry_after = ttl if ttl and ttl > 0 else 60
            return RateLimitResult(
                allowed=current <= limit,
                key=key,
                current=int(current),
                limit=limit,
                retry_after=retry_after,
            )
        except RedisError as exc:
            print(f"Redis 限流不可用，已降级放行：{exc}")
            return RateLimitResult(allowed=True, key=key, limit=limit, degraded=True)

    def get_cached_chat_answer(self, *, user_id: int, knowledge_base_id: int, question: str) -> dict | None:
        key = self.chat_cache_key(user_id=user_id, knowledge_base_id=knowledge_base_id, question=question)
        try:
            value = self.client.get(key)
        except RedisError as exc:
            print(f"Redis 问答缓存读取失败，已降级：{exc}")
            return None

        if not value:
            return None
        try:
            cached = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(cached, dict):
            cached["cache_hit"] = True
            cached["cache_key"] = key
            return cached
        return None

    def cache_chat_answer(
        self,
        *,
        user_id: int,
        knowledge_base_id: int,
        question: str,
        result: dict,
    ) -> str | None:
        key = self.chat_cache_key(user_id=user_id, knowledge_base_id=knowledge_base_id, question=question)
        payload = {
            key_name: value
            for key_name, value in result.items()
            if key_name not in {"cache_hit", "cache_key"}
        }
        try:
            self.client.setex(
                key,
                self.settings.chat_cache_ttl_seconds,
                json.dumps(payload, ensure_ascii=False),
            )
            return key
        except RedisError as exc:
            print(f"Redis 问答缓存写入失败，已降级：{exc}")
            return None

    def set_task_status(self, task_id: str | None, status: str, payload: dict | None = None) -> None:
        if not task_id:
            return
        key = self.task_status_key(task_id)
        data = {"task_id": task_id, "status": status, **(payload or {})}
        try:
            self.client.setex(
                key,
                self.settings.task_status_cache_ttl_seconds,
                json.dumps(data, ensure_ascii=False),
            )
        except RedisError as exc:
            print(f"Redis 任务状态缓存写入失败，已降级：{exc}")

    def get_task_status(self, task_id: str) -> dict | None:
        try:
            value = self.client.get(self.task_status_key(task_id))
        except RedisError as exc:
            print(f"Redis 任务状态缓存读取失败，已降级：{exc}")
            return None
        if not value:
            return None
        try:
            data = json.loads(value)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def chat_cache_key(*, user_id: int, knowledge_base_id: int, question: str) -> str:
        normalized_question = " ".join(question.strip().split()).lower()
        digest = hashlib.sha256(normalized_question.encode("utf-8")).hexdigest()
        return f"chat_cache:user:{user_id}:kb:{knowledge_base_id}:q:{digest}"

    @staticmethod
    def task_status_key(task_id: str) -> str:
        return f"task_status:{task_id}"


redis_service = RedisService()

