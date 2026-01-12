"""
Redis client for caching and real-time features
"""
import redis
import json
from typing import Any, Optional
from .config import settings


class RedisClient:
    def __init__(self):
        self.client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5
        )

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"Redis GET error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = None):
        """Set value in Redis with optional TTL"""
        try:
            serialized = json.dumps(value)
            if ttl:
                self.client.setex(key, ttl, serialized)
            else:
                self.client.set(key, serialized)
        except Exception as e:
            print(f"Redis SET error: {e}")

    def delete(self, key: str):
        """Delete key from Redis"""
        try:
            self.client.delete(key)
        except Exception as e:
            print(f"Redis DELETE error: {e}")

    def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return self.client.exists(key) > 0
        except Exception as e:
            print(f"Redis EXISTS error: {e}")
            return False

    def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter"""
        try:
            return self.client.incrby(key, amount)
        except Exception as e:
            print(f"Redis INCR error: {e}")
            return 0

    def set_with_expiry(self, key: str, value: Any, ttl: int):
        """Set with expiration"""
        self.set(key, value, ttl)


# Global Redis instance
redis_client = RedisClient()
