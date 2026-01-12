"""
Redis caching decorators and utilities
"""
import functools
import hashlib
import json
from typing import Any, Callable
from ..redis_client import redis_client


def cache_key_from_args(*args, **kwargs) -> str:
    """Generate cache key from function arguments"""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.md5(key_data.encode()).hexdigest()


def cached(ttl: int, prefix: str = ""):
    """
    Decorator for caching function results in Redis

    Args:
        ttl: Time to live in seconds
        prefix: Cache key prefix
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            args_hash = cache_key_from_args(*args, **kwargs)
            cache_key = f"{prefix}:{func.__name__}:{args_hash}"

            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            redis_client.set(cache_key, result, ttl)

            return result

        return wrapper
    return decorator


def invalidate_cache_pattern(pattern: str):
    """Invalidate all cache keys matching a pattern"""
    # Note: This requires SCAN in Redis
    # For production, consider using Redis SCAN command properly
    pass


def get_cache_stats() -> dict:
    """Get cache statistics"""
    try:
        info = redis_client.client.info("stats")
        return {
            "hits": info.get("keyspace_hits", 0),
            "misses": info.get("keyspace_misses", 0),
            "keys": info.get("db0", {}).get("keys", 0)
        }
    except Exception:
        return {}
