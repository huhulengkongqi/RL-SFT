import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class QueueMessage:
    id: str
    stream: str
    data: Dict[str, Any]
    group: Optional[str] = None


class BaseQueue(ABC):
    @abstractmethod
    async def push(self, stream: str, data: Dict[str, Any]) -> str:
        """Push a message to a stream."""
        pass

    @abstractmethod
    async def pull(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 1,
        timeout: int = 5000,
    ) -> List[QueueMessage]:
        """Pull messages from a stream using consumer group."""
        pass

    @abstractmethod
    async def ack(self, stream: str, group: str, message_id: str) -> None:
        """Acknowledge a message."""
        pass

    @abstractmethod
    async def nack(self, stream: str, group: str, message_id: str) -> None:
        """Negative acknowledge (return to queue)."""
        pass

    @abstractmethod
    async def len(self, stream: str) -> int:
        """Get approximate length of stream."""
        pass

    @abstractmethod
    async def ensure_group(self, stream: str, group: str) -> None:
        """Ensure consumer group exists."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up connections."""
        pass


class InMemoryQueue(BaseQueue):
    """In-memory queue for testing and single-node mode."""

    def __init__(self):
        self.streams: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        self.groups: Dict[str, Dict[str, set]] = {}
        self._lock = asyncio.Lock()
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"{self._counter}"

    async def push(self, stream: str, data: Dict[str, Any]) -> str:
        async with self._lock:
            if stream not in self.streams:
                self.streams[stream] = []
            msg_id = self._next_id()
            self.streams[stream].append((msg_id, data))
            logger.debug(f"Pushed message {msg_id} to stream {stream}")
            return msg_id

    async def pull(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 1,
        timeout: int = 5000,
    ) -> List[QueueMessage]:
        async with self._lock:
            if stream not in self.streams:
                return []

            group_key = f"{stream}:{group}"
            if group_key not in self.groups:
                self.groups[group_key] = set()

            pending = self.groups[group_key]
            messages = []

            for msg_id, data in self.streams[stream]:
                if msg_id not in pending and len(messages) < count:
                    pending.add(msg_id)
                    messages.append(QueueMessage(
                        id=msg_id,
                        stream=stream,
                        data=data,
                        group=group,
                    ))

            logger.debug(f"Pulled {len(messages)} messages from stream {stream}")
            return messages

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        async with self._lock:
            group_key = f"{stream}:{group}"
            if group_key in self.groups:
                self.groups[group_key].discard(message_id)

            if stream in self.streams:
                original_len = len(self.streams[stream])
                self.streams[stream] = [
                    (mid, data) for mid, data in self.streams[stream]
                    if mid != message_id
                ]
                logger.debug(
                    f"Acked message {message_id} from stream {stream} "
                    f"(removed: {original_len - len(self.streams[stream])} items)"
                )
            else:
                logger.debug(f"Acked message {message_id} but stream {stream} not found")

    async def nack(self, stream: str, group: str, message_id: str) -> None:
        async with self._lock:
            group_key = f"{stream}:{group}"
            if group_key in self.groups:
                self.groups[group_key].discard(message_id)
            logger.debug(f"Nacked message {message_id} from stream {stream} group {group}")

    async def len(self, stream: str) -> int:
        async with self._lock:
            return len(self.streams.get(stream, []))

    async def ensure_group(self, stream: str, group: str) -> None:
        async with self._lock:
            group_key = f"{stream}:{group}"
            if group_key not in self.groups:
                self.groups[group_key] = set()

    async def close(self) -> None:
        pass


class RedisStreamQueue(BaseQueue):
    """Redis Streams based queue for distributed mode."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0", stream_max_len: int = 10000):
        self.redis_url = redis_url
        self.stream_max_len = stream_max_len
        self._redis: Optional[Any] = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> Any:
        async with self._lock:
            if self._redis is None:
                from redis import asyncio as aioredis
                self._redis = aioredis.from_url(self.redis_url)
            return self._redis

    async def push(self, stream: str, data: Dict[str, Any]) -> str:
        redis = await self._get_client()
        payload = {"data": json.dumps(data, ensure_ascii=False)}
        msg_id = await redis.xadd(stream, payload, maxlen=self.stream_max_len)
        logger.debug(f"Pushed message {msg_id} to stream {stream}")
        return msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)

    async def pull(
        self,
        stream: str,
        group: str,
        consumer: str,
        count: int = 1,
        timeout: int = 5000,
    ) -> List[QueueMessage]:
        redis = await self._get_client()
        try:
            results = await redis.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams={stream: ">"},
                count=count,
                block=timeout,
            )
        except Exception as e:
            logger.debug(f"Error pulling from stream {stream}: {e}")
            return []

        messages = []
        for _, msg_list in results:
            for msg_id, payload in msg_list:
                try:
                    data = json.loads(payload[b"data"].decode())
                    msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
                    messages.append(QueueMessage(
                        id=msg_id_str,
                        stream=stream,
                        data=data,
                        group=group,
                    ))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse message {msg_id}: {e}")
                    continue

        logger.debug(f"Pulled {len(messages)} messages from stream {stream}")
        return messages

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        redis = await self._get_client()
        await redis.xack(stream, group, message_id)
        logger.debug(f"Acked message {message_id} from stream {stream} group {group}")

    async def nack(self, stream: str, group: str, message_id: str) -> None:
        redis = await self._get_client()
        try:
            pending = await redis.xpending(stream, group)
            if pending and int(pending.get("pending", 0)) > 0:
                await redis.xclaim(stream, group, "nack_reclaimer", 0, [message_id])
                await redis.xack(stream, group, message_id)
                await self.push(stream, await self._get_message_data(stream, message_id))
        except Exception as e:
            logger.warning(f"Failed to nack message {message_id}: {e}")

    async def _get_message_data(self, stream: str, message_id: str) -> Dict[str, Any]:
        redis = await self._get_client()
        messages = await redis.xrange(stream, min=message_id, max=message_id, count=1)
        if messages:
            _, payload = messages[0]
            return json.loads(payload[b"data"].decode())
        return {}

    async def len(self, stream: str) -> int:
        redis = await self._get_client()
        try:
            info = await redis.xinfo_stream(stream)
            return info.get("length", 0)
        except Exception:
            return 0

    async def ensure_group(self, stream: str, group: str) -> None:
        redis = await self._get_client()
        try:
            await redis.xgroup_create(stream, group, id="0", mkstream=True)
            logger.info(f"Created consumer group {group} for stream {stream}")
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.debug(f"Consumer group {group} already exists for stream {stream}")
            pass

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()
            self._redis = None


def create_queue(use_redis: bool = True, redis_url: str = "redis://localhost:6379/0") -> BaseQueue:
    """Create a queue instance."""
    if use_redis:
        try:
            return RedisStreamQueue(redis_url)
        except Exception as e:
            logger.warning(f"Failed to create Redis queue, falling back to in-memory: {e}")
    return InMemoryQueue()
