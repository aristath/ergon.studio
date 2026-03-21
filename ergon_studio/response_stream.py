from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Sequence
from typing import cast


class ResponseStream[UpdateT, FinalT](AsyncIterable[UpdateT]):
    def __init__(
        self,
        stream: AsyncIterable[UpdateT] | Awaitable[AsyncIterable[UpdateT]],
        *,
        finalizer: Callable[[Sequence[UpdateT]], FinalT],
    ) -> None:
        self._source: AsyncIterable[UpdateT] | Awaitable[AsyncIterable[UpdateT]] = (
            stream
        )
        self._iterator: AsyncIterator[UpdateT] | None = None
        self._updates: list[UpdateT] = []
        self._finalizer = finalizer
        self._finalized = False
        self._final_result: FinalT | None = None

    def __aiter__(self) -> ResponseStream[UpdateT, FinalT]:
        return self

    async def __anext__(self) -> UpdateT:
        iterator = await self._ensure_iterator()
        try:
            update = await iterator.__anext__()
        except StopAsyncIteration:
            raise
        self._updates.append(update)
        return update

    async def get_final_response(self) -> FinalT:
        if self._finalized:
            return cast(FinalT, self._final_result)
        async for _ in self:
            pass
        self._final_result = self._finalizer(tuple(self._updates))
        self._finalized = True
        return self._final_result

    async def _ensure_iterator(self) -> AsyncIterator[UpdateT]:
        if self._iterator is not None:
            return self._iterator
        source = self._source
        iterable: AsyncIterable[UpdateT]
        if inspect.isawaitable(source):
            iterable = await source
        else:
            iterable = source
        self._iterator = iterable.__aiter__()
        return self._iterator
