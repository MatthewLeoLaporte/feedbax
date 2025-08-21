from collections.abc import ValuesView
from typing import Annotated

type SeqOf[S] = list[S] | tuple[S, ...] | ValuesView[S]
type SeqOfT[S, Tag] = Annotated[SeqOf[S], Tag]
