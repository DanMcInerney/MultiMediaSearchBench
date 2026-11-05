"""The `files` system (DESIGN 3.2): the agent's own tools on the recordings and nothing else. There is nothing to ingest and nothing to
serve; the configuration `files` is the agent with its shell, `Read`, `Glob` and `Grep` over the dump."""

from __future__ import annotations

from pathlib import Path

from mmmb.core.contract import Description, Dump, IngestReport, PlainSession, Session, SessionContext

MODALITIES = frozenset({"image", "video", "audio", "text", "records"})


class Files:
    def describe(self, variant: str | None) -> Description:
        if variant is not None:
            raise ValueError(f"files has no variants, not {variant!r}")
        return Description(name="files", version="1", variant=None, reads=MODALITIES, local=True, needs_gpu=False, gpu_memory_gb=None,
                           network=False, license="none", config_hash="none", store_hash="none", horizon="view", has_oracle=False)

    def check_ready(self, variant: str | None) -> None:
        """Nothing to check: the agent's own tools are the agent's."""

    def ingest(self, dump: Dump, store: Path) -> IngestReport:
        return IngestReport(seconds=0.0, gpu_seconds=0.0, bytes_written=0, refused=(), misdated=())

    def open(self, store: Path, dump: Dump, context: SessionContext) -> Session:
        return PlainSession()

    def preview(self, store: Path, dump: Dump, context: SessionContext) -> Session:
        return PlainSession()
