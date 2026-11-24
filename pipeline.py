"""Orchestration: ingest -> validate -> analyse -> explain."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd

from .analytics import SESSION, daily_ohlcv, filter_data
from .ingestion import IngestResult, ingest
from .insights import Insight, InsightProvider, generate_insights
from .quality import QualityConfig, QualityReport, run_quality


@dataclass
class PipelineResult:
    ingest: IngestResult
    quality: QualityReport
    daily: pd.DataFrame
    insights: List[Insight]


def run_pipeline(
    sources: Iterable,
    filenames: Optional[Iterable[str]] = None,
    cfg: Optional[QualityConfig] = None,
    convention: str = SESSION,
    contracts=None,
    start=None,
    end=None,
    insight_provider: Optional[InsightProvider] = None,
) -> PipelineResult:
    ing = ingest(sources, filenames)
    data = filter_data(ing.data, contracts, start, end)
    rep = run_quality(data, cfg, rejected=ing.rejected)
    daily = daily_ohlcv(rep.clean, convention)
    ins = generate_insights(rep, data, insight_provider)
    return PipelineResult(ingest=ing, quality=rep, daily=daily, insights=ins)
