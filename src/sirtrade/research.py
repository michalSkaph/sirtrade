from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class StudyInsight:
    title: str
    year: int
    evidence_strength: str
    limitations: str
    overfit_risk: str
    proposal: str


EVIDENCE_BASE = [
    StudyInsight(
        title="Time-series momentum v krypto průřezech",
        year=2023,
        evidence_strength="střední",
        limitations="nestabilita režimů a vyšší poplatky na long-tail aktivech",
        overfit_risk="střední",
        proposal="Zvýšit trendový horizont z 10d na 14d ve stresových režimech.",
    ),
    StudyInsight(
        title="Portfolia řízená volatilitou",
        year=2017,
        evidence_strength="vysoká",
        limitations="falešné signály při prudkých obratech trendu",
        overfit_risk="nízké",
        proposal="Zpřísnit vol cap z 1.5x na 1.3x v clusterech vysoké volatility.",
    ),
    StudyInsight(
        title="Purged walk-forward validace pro finanční ML",
        year=2018,
        evidence_strength="vysoká",
        limitations="nižší efektivita vzorku",
        overfit_risk="nízké",
        proposal="Prodloužit embargo okno pro překrývající se labely o +2 dny.",
    ),
    StudyInsight(
        title="Hierarchická risk parity pro nestabilní korelace",
        year=2016,
        evidence_strength="střední",
        limitations="citlivost při krizových skocích korelací",
        overfit_risk="nízké až střední",
        proposal="Zvýšit penalizaci long-tail korelací ve stresových režimech.",
    ),
]


def daily_deep_research(seed: int) -> list[StudyInsight]:
    rnd = random.Random(seed)
    picks = rnd.sample(EVIDENCE_BASE, k=min(3, len(EVIDENCE_BASE)))
    return picks
