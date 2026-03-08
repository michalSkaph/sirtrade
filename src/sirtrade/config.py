from dataclasses import dataclass, field


@dataclass
class RiskPolicy:
    target_vol_annual: float = 0.12
    soft_dd_alert: float = 0.10
    hard_dd_limit: float = 0.16
    derisk_trigger: float = 0.07
    per_trade_risk_nav: float = 0.0035
    daily_loss_limit_nav: float = 0.0125
    max_asset_exposure: float = 0.25
    max_long_tail_bucket: float = 0.55


@dataclass
class DecisionWeights:
    sortino: float = 0.28
    calmar: float = 0.22
    cvar95: float = -0.18
    max_dd: float = -0.14
    cost: float = -0.10
    turnover: float = -0.08


@dataclass
class DecisionThresholds:
    min_sortino: float = 1.25
    min_calmar: float = 0.80
    max_dd: float = 0.16
    max_cvar95: float = 0.021


@dataclass
class AppConfig:
    exchange: str = "Binance"
    market_data_source: str = "simulation"
    default_symbol: str = "BTCUSDT"
    base_url_spot: str = "https://api.binance.com"
    base_url_futures: str = "https://fapi.binance.com"
    tax_residency: str = "CZ"
    allow_spot: bool = True
    allow_perpetuals: bool = True
    allow_shorts: bool = True
    allow_leverage: bool = False
    fully_autonomous: bool = True
    ethical_restrictions: list[str] = field(
        default_factory=lambda: ["exclude_privacy_coins", "exclude_aml_high_risk_assets"]
    )
    generation_horizon_weeks: int = 8
    review_frequency_days: int = 7
    fee_bps_assumption: float = 10.0
    risk: RiskPolicy = field(default_factory=RiskPolicy)
    weights: DecisionWeights = field(default_factory=DecisionWeights)
    thresholds: DecisionThresholds = field(default_factory=DecisionThresholds)


DEFAULT_CONFIG = AppConfig()
