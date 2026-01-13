# src/backtest.py

import numpy as np
import pandas as pd


def simple_long_flat_strategy(predicted_ret: pd.Series, threshold: float = 0.0):
    """
    Long/flat strategy:
    - Go long (position=1) if predicted return > threshold
    - Otherwise flat (position=0).[web:169]
    """
    positions = (predicted_ret > threshold).astype(int)
    return positions


def compute_strategy_pnl(
    positions: pd.Series,
    actual_ret: pd.Series,
    transaction_cost_bps: float = 5.0,
):
    """
    Compute daily PnL series for a long/flat strategy with simple transaction costs.

    transaction_cost_bps: cost in basis points per change in position
    (e.g. 5 bps = 0.0005 each time you enter/exit).[web:171][web:165]
    """
    positions = positions.astype(float)
    actual_ret = actual_ret.astype(float)

    # strategy gross return = position * next-day actual return
    gross_ret = positions * actual_ret

    # transaction costs: apply cost when position changes
    position_change = positions.diff().abs().fillna(positions.iloc[0].abs())
    cost_per_change = transaction_cost_bps / 10000.0
    costs = position_change * cost_per_change

    net_ret = gross_ret - costs
    equity_curve = (1.0 + net_ret).cumprod()

    df = pd.DataFrame(
        {
            "position": positions,
            "actual_ret": actual_ret,
            "gross_ret": gross_ret,
            "costs": costs,
            "net_ret": net_ret,
            "equity_curve": equity_curve,
        },
        index=positions.index,
    )
    return df


def sharpe_ratio(daily_ret: pd.Series, trading_days: int = 252):
    """
    Annualized Sharpe assuming 0 risk-free rate.[web:157][web:165]
    """
    mu = daily_ret.mean()
    sigma = daily_ret.std()
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return float(np.sqrt(trading_days) * mu / sigma)


def max_drawdown(equity_curve: pd.Series):
    """
    Maximum drawdown of an equity curve (peak-to-trough).[web:155][web:162]
    """
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1.0
    return float(drawdown.min())


def hit_rate(actual_ret: pd.Series, predicted_ret: pd.Series, threshold: float = 0.0):
    """
    Fraction of days where sign(predicted_ret) == sign(actual_ret) when
    |predicted_ret| > threshold (optional filter).[web:169]
    """
    mask = predicted_ret.abs() > threshold
    if mask.sum() == 0:
        return 0.0
    sign_true = np.sign(actual_ret[mask])
    sign_pred = np.sign(predicted_ret[mask])
    return float((sign_true == sign_pred).mean())


def summarize_backtest(df_pnl: pd.DataFrame, predicted_ret: pd.Series):
    """
    Compute trading metrics from PnL dataframe and predictions.

    Returns dict with:
    - total_return
    - sharpe
    - max_drawdown
    - hit_rate
    - turnover
    """
    net_ret = df_pnl["net_ret"]
    equity_curve = df_pnl["equity_curve"]

    total_return = equity_curve.iloc[-1] - 1.0
    sr = sharpe_ratio(net_ret)
    mdd = max_drawdown(equity_curve)

    hr = hit_rate(df_pnl["actual_ret"], predicted_ret)

    # turnover = average abs change in position
    turnover = df_pnl["position"].diff().abs().fillna(df_pnl["position"].abs()).mean()

    return {
        "total_return": float(total_return),
        "sharpe": sr,
        "max_drawdown": mdd,
        "hit_rate": hr,
        "turnover": float(turnover),
    }
