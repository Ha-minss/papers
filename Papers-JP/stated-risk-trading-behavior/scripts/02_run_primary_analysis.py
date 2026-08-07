from __future__ import annotations

import argparse
import os
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from catboost import CatBoostClassifier
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

RANDOM_SEED = 42
ACTUAL_RISKS = ["Conservative", "Income", "Balanced", "Aggressive"]


def load_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    frames = {
        "asset": pd.read_csv(data_dir / "asset_information.csv"),
        "customer": pd.read_csv(data_dir / "customer_information.csv"),
        "transactions": pd.read_csv(data_dir / "transactions.csv"),
        "prices": pd.read_csv(data_dir / "close_prices.csv"),
        "markets": pd.read_csv(data_dir / "markets.csv"),
    }
    for key, col in [("asset", "timestamp"), ("customer", "timestamp"), ("transactions", "timestamp"), ("prices", "timestamp")]:
        frames[key][col] = pd.to_datetime(frames[key][col], errors="coerce")
    frames["customer"]["lastQuestionnaireDate"] = pd.to_datetime(
        frames["customer"]["lastQuestionnaireDate"], errors="coerce"
    )
    return frames


def detect_events(asset: pd.DataFrame, prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Index]:
    asset_latest = asset.sort_values("timestamp").drop_duplicates("ISIN", keep="last")
    xath_isins = set(asset_latest.query("assetCategory == 'Stock' and marketID == 'XATH'").ISIN)
    p = prices[prices.ISIN.isin(xath_isins)].sort_values(["ISIN", "timestamp"])
    price_wide = p.pivot(index="timestamp", columns="ISIN", values="closePrice").sort_index()
    returns = price_wide.pct_change(fill_method=None)
    coverage = pd.DataFrame({
        "n_prices": price_wide.notna().sum(),
        "nonzero_return_share": (returns.abs() > 1e-12).sum() / returns.notna().sum(),
    })
    eligible = coverage.query("n_prices >= 500 and nonzero_return_share >= 0.30").index
    market_return = returns[eligible].median(axis=1, skipna=True)

    shifted_r = returns[eligible].shift(11)
    shifted_m = market_return.shift(11)
    rolling_mean_r = shifted_r.rolling(250, min_periods=200).mean()
    rolling_mean_m = shifted_m.rolling(250, min_periods=200).mean()
    rolling_mean_product = shifted_r.mul(shifted_m, axis=0).rolling(250, min_periods=200).mean()
    rolling_cov = rolling_mean_product - rolling_mean_r.mul(rolling_mean_m, axis=0)
    rolling_var_m = shifted_m.rolling(250, min_periods=200).var(ddof=0)
    beta = rolling_cov.div(rolling_var_m, axis=0)
    alpha = rolling_mean_r - beta.mul(rolling_mean_m, axis=0)
    expected = alpha + beta.mul(market_return, axis=0)
    abnormal = returns[eligible] - expected
    lower_1pct = abnormal.shift(11).rolling(250, min_periods=200).quantile(0.01)
    event_mask = (returns[eligible] <= -0.05) & (abnormal <= lower_1pct)

    daily_share = event_mask.sum(axis=1) / len(eligible)
    event_rows: List[Tuple] = []
    for isin in eligible:
        last_index = -10_000
        for event_date in event_mask.index[event_mask[isin].fillna(False)]:
            index = event_mask.index.get_loc(event_date)
            if index - last_index >= 20:
                event_rows.append((
                    isin,
                    event_date,
                    float(returns.at[event_date, isin]),
                    float(abnormal.at[event_date, isin]),
                    float(market_return.at[event_date]),
                    bool(daily_share.at[event_date] >= 0.05),
                ))
                last_index = index
    events = pd.DataFrame(
        event_rows,
        columns=["ISIN", "event_date", "raw_return", "abnormal_return", "market_return", "broad_shock"],
    )
    return events, price_wide, returns, abnormal, market_return, eligible


def prepare_customers_and_transactions(
    customer: pd.DataFrame,
    transactions: pd.DataFrame,
    asset: pd.DataFrame,
    eligible_assets: Iterable[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customer_latest = customer.sort_values("timestamp").drop_duplicates("customerID", keep="last")
    eligible_customers = customer_latest[
        customer_latest.customerType.isin(["Mass", "Premium"])
        & customer_latest.riskLevel.isin(ACTUAL_RISKS)
    ].copy()

    stock_xath = transactions[
        transactions.ISIN.isin(set(eligible_assets)) & transactions.marketID.eq("XATH")
    ].copy()
    stock_xath["signed_units"] = np.where(
        stock_xath.transactionType.eq("Buy"), stock_xath.units, -stock_xath.units
    )
    synthetic_pairs_frame = stock_xath.loc[
        stock_xath.transactionID < 0, ["customerID", "ISIN"]
    ].drop_duplicates()
    synthetic_pairs = pd.MultiIndex.from_frame(synthetic_pairs_frame)
    actual = stock_xath[stock_xath.transactionID >= 0].copy()
    pair_index = pd.MultiIndex.from_frame(actual[["customerID", "ISIN"]])
    actual = actual.loc[~pair_index.isin(synthetic_pairs)].copy()
    actual = actual[actual.customerID.isin(eligible_customers.customerID)]

    asset_latest = asset.sort_values("timestamp").drop_duplicates("ISIN", keep="last")
    asset_meta = asset_latest.set_index("ISIN")[["assetName", "sector", "industry"]]
    return eligible_customers, actual, customer_latest, asset_meta


def build_exposures(
    events: pd.DataFrame,
    stock_actual: pd.DataFrame,
    eligible_customers: pd.DataFrame,
    price_wide: pd.DataFrame,
) -> pd.DataFrame:
    trading_dates = price_wide.index
    date_position = {date: index for index, date in enumerate(trading_dates)}
    available_assets = set(stock_actual.ISIN.unique())
    parts: List[pd.DataFrame] = []

    for isin, event_group in events.groupby("ISIN"):
        if isin not in available_assets:
            continue
        tx_asset = stock_actual[stock_actual.ISIN == isin].copy()
        daily = tx_asset.groupby(["timestamp", "customerID"], as_index=False)["signed_units"].sum()
        matrix = daily.pivot(index="timestamp", columns="customerID", values="signed_units").fillna(0.0)
        matrix = matrix.reindex(trading_dates, fill_value=0.0)
        holdings_before = matrix.cumsum().shift(1).fillna(0.0)
        histories = {
            customer_id: group.sort_values(["timestamp", "transactionID"])
            for customer_id, group in tx_asset.groupby("customerID")
        }

        for event in event_group.sort_values("event_date").itertuples(index=False):
            event_date = event.event_date
            event_index = date_position.get(event_date)
            if event_index is None or event_index == 0 or event_index + 5 >= len(trading_dates):
                continue
            holders = holdings_before.loc[event_date]
            holders = holders[holders > 1e-8]
            if holders.empty:
                continue
            response_dates = set(trading_dates[event_index + 1 : event_index + 6])
            response_tx = tx_asset[
                tx_asset.timestamp.isin(response_dates) & tx_asset.customerID.isin(holders.index)
            ].sort_values(["customerID", "timestamp", "transactionID"])
            first_response = response_tx.groupby("customerID").first() if not response_tx.empty else pd.DataFrame()
            previous_price = price_wide.at[trading_dates[event_index - 1], isin]
            rows = []
            for customer_id, units_before in holders.items():
                history = histories[customer_id]
                prior = history[history.timestamp < event_date]
                response = "NoAction"
                response_date = pd.NaT
                if not first_response.empty and customer_id in first_response.index:
                    response = first_response.at[customer_id, "transactionType"]
                    response_date = first_response.at[customer_id, "timestamp"]
                rows.append({
                    "customerID": customer_id,
                    "ISIN": isin,
                    "event_date": event_date,
                    "response": response,
                    "response_date": response_date,
                    "units_before": float(units_before),
                    "position_value_before": float(units_before * previous_price),
                    "asset_prior_trades": len(prior),
                    "asset_prior_buys": int((prior.transactionType == "Buy").sum()),
                    "asset_prior_sells": int((prior.transactionType == "Sell").sum()),
                    "days_since_asset_trade": (event_date - prior.timestamp.max()).days,
                    "raw_return": event.raw_return,
                    "abnormal_return": event.abnormal_return,
                    "market_return": event.market_return,
                    "broad_shock": event.broad_shock,
                })
            parts.append(pd.DataFrame(rows))

    exposures = pd.concat(parts, ignore_index=True)
    profile_cols = [
        "customerID",
        "customerType",
        "riskLevel",
        "investmentCapacity",
        "lastQuestionnaireDate",
    ]
    exposures = exposures.merge(eligible_customers[profile_cols], on="customerID", how="left")
    exposures["questionnaire_age_days"] = (
        exposures.event_date - exposures.lastQuestionnaireDate
    ).dt.days
    # Keep pre-questionnaire exposures temporarily so they can contribute to
    # strictly historical revealed-behavior features. The prediction sample is
    # filtered only after those lagged features are constructed.
    return exposures


def add_behavior_features(
    exposures: pd.DataFrame,
    transactions: pd.DataFrame,
    eligible_customers: pd.DataFrame,
    stock_isins: set,
) -> pd.DataFrame:
    actual = transactions[
        (transactions.transactionID >= 0)
        & transactions.customerID.isin(eligible_customers.customerID)
    ].copy()
    actual = actual.sort_values(["customerID", "timestamp", "transactionID"])
    actual["is_buy"] = actual.transactionType.eq("Buy").astype(int)
    actual["is_stock"] = actual.ISIN.isin(stock_isins).astype(int)
    actual["is_internet"] = actual.channel.eq("Internet Banking").astype(int)
    actual["log_value"] = np.log1p(actual.totalValue.clip(lower=0))

    histories = {}
    for customer_id, group in actual.groupby("customerID"):
        histories[customer_id] = {
            "dates": group.timestamp.values.astype("datetime64[D]"),
            "buy_cum": np.cumsum(group.is_buy.to_numpy()),
            "stock_cum": np.cumsum(group.is_stock.to_numpy()),
            "internet_cum": np.cumsum(group.is_internet.to_numpy()),
            "logvalue_cum": np.cumsum(group.log_value.to_numpy()),
            "isins": group.ISIN.to_numpy(),
        }

    feature_rows = []
    for row in exposures[["customerID", "event_date"]].itertuples(index=False):
        date = np.datetime64(row.event_date.date(), "D")
        history = histories.get(row.customerID)
        if history is None:
            feature_rows.append((0, 0, 0, 0, np.nan, 0, 0, np.nan))
            continue
        index = np.searchsorted(history["dates"], date, side="left")
        if index == 0:
            feature_rows.append((0, 0, 0, 0, np.nan, 0, 0, np.nan))
            continue
        buys = history["buy_cum"][index - 1]
        stocks = history["stock_cum"][index - 1]
        internet = history["internet_cum"][index - 1]
        index_90 = np.searchsorted(history["dates"], date - np.timedelta64(90, "D"), side="left")
        feature_rows.append((
            index,
            buys / index,
            stocks / index,
            internet / index,
            history["logvalue_cum"][index - 1] / index,
            index - index_90,
            len(set(history["isins"][:index])),
            int((date - history["dates"][index - 1]).astype(int)),
        ))
    names = [
        "prior_transactions",
        "prior_buy_share",
        "prior_stock_share",
        "prior_internet_share",
        "prior_mean_log_value",
        "prior_transactions_90d",
        "prior_unique_assets",
        "days_since_any_trade",
    ]
    features = pd.DataFrame(feature_rows, columns=names)
    data = pd.concat([exposures.reset_index(drop=True), features], axis=1)
    data = data.sort_values(["event_date", "ISIN", "customerID"]).reset_index(drop=True)
    data["acted"] = data.response.ne("NoAction").astype(int)
    data["bought"] = data.response.eq("Buy").astype(int)
    data["sold"] = data.response.eq("Sell").astype(int)
    grouped = data.groupby("customerID", sort=False)
    data["prior_shock_exposures"] = grouped.cumcount()
    data["prior_shock_actions"] = grouped.acted.cumsum() - data.acted
    data["prior_shock_buys"] = grouped.bought.cumsum() - data.bought
    data["prior_shock_sells"] = grouped.sold.cumsum() - data.sold
    data["prior_action_rate_smoothed"] = (
        data.prior_shock_actions + 1.0
    ) / (data.prior_shock_exposures + 2.0)
    data["prior_sell_rate_smoothed"] = (
        data.prior_shock_sells + 1.0
    ) / (data.prior_shock_actions + 2.0)
    data["year"] = data.event_date.dt.year
    return data



def add_response_windows(data: pd.DataFrame, stock_actual: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Attach first-action labels for 1, 10, and 20 trading-day windows."""
    left = data.reset_index().rename(columns={"index": "_row_id"}).sort_values("event_date")
    right = (stock_actual[["customerID", "ISIN", "timestamp", "transactionID", "transactionType"]]
             .sort_values(["timestamp", "transactionID"])
             .drop_duplicates(["customerID", "ISIN", "timestamp"], keep="first")
             .sort_values("timestamp"))
    matched = pd.merge_asof(
        left, right, left_on="event_date", right_on="timestamp",
        by=["customerID", "ISIN"], direction="forward", allow_exact_matches=False,
    )
    pos = pd.Series(np.arange(len(trading_dates)), index=trading_dates)
    event_pos = matched["event_date"].map(pos)
    tx_pos = matched["timestamp"].map(pos)
    gap = tx_pos - event_pos
    last_pos = len(trading_dates) - 1
    for window in (1, 10, 20):
        complete = event_pos + window <= last_pos
        action = matched["transactionType"].where(gap.le(window), "NoAction")
        action = action.where(complete, np.nan)
        matched[f"response_{window}d"] = action
        matched[f"acted_{window}d"] = np.where(action.isna(), np.nan, action.ne("NoAction").astype(float))
    cols = ["_row_id"] + [f"response_{w}d" for w in (1,10,20)] + [f"acted_{w}d" for w in (1,10,20)]
    extra = matched[cols].set_index("_row_id").sort_index()
    result = data.reset_index(drop=True).copy()
    for column in extra.columns:
        result[column] = extra[column].to_numpy()
    return result

def engineer_model_features(data: pd.DataFrame, asset_meta: pd.DataFrame) -> pd.DataFrame:
    data = data.merge(asset_meta.reset_index(), on="ISIN", how="left")
    transforms = {
        "log_position_value": data.position_value_before,
        "log_units_before": data.units_before,
        "log_asset_prior_trades": data.asset_prior_trades,
        "log_prior_transactions": data.prior_transactions,
        "log_prior_transactions_90d": data.prior_transactions_90d,
        "log_prior_unique_assets": data.prior_unique_assets,
        "log_days_since_any_trade": data.days_since_any_trade.clip(lower=0),
        "log_days_since_asset_trade": data.days_since_asset_trade.clip(lower=0),
        "log_prior_shock_exposures": data.prior_shock_exposures,
    }
    for name, values in transforms.items():
        data[name] = np.log1p(values)
    return data


def prepare_frames(data: pd.DataFrame, features: List[str], train_mask, val_mask, test_mask):
    categorical = [
        column for column in features
        if data[column].dtype == object and column not in {"market_shock", "broad_shock"}
    ]
    numeric = [column for column in features if column not in categorical]
    frames = [data.loc[mask, features].copy() for mask in (train_mask, val_mask, test_mask)]
    medians = frames[0][numeric].replace([np.inf, -np.inf], np.nan).median()
    for frame in frames:
        frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan).fillna(medians)
        for column in categorical:
            frame[column] = frame[column].fillna("Missing").astype(str)
    return frames[0], frames[1], frames[2], categorical


def fit_catboost(data: pd.DataFrame, features: List[str], target: str, model_name: str):
    train_mask = data.year <= 2020
    validation_mask = data.year == 2021
    test_mask = data.year == 2022
    x_train, x_validation, x_test, categorical = prepare_frames(
        data, features, train_mask, validation_mask, test_mask
    )
    y_train = data.loc[train_mask, target].astype(int)
    y_validation = data.loc[validation_mask, target].astype(int)
    y_test = data.loc[test_mask, target].astype(int)
    model = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_SEED,
        verbose=False,
        l2_leaf_reg=5,
        random_strength=0.5,
    )
    model.fit(
        x_train,
        y_train,
        cat_features=categorical,
        eval_set=(x_validation, y_validation),
        early_stopping_rounds=60,
        verbose=False,
    )
    validation_raw = model.predict_proba(x_validation)[:, 1]
    test_raw = model.predict_proba(x_test)[:, 1]
    epsilon = 1e-6
    validation_logit = np.log(
        np.clip(validation_raw, epsilon, 1 - epsilon)
        / (1 - np.clip(validation_raw, epsilon, 1 - epsilon))
    ).reshape(-1, 1)
    calibrator = LogisticRegression().fit(validation_logit, y_validation)
    test_logit = np.log(
        np.clip(test_raw, epsilon, 1 - epsilon)
        / (1 - np.clip(test_raw, epsilon, 1 - epsilon))
    ).reshape(-1, 1)
    test_probability = calibrator.predict_proba(test_logit)[:, 1]
    metrics = {
        "model": model_name,
        "n_train": len(y_train),
        "n_val": len(y_validation),
        "n_test": len(y_test),
        "test_prevalence": y_test.mean(),
        "PR_AUC": average_precision_score(y_test, test_probability),
        "ROC_AUC": roc_auc_score(y_test, test_probability),
        "Brier": brier_score_loss(y_test, test_probability),
        "LogLoss": log_loss(y_test, test_probability),
        "best_iteration": model.get_best_iteration(),
    }
    return model, metrics, y_test.to_numpy(), test_probability


def fit_inference_model(data: pd.DataFrame, target: str, sell_stage: bool = False) -> pd.DataFrame:
    work = data.copy()
    work["event_id"] = work.event_date.astype(str) + "|" + work.ISIN
    prior_rate = "prior_sell_rate_smoothed" if sell_stage else "prior_action_rate_smoothed"
    numeric = [
        "raw_return", "abnormal_return", "market_return", "log_position_value",
        "log_asset_prior_trades", "log_days_since_asset_trade", "log_prior_transactions",
        "prior_buy_share", "prior_internet_share", "log_prior_transactions_90d",
        "log_prior_unique_assets", "log_days_since_any_trade", prior_rate,
        "questionnaire_age_days",
    ]
    for column in numeric:
        values = work[column].replace([np.inf, -np.inf], np.nan)
        work[column + "_z"] = ((values - values.mean()) / values.std()).fillna(0)
    formula = f"""{target} ~ C(riskLevel, Treatment(reference='Conservative'))
        + C(investmentCapacity) + C(customerType)
        + raw_return_z + abnormal_return_z + market_return_z + broad_shock
        + log_position_value_z + log_asset_prior_trades_z + log_days_since_asset_trade_z
        + log_prior_transactions_z + prior_buy_share_z + prior_internet_share_z
        + log_prior_transactions_90d_z + log_prior_unique_assets_z + log_days_since_any_trade_z
        + {prior_rate}_z + questionnaire_age_days_z"""
    y, x = patsy.dmatrices(formula, work, return_type="dataframe", NA_action="drop")
    result = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=200, disp=0)
    customer_codes = pd.factorize(work.loc[x.index, "customerID"])[0]
    event_codes = pd.factorize(work.loc[x.index, "event_id"])[0]
    covariance, _, _ = cov_cluster_2groups(result, customer_codes, event_codes)
    standard_error = np.sqrt(np.diag(covariance))
    coefficients = result.params
    p_values = 2 * (1 - norm.cdf(np.abs(coefficients / standard_error)))
    return pd.DataFrame({
        "term": coefficients.index,
        "coef": coefficients.values,
        "se_2way": standard_error,
        "p": p_values,
        "OR": np.exp(coefficients.values),
        "OR_low": np.exp(coefficients.values - 1.96 * standard_error),
        "OR_high": np.exp(coefficients.values + 1.96 * standard_error),
    })


def event_bootstrap_difference(meta: pd.DataFrame, probability_a: np.ndarray, probability_b: np.ndarray, n_boot: int = 500):
    work = meta.copy().reset_index(drop=True)
    work["probability_a"] = probability_a
    work["probability_b"] = probability_b
    work["event_id"] = work.event_date.astype(str) + "|" + work.ISIN
    groups = {event_id: index.to_numpy() for event_id, index in work.groupby("event_id").groups.items()}
    event_ids = np.array(list(groups))
    rng = np.random.default_rng(123)
    pr_differences, brier_differences, auc_differences = [], [], []
    y_all = work.acted.to_numpy()
    for _ in range(n_boot):
        sampled = rng.choice(event_ids, size=len(event_ids), replace=True)
        indices = np.concatenate([groups[event_id] for event_id in sampled])
        y = y_all[indices]
        if y.min() == y.max():
            continue
        p_a = work.probability_a.to_numpy()[indices]
        p_b = work.probability_b.to_numpy()[indices]
        pr_differences.append(average_precision_score(y, p_a) - average_precision_score(y, p_b))
        brier_differences.append(brier_score_loss(y, p_a) - brier_score_loss(y, p_b))
        auc_differences.append(roc_auc_score(y, p_a) - roc_auc_score(y, p_b))
    def summarize(values):
        return float(np.mean(values)), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))
    return {"PR": summarize(pr_differences), "Brier": summarize(brier_differences), "AUC": summarize(auc_differences)}


def main(data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = load_data(data_dir)
    events, price_wide, _, _, _, eligible_assets = detect_events(frames["asset"], frames["prices"])
    eligible_customers, stock_actual, _, asset_meta = prepare_customers_and_transactions(
        frames["customer"], frames["transactions"], frames["asset"], eligible_assets
    )
    exposures = build_exposures(events, stock_actual, eligible_customers, price_wide)
    stock_isins = set(
        frames["asset"].sort_values("timestamp").drop_duplicates("ISIN", keep="last")
        .query("assetCategory == 'Stock'").ISIN
    )
    data = add_behavior_features(exposures, frames["transactions"], eligible_customers, stock_isins)
    data = data[data.questionnaire_age_days >= 0].copy()
    data = engineer_model_features(data, asset_meta)
    data = add_response_windows(data, stock_actual, price_wide.index)

    base_features = [
        "raw_return", "abnormal_return", "market_return", "broad_shock",
        "log_position_value", "log_units_before", "log_asset_prior_trades",
        "asset_prior_buys", "asset_prior_sells", "log_days_since_asset_trade", "sector",
    ]
    profile_features = ["riskLevel", "investmentCapacity", "customerType", "questionnaire_age_days"]
    behavior_features = [
        "log_prior_transactions", "prior_buy_share", "prior_stock_share", "prior_internet_share",
        "prior_mean_log_value", "log_prior_transactions_90d", "log_prior_unique_assets",
        "log_days_since_any_trade",
    ]
    shock_features = ["log_prior_shock_exposures", "prior_action_rate_smoothed", "prior_sell_rate_smoothed"]
    feature_sets = {
        "M0_event_position": base_features,
        "M1_plus_profile": base_features + profile_features,
        "M2_plus_behavior": base_features + behavior_features,
        "M3_profile_behavior": base_features + profile_features + behavior_features,
        "M4_plus_prior_shocks": base_features + profile_features + behavior_features + shock_features,
    }

    action_metrics, action_predictions, action_models = [], {}, {}
    for name, features in feature_sets.items():
        model, metrics, y_test, probability = fit_catboost(data, features, "acted", name)
        action_metrics.append(metrics)
        action_predictions[name] = probability
        action_models[name] = model
    pd.DataFrame(action_metrics).to_csv(output_dir / "action_model_metrics.csv", index=False)

    acted_data = data[data.acted == 1].copy()
    sell_metrics = []
    for name, features in feature_sets.items():
        _, metrics, y_test_sell, probability_sell = fit_catboost(acted_data, features, "sold", name)
        metrics["BalancedAcc_05"] = balanced_accuracy_score(y_test_sell, probability_sell >= 0.5)
        metrics["MacroF1_05"] = f1_score(y_test_sell, probability_sell >= 0.5, average="macro")
        sell_metrics.append(metrics)
    pd.DataFrame(sell_metrics).to_csv(output_dir / "sell_model_metrics.csv", index=False)

    risk_description = data.groupby("riskLevel").agg(
        exposures=("acted", "size"), action_rate=("acted", "mean"),
        buy_rate=("bought", "mean"), sell_rate=("sold", "mean"),
    ).reset_index()
    risk_description.to_csv(output_dir / "descriptive_by_risk.csv", index=False)
    data.groupby("year").agg(
        exposures=("acted", "size"), action_rate=("acted", "mean"),
        buy_rate=("bought", "mean"), sell_rate=("sold", "mean"),
    ).reset_index().to_csv(output_dir / "descriptive_by_year.csv", index=False)

    fit_inference_model(data, "acted", sell_stage=False).to_csv(
        output_dir / "action_inference_odds_ratios.csv", index=False
    )
    fit_inference_model(acted_data, "sold", sell_stage=True).to_csv(
        output_dir / "sell_inference_odds_ratios.csv", index=False
    )

    test_mask = data.year == 2022
    test_meta = data.loc[test_mask, ["event_date", "ISIN", "customerID", "acted"]].reset_index(drop=True)
    bootstrap = event_bootstrap_difference(
        test_meta,
        action_predictions["M4_plus_prior_shocks"],
        action_predictions["M3_profile_behavior"],
    )
    rows = []
    for metric, values in bootstrap.items():
        rows.append({"comparison": "M4_minus_M3", "metric": metric, "mean_diff": values[0], "ci_low": values[1], "ci_high": values[2]})
    pd.DataFrame(rows).to_csv(output_dir / "bootstrap_M4_vs_M3.csv", index=False)

    importance = pd.DataFrame({
        "feature": feature_sets["M4_plus_prior_shocks"],
        "importance": action_models["M4_plus_prior_shocks"].get_feature_importance(),
    }).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "action_model_feature_importance.csv", index=False)

    events.to_csv(output_dir / "extreme_return_events.csv", index=False)
    data.to_csv(output_dir / "final_exposure_dataset.csv.gz", index=False, compression="gzip")
    summary = {
        "raw_customers": int(frames["customer"].customerID.nunique()),
        "raw_transactions": int(len(frames["transactions"])),
        "eligible_assets": int(len(eligible_assets)),
        "detected_events": int(len(events)),
        "final_exposures": int(len(data)),
        "final_customers": int(data.customerID.nunique()),
        "actions": int(data.acted.sum()),
        "buys": int(data.bought.sum()),
        "sells": int(data.sold.sum()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("FAR_TRANS_DATA_DIR", "data/raw/FAR-Trans")),
        help="Directory containing the extracted FAR-Trans CSV files.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("./results_rerun"))
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
