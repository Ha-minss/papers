from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from lightgbm import LGBMClassifier, early_stopping
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss

from run_analysis import event_bootstrap_difference

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / 'data' / 'derived'
EXTERNAL = ROOT / 'data' / 'external'
OUT = ROOT / 'results_external'
EXTERNAL.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

BASE_FEATURES = [
    'raw_return','abnormal_return','market_return','broad_shock',
    'log_position_value','log_units_before','log_asset_prior_trades',
    'asset_prior_buys','asset_prior_sells','log_days_since_asset_trade','sector',
]
PROFILE_FEATURES = ['riskLevel','investmentCapacity','customerType','questionnaire_age_days']
BEHAVIOR_FEATURES = [
    'log_prior_transactions','prior_buy_share','prior_stock_share','prior_internet_share',
    'prior_mean_log_value','log_prior_transactions_90d','log_prior_unique_assets',
    'log_days_since_any_trade',
]
SHOCK_FEATURES = ['log_prior_shock_exposures','prior_action_rate_smoothed','prior_sell_rate_smoothed']
FEATURE_SETS = {
    'M0_event_position': BASE_FEATURES,
    'M1_plus_profile': BASE_FEATURES + PROFILE_FEATURES,
    'M3_profile_behavior': BASE_FEATURES + PROFILE_FEATURES + BEHAVIOR_FEATURES,
    'M4_plus_prior_shocks': BASE_FEATURES + PROFILE_FEATURES + BEHAVIOR_FEATURES + SHOCK_FEATURES,
}

# High-confidence official ATHEX corporate-action dates that overlap detected events.
# Exclusion window is exact event date plus adjacent trading day where the event date in
# FAR-Trans can lag an effective corporate-action date by one session.
CONFIRMED_ACTIONS = [
    {
        'ISIN':'GRS001003037','effective_date':'2021-04-29','issuer':'ATTICA BANK S.A.',
        'action':'Trading resumption after suspension / adjusted trading conditions',
        'source':'ATHEX Securities Market Information Bulletin and issuer corporate-actions archive',
        'source_url':'https://www.athexgroup.gr/en/market-data/issuers/50/corporate-actions',
        'screen_window_days':1,
    },
    {
        'ISIN':'GRS001003037','effective_date':'2021-09-30','issuer':'ATTICA BANK S.A.',
        'action':'Reverse share split and adjusted start price',
        'source':'ATHEX Securities Market Information Bulletin 27/09/2021',
        'source_url':'https://www.athexgroup.gr/en/more-options/announcements/securities-market-information-bulletin-27092021',
        'screen_window_days':1,
    },
    {
        'ISIN':'GRS001003037','effective_date':'2021-11-22','issuer':'ATTICA BANK S.A.',
        'action':'Ex-rights trading / share capital increase price adjustment',
        'source':'ATHEX clarification of ATTICA BANK share-price adjustment',
        'source_url':'https://www.athexgroup.gr/en/node/704182',
        'screen_window_days':1,
    },
    {
        'ISIN':'GRS014003032','effective_date':'2021-04-19','issuer':'PIRAEUS FINANCIAL HOLDINGS S.A.',
        'action':'Capital restructuring / share capital increase sequence',
        'source':'ATHEX Piraeus issuer corporate-actions archive',
        'source_url':'https://www.athexgroup.gr/en/market-data/issuers/63/corporate-actions',
        'screen_window_days':2,
    },
]


def load_inputs(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(DERIVED / 'final_exposure_dataset.csv.gz', dtype={'ISIN':str})
    data['event_date'] = pd.to_datetime(data['event_date'])
    data['year'] = data['event_date'].dt.year
    asset = pd.read_csv(raw_dir / 'asset_information.csv', dtype={'ISIN':str})
    asset['timestamp'] = pd.to_datetime(asset['timestamp'], errors='coerce')
    prices = pd.read_csv(raw_dir / 'close_prices.csv', dtype={'ISIN':str})
    prices['timestamp'] = pd.to_datetime(prices['timestamp'], errors='coerce')
    original_events = data[['ISIN','event_date','raw_return','abnormal_return','market_return','broad_shock']].drop_duplicates()
    return data, asset, prices, original_events


def build_price_panel(asset: pd.DataFrame, prices: pd.DataFrame):
    latest = asset.sort_values('timestamp').drop_duplicates('ISIN', keep='last')
    isins = set(latest.query("assetCategory == 'Stock' and marketID == 'XATH'").ISIN)
    p = prices[prices.ISIN.isin(isins)].sort_values(['ISIN','timestamp'])
    wide = p.pivot(index='timestamp', columns='ISIN', values='closePrice').sort_index()
    returns = wide.pct_change(fill_method=None)
    coverage = pd.DataFrame({
        'n_prices':wide.notna().sum(),
        'nonzero_return_share':(returns.abs()>1e-12).sum()/returns.notna().sum(),
    })
    eligible = coverage.query('n_prices >= 500 and nonzero_return_share >= 0.30').index
    return wide, returns[eligible], eligible


def proxy_series(returns: pd.DataFrame) -> Dict[str,pd.Series]:
    def trimmed(row):
        x = row.dropna().to_numpy()
        return float(trim_mean(x, 0.10)) if len(x) else np.nan
    winsor = returns.clip(lower=returns.quantile(0.02, axis=1), upper=returns.quantile(0.98, axis=1), axis=0)
    return {
        'median':returns.median(axis=1, skipna=True),
        'equal_weight_mean':returns.mean(axis=1, skipna=True),
        'trimmed_mean_10pct':returns.apply(trimmed, axis=1),
        'winsorized_mean_2pct':winsor.mean(axis=1, skipna=True),
    }


def detect_with_market_proxy(returns: pd.DataFrame, market: pd.Series, proxy_name: str) -> pd.DataFrame:
    shifted_r = returns.shift(11)
    shifted_m = market.shift(11)
    mean_r = shifted_r.rolling(250,min_periods=200).mean()
    mean_m = shifted_m.rolling(250,min_periods=200).mean()
    mean_prod = shifted_r.mul(shifted_m,axis=0).rolling(250,min_periods=200).mean()
    cov = mean_prod - mean_r.mul(mean_m,axis=0)
    var_m = shifted_m.rolling(250,min_periods=200).var(ddof=0)
    beta = cov.div(var_m,axis=0)
    alpha = mean_r - beta.mul(mean_m,axis=0)
    expected = alpha + beta.mul(market,axis=0)
    abnormal = returns - expected
    q01 = abnormal.shift(11).rolling(250,min_periods=200).quantile(0.01)
    mask = (returns <= -0.05) & (abnormal <= q01)
    daily_share = mask.sum(axis=1)/len(returns.columns)
    rows=[]
    for isin in returns.columns:
        last=-10000
        dates = mask.index[mask[isin].fillna(False)]
        for date in dates:
            idx=mask.index.get_loc(date)
            if idx-last>=20:
                rows.append({
                    'ISIN':str(isin),'event_date':date,'proxy':proxy_name,
                    'raw_return':float(returns.at[date,isin]),
                    'abnormal_return':float(abnormal.at[date,isin]),
                    'market_return':float(market.at[date]),
                    'broad_shock':bool(daily_share.at[date]>=0.05),
                })
                last=idx
    return pd.DataFrame(rows)


def build_market_proxy_validation(asset: pd.DataFrame, prices: pd.DataFrame, original_events: pd.DataFrame):
    wide, returns, eligible = build_price_panel(asset, prices)
    event_frames=[]
    for name, series in proxy_series(returns).items():
        event_frames.append(detect_with_market_proxy(returns, series, name))
    alt = pd.concat(event_frames,ignore_index=True)
    alt['event_id']=alt['ISIN']+'|'+alt['event_date'].dt.strftime('%Y-%m-%d')
    counts=alt.groupby('event_id').proxy.nunique().rename('proxy_count').reset_index()
    original=original_events.copy()
    original['event_id']=original.ISIN+'|'+original.event_date.dt.strftime('%Y-%m-%d')
    original_ids=set(original.event_id)
    overlap=[]
    for name,g in alt.groupby('proxy'):
        ids=set(g.event_id)
        overlap.append({
            'proxy':name,'n_events':len(ids),'overlap_original':len(ids&original_ids),
            'original_recall':len(ids&original_ids)/len(original_ids),
            'jaccard':len(ids&original_ids)/len(ids|original_ids),
        })
    overlap=pd.DataFrame(overlap)
    overlap.to_csv(OUT/'market_proxy_event_overlap.csv',index=False)
    alt.to_csv(OUT/'market_proxy_detected_events.csv',index=False)
    counts.to_csv(OUT/'market_proxy_consensus_counts.csv',index=False)
    consensus_ids=set(counts.loc[counts.proxy_count>=3,'event_id'])
    return consensus_ids, overlap, alt, wide


def build_action_screen(price_wide: pd.DataFrame, original_events: pd.DataFrame):
    actions=pd.DataFrame(CONFIRMED_ACTIONS)
    actions['effective_date']=pd.to_datetime(actions.effective_date)
    actions.to_csv(EXTERNAL/'confirmed_corporate_action_event_screen.csv',index=False)

    screen_ids=set()
    matched=[]
    for action in actions.itertuples(index=False):
        ev=original_events[original_events.ISIN.eq(action.ISIN)].copy()
        distance=(ev.event_date-action.effective_date).abs().dt.days
        hit=ev.loc[distance<=action.screen_window_days]
        for r in hit.itertuples(index=False):
            eid=f'{r.ISIN}|{r.event_date:%Y-%m-%d}'
            screen_ids.add(eid)
            matched.append({
                'event_id':eid,'ISIN':r.ISIN,'event_date':r.event_date,
                'raw_return':r.raw_return,'matched_action_date':action.effective_date,
                'issuer':action.issuer,'action':action.action,'source':action.source,
                'source_url':action.source_url,
            })
    matched=pd.DataFrame(matched)
    matched.to_csv(OUT/'confirmed_action_matched_events.csv',index=False)

    # Previous observed-price gap: identifies suspension/resumption and stale-price events.
    gap_rows=[]
    for r in original_events.itertuples(index=False):
        if r.ISIN not in price_wide.columns or r.event_date not in price_wide.index:
            continue
        series=price_wide[r.ISIN].dropna()
        pos=series.index.searchsorted(r.event_date)
        if pos<=0 or pos>=len(series.index):
            gap=np.nan
        else:
            gap=(series.index[pos]-series.index[pos-1]).days
        gap_rows.append({'event_id':f'{r.ISIN}|{r.event_date:%Y-%m-%d}','previous_price_gap_days':gap})
    gaps=pd.DataFrame(gap_rows)
    gaps.to_csv(OUT/'event_previous_price_gaps.csv',index=False)
    return screen_ids, gaps


def _prepare_lgbm(data: pd.DataFrame, features: List[str]):
    train_mask=data.year<=2020; val_mask=data.year==2021; test_mask=data.year==2022
    all_x=data[features].copy()
    categorical=[]
    for c in features:
        if all_x[c].dtype==object:
            categorical.append(c)
            all_x[c]=all_x[c].fillna('Missing').astype('category')
        else:
            all_x[c]=all_x[c].replace([np.inf,-np.inf],np.nan)
            med=all_x.loc[train_mask,c].median()
            all_x[c]=all_x[c].fillna(med)
    return all_x.loc[train_mask],all_x.loc[val_mask],all_x.loc[test_mask],train_mask,val_mask,test_mask,categorical


def run_models_for_scenario(data: pd.DataFrame, scenario: str) -> Tuple[pd.DataFrame,pd.DataFrame]:
    metrics=[]; predictions={}
    for name,features in FEATURE_SETS.items():
        xtr,xv,xt,tm,vm,sm,cats=_prepare_lgbm(data,features)
        ytr=data.loc[tm,'acted'].astype(int); yv=data.loc[vm,'acted'].astype(int); yt=data.loc[sm,'acted'].astype(int)
        model=LGBMClassifier(
            n_estimators=300, learning_rate=0.035, num_leaves=31, max_depth=-1,
            min_child_samples=30, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=2.0, class_weight='balanced', random_state=42,
            verbosity=-1, n_jobs=4,
        )
        model.fit(xtr,ytr,categorical_feature=cats,eval_set=[(xv,yv)],eval_metric='auc',callbacks=[early_stopping(35,verbose=False)])
        pv=model.predict_proba(xv)[:,1]; pt=model.predict_proba(xt)[:,1]
        eps=1e-6
        lv=np.log(np.clip(pv,eps,1-eps)/(1-np.clip(pv,eps,1-eps))).reshape(-1,1)
        lt=np.log(np.clip(pt,eps,1-eps)/(1-np.clip(pt,eps,1-eps))).reshape(-1,1)
        cal=LogisticRegression(max_iter=1000).fit(lv,yv)
        prob=cal.predict_proba(lt)[:,1]
        row={
            'scenario':scenario,'model_short':name,'model_family':'LightGBM robustness rerun',
            'n_train':len(ytr),'n_val':len(yv),'n_test':len(yt),'test_prevalence':float(yt.mean()),
            'PR_AUC':average_precision_score(yt,prob),'ROC_AUC':roc_auc_score(yt,prob),
            'Brier':brier_score_loss(yt,prob),'LogLoss':log_loss(yt,prob),
            'best_iteration':int(model.best_iteration_ or 300),
        }
        metrics.append(row); predictions[name]=prob
    met=pd.DataFrame(metrics)
    test=data.loc[data.year.eq(2022),['event_date','ISIN','customerID','acted']].reset_index(drop=True)
    boot=event_bootstrap_difference(test,predictions['M4_plus_prior_shocks'],predictions['M3_profile_behavior'],n_boot=300)
    b=[]
    for metric,(mean,low,high) in boot.items():
        b.append({'scenario':scenario,'comparison':'M4_minus_M3','metric':metric,'mean_diff':mean,'ci_low':low,'ci_high':high})
    return met,pd.DataFrame(b)

def main(raw_dir: Path):
    data,asset,prices,events=load_inputs(raw_dir)
    data['event_id']=data.ISIN+'|'+data.event_date.dt.strftime('%Y-%m-%d')
    price_wide,_,_=build_price_panel(asset,prices)
    consensus_ids,overlap,alt,_=build_market_proxy_validation(asset,prices,events)
    confirmed_ids,gaps=build_action_screen(price_wide,events)
    gap_map=gaps.set_index('event_id').previous_price_gap_days
    data['previous_price_gap_days']=data.event_id.map(gap_map)

    severe_ids=set(data.loc[(data.raw_return<=-0.20)&(~data.broad_shock),'event_id'])
    round_limit_ids=set(data.loc[(~data.broad_shock)&(
        (data.raw_return.add(0.20).abs()<=0.0035)|(data.raw_return.add(0.30).abs()<=0.0035)
    ),'event_id'])
    stale_gap_ids=set(data.loc[data.previous_price_gap_days.fillna(0)>7,'event_id'])

    scenarios={
        'original':data,
        'official_action_screened':data[~data.event_id.isin(confirmed_ids)].copy(),
        'proxy_consensus_3of4':data[data.event_id.isin(consensus_ids)].copy(),
        'mechanical_conservative':data[~data.event_id.isin(confirmed_ids|severe_ids|round_limit_ids|stale_gap_ids)].copy(),
    }
    scenario_stats=[]; all_metrics=[]; all_boot=[]
    for name,frame in scenarios.items():
        n_train=(frame.year<=2020).sum(); n_val=(frame.year==2021).sum(); n_test=(frame.year==2022).sum()
        if min(n_train,n_val,n_test)==0 or frame.loc[frame.year==2022,'acted'].nunique()<2:
            continue
        scenario_stats.append({
            'scenario':name,'exposures':len(frame),'customers':frame.customerID.nunique(),
            'events':frame.event_id.nunique(),'actions':int(frame.acted.sum()),
            'buys':int(frame.bought.sum()),'sells':int(frame.sold.sum()),
            'test_exposures':int(n_test),'test_action_rate':float(frame.loc[frame.year==2022,'acted'].mean()),
        })
        m,b=run_models_for_scenario(frame,name)
        all_metrics.append(m); all_boot.append(b)
    stats=pd.DataFrame(scenario_stats)
    metrics=pd.concat(all_metrics,ignore_index=True)
    boots=pd.concat(all_boot,ignore_index=True)
    stats.to_csv(OUT/'external_validation_sample_summary.csv',index=False)
    metrics.to_csv(OUT/'external_validation_model_metrics.csv',index=False)
    boots.to_csv(OUT/'external_validation_bootstrap.csv',index=False)

    screen_summary=pd.DataFrame([
        {'screen':'confirmed_official_actions','n_events':len(confirmed_ids)},
        {'screen':'non_broad_raw_drop_le_20pct','n_events':len(severe_ids)},
        {'screen':'round_20_or_30pct_limit_move','n_events':len(round_limit_ids)},
        {'screen':'previous_price_gap_gt_7_days','n_events':len(stale_gap_ids)},
        {'screen':'proxy_consensus_3of4','n_events':len(consensus_ids)},
    ])
    screen_summary.to_csv(OUT/'event_screen_counts.csv',index=False)

    manifest=pd.DataFrame([
        {
            'source':'Bank of Greece - Share price indices of the Athens Exchange',
            'authority':'Official central-bank publication; underlying source Athens Exchange',
            'coverage':'Historic January 2001-December 2023 plus January 2022 onward',
            'url':'https://www.bankofgreece.gr/en/statistics/financial-markets-and-interest-rates/share-price-indices-of-the-athens-exchange',
            'acquisition_status':'Metadata and document availability verified; binary daily workbook could not be retrieved in the isolated runtime',
        },
        {
            'source':'ATHEX Composite Share Price Index (GD)',
            'authority':'Official Athens Exchange index page',
            'coverage':'Current methodology and identity (ISIN GRI99117A004)',
            'url':'https://www.athexgroup.gr/en/market-data/indices/athex-composite-share-price-index',
            'acquisition_status':'Index identity verified; historical binary time series not retrievable in isolated runtime',
        },
        {
            'source':'FAR-Trans XATH cross-section',
            'authority':'Study dataset',
            'coverage':'2018-2022 daily closing prices',
            'url':'local:data/close_prices.csv',
            'acquisition_status':'Used for median, equal-weight mean, trimmed mean and winsorized-mean proxy validation',
        },
    ])
    manifest.to_csv(EXTERNAL/'market_index_source_manifest.csv',index=False)

    summary={
        'confirmed_action_events_excluded':len(confirmed_ids),
        'severe_non_broad_events':len(severe_ids),
        'round_limit_events':len(round_limit_ids),
        'stale_gap_events':len(stale_gap_ids),
        'consensus_events_3of4':len(consensus_ids),
        'scenarios':stats.to_dict(orient='records'),
        'metrics':metrics[['scenario','model_short','PR_AUC','ROC_AUC','Brier','LogLoss']].to_dict(orient='records'),
    }
    (OUT/'external_validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(os.environ.get('FAR_TRANS_DATA_DIR', 'data/raw/FAR-Trans')),
        help='Directory containing the extracted FAR-Trans CSV files.',
    )
    args = parser.parse_args()
    main(args.data_dir)
