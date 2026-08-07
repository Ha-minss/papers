from pathlib import Path
import json
import pandas as pd
from run_analysis import fit_catboost

ROOT=Path(__file__).resolve().parents[1]
data=pd.read_csv(ROOT/'data/derived/final_exposure_dataset.csv.gz', parse_dates=['event_date'])
base_features=[
    'raw_return','abnormal_return','market_return','broad_shock',
    'log_position_value','log_units_before','log_asset_prior_trades',
    'asset_prior_buys','asset_prior_sells','log_days_since_asset_trade','sector',
]
profile_features=['riskLevel','investmentCapacity','customerType','questionnaire_age_days']
behavior_features=[
    'log_prior_transactions','prior_buy_share','prior_stock_share','prior_internet_share',
    'prior_mean_log_value','log_prior_transactions_90d','log_prior_unique_assets',
    'log_days_since_any_trade',
]
shock_features=['log_prior_shock_exposures','prior_action_rate_smoothed','prior_sell_rate_smoothed']
feature_sets={
'M0_event_position':base_features,
'M1_plus_profile':base_features+profile_features,
'M2_plus_behavior':base_features+behavior_features,
'M3_profile_behavior':base_features+profile_features+behavior_features,
'M4_plus_prior_shocks':base_features+profile_features+behavior_features+shock_features,
}
rows=[]
for name, features in feature_sets.items():
    _,m,_,_=fit_catboost(data,features,'acted',name)
    rows.append(m)
out=pd.DataFrame(rows)
out.to_csv(ROOT/'analysis/verified_action_model_metrics.csv',index=False)
summary={
    'exposures':len(data), 'customers':int(data.customerID.nunique()),
    'events':int((data.event_date.astype(str)+'|'+data.ISIN).nunique()),
    'actions':int(data.acted.sum()), 'buys':int(data.bought.sum()), 'sells':int(data.sold.sum()),
}
(ROOT/'analysis/verified_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
print(out.to_string(index=False))
