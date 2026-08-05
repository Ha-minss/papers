from pathlib import Path
import numpy as np, pandas as pd
from run_analysis import fit_catboost
ROOT=Path(__file__).resolve().parents[1]
data=pd.read_csv(ROOT/'data/derived/final_exposure_dataset.csv.gz',parse_dates=['event_date'])
base=['raw_return','abnormal_return','market_return','broad_shock','log_position_value','log_units_before','log_asset_prior_trades','asset_prior_buys','asset_prior_sells','log_days_since_asset_trade','sector']
profile=['riskLevel','investmentCapacity','customerType','questionnaire_age_days']
behavior=['log_prior_transactions','prior_buy_share','prior_stock_share','prior_internet_share','prior_mean_log_value','log_prior_transactions_90d','log_prior_unique_assets','log_days_since_any_trade']
shock=['log_prior_shock_exposures','prior_action_rate_smoothed','prior_sell_rate_smoothed']
sets={'M0_event_position':base,'M1_plus_profile':base+profile,'M2_plus_behavior':base+behavior,'M3_profile_behavior':base+profile+behavior,'M4_plus_prior_shocks':base+profile+behavior+shock,'R1_risk_only':base+['riskLevel'],'R2_behavior_plus_risk':base+behavior+['riskLevel']}
mask=data.year==2022
pred=data.loc[mask,['event_date','ISIN','customerID','acted']].reset_index(drop=True)
rows=[]
for name,features in sets.items():
    model,m,y,p=fit_catboost(data,features,'acted',name); pred[name]=p; rows.append(m)
pred.to_csv(ROOT/'results_final/action_test_predictions.csv.gz',index=False,compression='gzip')
pd.DataFrame(rows).to_csv(ROOT/'results_final/action_model_metrics.csv',index=False)
print(pred.shape)
