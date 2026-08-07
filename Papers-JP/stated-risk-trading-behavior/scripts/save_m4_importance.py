from pathlib import Path
import pandas as pd
from run_analysis import fit_catboost
r=Path(__file__).resolve().parents[1]; d=pd.read_csv(r/'data/derived/final_exposure_dataset.csv.gz')
base=['raw_return','abnormal_return','market_return','broad_shock','log_position_value','log_units_before','log_asset_prior_trades','asset_prior_buys','asset_prior_sells','log_days_since_asset_trade','sector']
profile=['riskLevel','investmentCapacity','customerType','questionnaire_age_days'];behavior=['log_prior_transactions','prior_buy_share','prior_stock_share','prior_internet_share','prior_mean_log_value','log_prior_transactions_90d','log_prior_unique_assets','log_days_since_any_trade'];shock=['log_prior_shock_exposures','prior_action_rate_smoothed','prior_sell_rate_smoothed'];f=base+profile+behavior+shock
m,_,_,_=fit_catboost(d,f,'acted','M4_plus_prior_shocks')
pd.DataFrame({'feature':f,'importance':m.get_feature_importance()}).sort_values('importance',ascending=False).to_csv(r/'results_final/action_model_feature_importance.csv',index=False)
