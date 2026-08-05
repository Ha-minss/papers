from pathlib import Path
import argparse, json, time
import pandas as pd
from run_analysis import load_data, detect_events, prepare_customers_and_transactions, build_exposures, add_behavior_features, engineer_model_features, add_response_windows

def main(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    t=time.time(); print('1 load', flush=True)
    f=load_data(data_dir); print('loaded', round(time.time()-t,1), flush=True)
    t=time.time(); print('2 events', flush=True)
    events, price_wide, _, _, _, eligible_assets=detect_events(f['asset'], f['prices']); print('events',len(events),round(time.time()-t,1),flush=True)
    t=time.time(); print('3 tx/customers', flush=True)
    eligible_customers, stock_actual, _, asset_meta=prepare_customers_and_transactions(f['customer'],f['transactions'],f['asset'],eligible_assets); print('tx',len(stock_actual),round(time.time()-t,1),flush=True)
    t=time.time(); print('4 exposures', flush=True)
    exposures=build_exposures(events,stock_actual,eligible_customers,price_wide); print('exposures',len(exposures),round(time.time()-t,1),flush=True)
    t=time.time(); print('5 behavior', flush=True)
    stock_isins=set(f['asset'].sort_values('timestamp').drop_duplicates('ISIN',keep='last').query("assetCategory == 'Stock'").ISIN)
    data=add_behavior_features(exposures,f['transactions'],eligible_customers,stock_isins)
    data=data[data.questionnaire_age_days >= 0].copy()
    data=engineer_model_features(data,asset_meta)
    data=add_response_windows(data,stock_actual,price_wide.index); print('features',data.shape,round(time.time()-t,1),flush=True)
    events.to_csv(output_dir/'extreme_return_events.csv',index=False)
    data.to_csv(output_dir/'final_exposure_dataset.csv.gz',index=False,compression='gzip')
    summary={'exposures':len(data),'customers':int(data.customerID.nunique()),'events':int((data.event_date.astype(str)+'|'+data.ISIN).nunique()),'actions':int(data.acted.sum()),'buys':int(data.bought.sum()),'sells':int(data.sold.sum())}
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args(); main(a.data_dir,a.output_dir)
