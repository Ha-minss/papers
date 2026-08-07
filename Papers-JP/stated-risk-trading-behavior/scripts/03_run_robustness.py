from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from run_analysis import load_data, prepare_customers_and_transactions, build_exposures

ACTUAL_RISKS=['Conservative','Income','Balanced','Aggressive']

def detect_sigma_events(asset, prices):
    latest=asset.sort_values('timestamp').drop_duplicates('ISIN',keep='last')
    xath=set(latest.query("assetCategory=='Stock' and marketID=='XATH'").ISIN)
    p=prices[prices.ISIN.isin(xath)].sort_values(['ISIN','timestamp'])
    wide=p.pivot(index='timestamp',columns='ISIN',values='closePrice').sort_index(); ret=wide.pct_change(fill_method=None)
    cov=pd.DataFrame({'n':wide.notna().sum(),'nz':(ret.abs()>1e-12).sum()/ret.notna().sum()});eligible=cov.query('n>=500 and nz>=0.30').index
    m=ret[eligible].median(axis=1); sr=ret[eligible].shift(11); sm=m.shift(11)
    mr=sr.rolling(250,min_periods=200).mean(); mm=sm.rolling(250,min_periods=200).mean(); mp=sr.mul(sm,axis=0).rolling(250,min_periods=200).mean(); cv=mp-mr.mul(mm,axis=0); vm=sm.rolling(250,min_periods=200).var(ddof=0); beta=cv.div(vm,axis=0);alpha=mr-beta.mul(mm,axis=0); ar=ret[eligible]-(alpha+beta.mul(m,axis=0))
    mean=ar.shift(11).rolling(250,min_periods=200).mean();sd=ar.shift(11).rolling(250,min_periods=200).std();mask=(ret[eligible]<=-.05)&(ar<=mean-3*sd);share=mask.sum(axis=1)/len(eligible)
    rows=[]
    for isin in eligible:
        last=-10000
        for date in mask.index[mask[isin].fillna(False)]:
            ix=mask.index.get_loc(date)
            if ix-last>=20:
                rows.append((isin,date,float(ret.at[date,isin]),float(ar.at[date,isin]),float(m.at[date]),bool(share.at[date]>=.05)));last=ix
    return pd.DataFrame(rows,columns=['ISIN','event_date','raw_return','abnormal_return','market_return','broad_shock']),wide,eligible

def main(data_dir:Path,project_dir:Path):
    f=load_data(data_dir);events,wide,eligible=detect_sigma_events(f['asset'],f['prices']);cust,tx,_,_=prepare_customers_and_transactions(f['customer'],f['transactions'],f['asset'],eligible);exp=build_exposures(events,tx,cust,wide);exp=exp[exp.questionnaire_age_days>=0].copy();exp['acted']=exp.response.ne('NoAction').astype(int)
    main_data=pd.read_csv(project_dir/'data/derived/final_exposure_dataset.csv.gz')
    rows=[]
    for days,col in [(1,'acted_1d'),(5,'acted'),(10,'acted_10d'),(20,'acted_20d')]:
        for risk,g in main_data.groupby('riskLevel'):rows.append({'check':f'{days}-day response window','riskLevel':risk,'action_rate':g[col].mean(),'n':int(g[col].notna().sum())})
    for label,subset in [('idiosyncratic-only',main_data[~main_data.broad_shock]),('broad-market',main_data[main_data.broad_shock])]:
        for risk,g in subset.groupby('riskLevel'):rows.append({'check':label,'riskLevel':risk,'action_rate':g.acted.mean(),'n':len(subset)})
    for risk,g in exp.groupby('riskLevel'):rows.append({'check':'-3 sigma event definition','riskLevel':risk,'action_rate':g.acted.mean(),'n':len(exp)})
    out=project_dir/'results_final';out.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(out/'robustness_summary.csv',index=False)
    print({'sigma_events':len(events),'sigma_exposures':len(exp),'sigma_actions':int(exp.acted.sum())})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--project-dir',type=Path,default=Path(__file__).resolve().parents[1]);a=ap.parse_args();main(a.data_dir,a.project_dir)
