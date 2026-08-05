from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results_final'; FIG=ROOT/'figures_final'; FIG.mkdir(exist_ok=True)
data=pd.read_csv(ROOT/'data/derived/final_exposure_dataset.csv.gz',parse_dates=['event_date'])
data['event_id']=data.event_date.astype(str)+'|'+data.ISIN
pred=pd.read_csv(OUT/'action_test_predictions.csv.gz',parse_dates=['event_date']); pred['event_id']=pred.event_date.astype(str)+'|'+pred.ISIN
y=pred.acted.to_numpy(); model_cols=[c for c in pred if c.startswith('M') or c.startswith('R')]
rows=[]
for m in model_cols:
 p=pred[m].to_numpy(); order=np.argsort(-p)
 for frac in [.05,.10]:
  k=int(np.ceil(len(p)*frac)); ix=order[:k]
  rows.append({'model':m,'top_fraction':frac,'recall':y[ix].sum()/y.sum(),'precision':y[ix].mean(),'captured_actions':int(y[ix].sum()),'flagged':k})
pd.DataFrame(rows).to_csv(OUT/'action_top_fraction_metrics.csv',index=False)
indices={e:np.asarray(ix,dtype=int) for e,ix in pred.groupby('event_id').groups.items()}; eids=np.array(list(indices)); rng=np.random.default_rng(20260801)
def boot(a,b,B=300):
 vals=[]; pa=pred[a].to_numpy(); pb=pred[b].to_numpy()
 for _ in range(B):
  sampled=rng.choice(eids,size=len(eids),replace=True)
  ix=np.concatenate([indices[e] for e in sampled]); yy=y[ix]
  if yy.min()==yy.max(): continue
  vals.append((average_precision_score(yy,pa[ix])-average_precision_score(yy,pb[ix]),brier_score_loss(yy,pa[ix])-brier_score_loss(yy,pb[ix]),roc_auc_score(yy,pa[ix])-roc_auc_score(yy,pb[ix])))
 arr=np.asarray(vals); out=[]
 for j,nm in enumerate(['PR_diff','Brier_diff','AUC_diff']):out.append({'comparison':f'{a} minus {b}','metric':nm,'mean_diff':arr[:,j].mean(),'ci_low':np.quantile(arr[:,j],.025),'ci_high':np.quantile(arr[:,j],.975),'bootstrap_draws':len(arr)})
 return out
br=[]
for a,b in [('M1_plus_profile','M0_event_position'),('M4_plus_prior_shocks','M3_profile_behavior'),('R1_risk_only','M0_event_position'),('R2_behavior_plus_risk','M2_plus_behavior')]: br+=boot(a,b)
pd.DataFrame(br).to_csv(OUT/'bootstrap_metric_differences.csv',index=False)
def cal(m):
 t=pd.DataFrame({'p':pred[m],'y':y});t['decile']=pd.qcut(t.p,10,labels=False,duplicates='drop')+1;z=t.groupby('decile').agg(mean_pred=('p','mean'),observed=('y','mean'),n=('y','size')).reset_index();z['model']=m;return z
caldf=pd.concat([cal('M0_event_position'),cal('M4_plus_prior_shocks')]);caldf.to_csv(OUT/'action_calibration_deciles.csv',index=False)
risk=data.groupby('riskLevel').agg(exposures=('acted','size'),actions=('acted','sum'),action_rate=('acted','mean'),buy_rate=('bought','mean'),sell_rate=('sold','mean')).reset_index();risk['sell_share_among_actors']=risk.sell_rate/risk.action_rate;risk.to_csv(OUT/'descriptive_by_risk.csv',index=False)
yr=data.groupby('year').agg(exposures=('acted','size'),actions=('acted','sum'),action_rate=('acted','mean'),buy_rate=('bought','mean'),sell_rate=('sold','mean')).reset_index();yr.to_csv(OUT/'descriptive_by_year.csv',index=False)
rr=[]
for days,col in [(1,'acted_1d'),(5,'acted'),(10,'acted_10d'),(20,'acted_20d')]:
 for r,g in data.groupby('riskLevel'):rr.append({'window_days':days,'riskLevel':r,'action_rate':g[col].mean(),'n':len(g)})
rr=pd.DataFrame(rr);rr.to_csv(OUT/'response_window_robustness.csv',index=False)
order=['Conservative','Income','Balanced','Aggressive']; r=risk.set_index('riskLevel').loc[order]
plt.figure(figsize=(8,4.6));x=np.arange(4);w=.24;plt.bar(x-w,r.buy_rate*100,w,label='Buy');plt.bar(x,r.sell_rate*100,w,label='Sell');plt.bar(x+w,r.action_rate*100,w,label='Any action');plt.xticks(x,order);plt.ylabel('Share of exposures (%)');plt.title('Observed trading within five days of an extreme negative return');plt.legend();plt.tight_layout();plt.savefig(FIG/'fig2_response_by_risk.png',dpi=180);plt.close()
plt.figure(figsize=(8,4.6));plt.plot(yr.year,yr.action_rate*100,marker='o',label='Any action');plt.plot(yr.year,yr.buy_rate*100,marker='o',label='Buy');plt.plot(yr.year,yr.sell_rate*100,marker='o',label='Sell');plt.xticks(yr.year);plt.ylabel('Share of exposures (%)');plt.title('Observed response rates changed materially across years');plt.legend();plt.tight_layout();plt.savefig(FIG/'fig3_response_by_year.png',dpi=180);plt.close()
metrics=pd.read_csv(OUT/'action_model_metrics.csv').set_index('model'); labels={'M0_event_position':'Event + position','M1_plus_profile':'+ customer profile','M2_plus_behavior':'+ revealed behavior','M3_profile_behavior':'Profile + behavior','M4_plus_prior_shocks':'+ prior event response'};mm=metrics.loc[list(labels)]
plt.figure(figsize=(8,4.7));yy=np.arange(len(mm));plt.barh(yy,mm.PR_AUC);plt.yticks(yy,[labels[i] for i in mm.index]);plt.xlabel('PR-AUC on 2022 holdout');plt.title('Prior observed responses provide the clearest incremental predictive value');
for i,v in enumerate(mm.PR_AUC):plt.text(v+.002,i,f'{v:.3f}',va='center');plt.xlim(0,mm.PR_AUC.max()*1.18);plt.tight_layout();plt.savefig(FIG/'fig4_action_model_performance.png',dpi=180);plt.close()
plt.figure(figsize=(6.5,5));
for m,l in [('M0_event_position','Event + position'),('M4_plus_prior_shocks','Full model')]:
 z=caldf[caldf.model==m];plt.plot(z.mean_pred,z.observed,marker='o',label=l)
mx=max(caldf.mean_pred.max(),caldf.observed.max());plt.plot([0,mx],[0,mx],'--',label='Perfect calibration');plt.xlabel('Mean predicted probability');plt.ylabel('Observed action rate');plt.title('Calibration on the 2022 holdout');plt.legend();plt.tight_layout();plt.savefig(FIG/'fig5_calibration.png',dpi=180);plt.close()
plt.figure(figsize=(7.5,4.6));
for rsk in order:
 z=rr[rr.riskLevel==rsk];plt.plot(z.window_days,z.action_rate*100,marker='o',label=rsk)
plt.xticks([1,5,10,20]);plt.xlabel('Response window (trading days)');plt.ylabel('Any-action rate (%)');plt.title('The non-monotonic risk-profile pattern persists across windows');plt.legend();plt.tight_layout();plt.savefig(FIG/'fig7_response_window_robustness.png',dpi=180);plt.close()
for f in ['fig1_sample_flow.png','fig6_adjusted_odds_ratios.png','fig8_event_timeline.png']:
 src=ROOT/'figures'/f
 if src.exists(): (FIG/f).write_bytes(src.read_bytes())
summary={'exposures':len(data),'customers':int(data.customerID.nunique()),'events':int(data.event_id.nunique()),'actions':int(data.acted.sum()),'buys':int(data.bought.sum()),'sells':int(data.sold.sum()),'test_prevalence':float(y.mean()),'m0_pr_auc':float(metrics.loc['M0_event_position','PR_AUC']),'m4_pr_auc':float(metrics.loc['M4_plus_prior_shocks','PR_AUC'])}
(OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2));print(pd.DataFrame(br).to_string(index=False))
