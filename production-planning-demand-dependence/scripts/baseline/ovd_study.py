import json, math, random, statistics, csv, os, time
from collections import defaultdict
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix



def sheet_rows(D, s):
    h = D[s][0]
    return [dict(zip(h, r)) for r in D[s][1:] if any(x is not None for x in r)]

class InstanceData:
    def __init__(self, D, inst):
        self.inst = inst
        self.problem = next(r for r in sheet_rows(D,'ProblemInstance') if r['probleminstanceid']==inst)
        self.headers = [r for r in sheet_rows(D,'BOMHeader') if r['probleminstanceid']==inst]
        self.bomitems = [r for r in sheet_rows(D,'BOMItem') if r['probleminstanceid']==inst]
        self.costrows = [r for r in sheet_rows(D,'MaterialCost') if r['probleminstanceid']==inst]
        self.caprows = [r for r in sheet_rows(D,'Capacity') if r['probleminstanceid']==inst]
        self.setuprows = [r for r in sheet_rows(D,'SetupMatrix') if r['probleminstanceid']==inst]
        self.demrows = [r for r in sheet_rows(D,'Demand') if r['probleminstanceid']==inst]
        self.materials = [r['materialid'] for r in self.headers]
        self.pidx = {p:i for i,p in enumerate(self.materials)}
        self.machine_of = {r['materialid']:r['machineid'] for r in self.headers}
        self.machines = sorted(set(self.machine_of.values()))
        self.pm = {m:[p for p in self.materials if self.machine_of[p]==m] for m in self.machines}
        self.prodtime = {r['materialid']:float(r['productiontime'] or 0) for r in self.headers}
        self.prodcost = {r['materialid']:float(r['productioncost'] or 0) for r in self.headers}
        cr = {r['materialid']:r for r in self.costrows}
        self.invcost = {p:float(cr[p]['inventoryholding'] or 0) for p in self.materials}
        self.bocost = {p:float(cr[p]['backorder'] or 0) for p in self.materials}
        # Sequence-independent setup: average across materialidto for each materialidfrom, per published data-loading rule.
        setup_by = defaultdict(list)
        for r in self.setuprows:
            setup_by[(r['machineid'],r['materialidfrom'])].append((float(r['setuptime'] or 0), float(r['setupcost'] or 0)))
        self.setuptime = {}
        self.setupcost = {}
        for p in self.materials:
            vals = setup_by.get((self.machine_of[p],p), [])
            if vals:
                self.setuptime[p] = sum(v[0] for v in vals)/len(vals)
                self.setupcost[p] = sum(v[1] for v in vals)/len(vals)
            else:
                self.setuptime[p] = 0.0
                self.setupcost[p] = 0.0
        # capacity is total over 50 official weekly periods
        self.weekly_cap = {r['machineid']:float(r['capacity'])/50.0 for r in self.caprows}
        # BOM: ingredient p -> successor q coefficient r[p,q]
        header_mat = {r['bomheaderid']:r['materialid'] for r in self.headers}
        self.ratio = defaultdict(float)
        for r in self.bomitems:
            q = header_mat.get(r['bomheaderid'])
            p = r['materialid']
            if p in self.pidx and q in self.pidx:
                self.ratio[(p,q)] += float(r['ratio'] or 0) * (1.0 + float(r['scrapvariable'] or 0))
        self.successors = {p:[q for q in self.materials if self.ratio.get((p,q),0)>0] for p in self.materials}
        demanded = sorted(set(r['materialid'] for r in self.demrows))
        self.finished = [p for p in demanded if p in self.pidx]
        self.intermediate = [p for p in self.materials if p not in self.finished]
        # official 50-week horizon: demand dates from start through capacity validity end, one date per week
        start = float(self.problem['planningstartdate'])
        end = max(float(r['validitydateto']) for r in self.caprows)
        dates = sorted(set(float(r['deliverydate']) for r in self.demrows if start <= float(r['deliverydate']) <= end))
        # planning start Monday, delivery dates Thu; this yields 50 dates.
        if len(dates) != 50:
            # fallback first 50 demand dates >= start
            dates = sorted(set(float(r['deliverydate']) for r in self.demrows if float(r['deliverydate']) >= start))[:50]
        self.dates = dates
        demmap = {(r['materialid'],float(r['deliverydate'])):float(r['quantity'] or 0) for r in self.demrows}
        self.demand = np.zeros((len(self.finished),len(self.dates)))
        self.fidx = {p:i for i,p in enumerate(self.finished)}
        for p in self.finished:
            for t,dt in enumerate(self.dates):
                self.demand[self.fidx[p],t] = demmap.get((p,dt),0.0)

    def workload_stats(self):
        # propagation: demand-driven same-period gross requirements through BOM DAG, then capacity by machine.
        P,T=len(self.materials),len(self.dates)
        req=np.zeros((P,T))
        for p in self.finished:
            req[self.pidx[p],:] += self.demand[self.fidx[p],:]
        # topological propagation from finals down ingredients; repeatedly add requirements until stable using DAG depth.
        # ratio maps ingredient->successor, so for each successor q, its production requirement induces ingredient demand.
        # Iterate P times; structure acyclic in supplied data.
        for _ in range(P):
            changed=False
            add=np.zeros_like(req)
            for (p,q),r in self.ratio.items():
                add[self.pidx[p],:] += r*req[self.pidx[q],:]
            # recompute from external finished each iteration would double count, so instead derive by topo recursion below.
            break
        # recursive gross req per material from final demands
        from functools import lru_cache
        @lru_cache(None)
        def coeff_to_final(p):
            out={}
            if p in self.finished:
                out[p]=1.0
            for q in self.successors[p]:
                for f,c in coeff_to_final(q).items():
                    out[f]=out.get(f,0)+self.ratio[(p,q)]*c
            return out
        req=np.zeros((P,T))
        for p in self.materials:
            co=coeff_to_final(p)
            for f,c in co.items():
                req[self.pidx[p],:] += c*self.demand[self.fidx[f],:]
        stats={}
        for m in self.machines:
            load=np.zeros(T)
            for p in self.pm[m]:
                load += self.prodtime[p]*req[self.pidx[p],:]
            u=load/self.weekly_cap[m]
            stats[m]={'mean':float(np.mean(u)),'p90':float(np.quantile(u,.9)),'max':float(np.max(u))}
        return stats


def trailing_mean_forecast(y, origin, H, window=8):
    # y shape F,T; forecast each horizon as same recent-window mean (transparent benchmark).
    lo=max(0,origin-window)
    mu=np.mean(y[:,lo:origin],axis=1) if origin>lo else np.zeros(y.shape[0])
    return np.repeat(mu[:,None],H,axis=1)

def historical_error_blocks(y, origin, H, window=8):
    # candidate direct H-step forecast errors from historical origins j fully observed before current origin.
    blocks=[]
    for j in range(window, origin-H+1):
        fc=trailing_mean_forecast(y,j,H,window)
        blocks.append((j, y[:,j:j+H]-fc))
    return blocks

def make_matched_scenarios(inst, origin, H, S, seed, window=8, treatment='joint'):
    y=inst.demand
    fc=trailing_mean_forecast(y,origin,H,window)
    blocks=historical_error_blocks(y,origin,H,window)
    if not blocks:
        # fallback zero residuals
        base=np.repeat(fc[None,:,:],S,axis=0)
        return base
    rng=random.Random(seed*100003 + origin*997 + H*31 + S)
    if len(blocks)>=S:
        picks=rng.sample(range(len(blocks)),S)
    else:
        picks=[rng.randrange(len(blocks)) for _ in range(S)]
    # base item-path multiset
    err=np.stack([blocks[k][1] for k in picks],axis=0) # S,F,H
    joint=np.maximum(0.0, fc[None,:,:] + err)
    if treatment=='joint': return joint
    ind=np.empty_like(joint)
    for fi in range(len(inst.finished)):
        perm=list(range(S)); rng.shuffle(perm)
        ind[:,fi,:]=joint[perm,fi,:]
    return ind

class MilpBuilder:
    def __init__(self):
        self.c=[]; self.lb=[]; self.ub=[]; self.intg=[]; self.names=[]
        self.rows=[]; self.cols=[]; self.vals=[]; self.clb=[]; self.cub=[]
    def var(self,name,lb=0,ub=np.inf,integer=0,cost=0):
        i=len(self.c); self.c.append(cost); self.lb.append(lb); self.ub.append(ub); self.intg.append(integer); self.names.append(name); return i
    def con(self,coef,lb=-np.inf,ub=np.inf):
        r=len(self.clb)
        for j,v in coef.items():
            if abs(v)>1e-12:
                self.rows.append(r); self.cols.append(j); self.vals.append(v)
        self.clb.append(lb); self.cub.append(ub)
    def solve(self,time_limit=20,mip_gap=0.005):
        A=coo_matrix((self.vals,(self.rows,self.cols)),shape=(len(self.clb),len(self.c))).tocsr()
        res=milp(c=np.asarray(self.c),integrality=np.asarray(self.intg),bounds=Bounds(np.asarray(self.lb),np.asarray(self.ub)),constraints=LinearConstraint(A,np.asarray(self.clb),np.asarray(self.cub)),options={'time_limit':time_limit,'mip_rel_gap':mip_gap,'presolve':True,'disp':False})
        return res

def solve_policy(inst, scenarios, state, prev_l, cap_scale=1.0, time_limit=20, mip_gap=0.005, future_binary=True, shared_future_setup=False):
    # scenarios S,F,H. Common first-period q/su/l; future recourse scenario-specific.
    S,F,H=scenarios.shape; P=len(inst.materials)
    b=MilpBuilder()
    q={}; inv={}; bo={}; su={}; link={}
    # vars by scenario; first-stage equality later.
    for s in range(S):
      for h in range(H):
       for p in inst.materials:
        m=inst.machine_of[p]
        q[(s,p,h)] = b.var(f'q[{s},{p},{h}]',0,np.inf,0,0)
        inv[(s,p,h)] = b.var(f'inv[{s},{p},{h}]',0,np.inf,0,inst.invcost[p]/S)
        bo_ub = 0.0 if p in inst.intermediate else np.inf
        bo[(s,p,h)] = b.var(f'bo[{s},{p},{h}]',0,bo_ub,0,inst.bocost[p]/S)
        intflag=1 if (h==0 or future_binary) else 0
        su[(s,p,h)] = b.var(f'su[{s},{p},{h}]',0,1,intflag,inst.setupcost[p]/S)
        if h < H-1:
            link[(s,p,h)] = b.var(f'l[{s},{p},{h}]',0,1,intflag,0)
    # first-stage nonanticipativity: q/su and link0 equal across scenarios
    for s in range(1,S):
      for p in inst.materials:
        b.con({q[(s,p,0)]:1,q[(0,p,0)]:-1},0,0)
        b.con({su[(s,p,0)]:1,su[(0,p,0)]:-1},0,0)
        if H>1:
            b.con({link[(s,p,0)]:1,link[(0,p,0)]:-1},0,0)
        # Optional exact two-stage robustness: setup/carryover schedule is
        # here-and-now across the look-ahead horizon, while production,
        # inventory and backlog quantities retain scenario recourse.
        if shared_future_setup:
            for h in range(1,H):
                b.con({su[(s,p,h)]:1,su[(0,p,h)]:-1},0,0)
                if h < H-1:
                    b.con({link[(s,p,h)]:1,link[(0,p,h)]:-1},0,0)
    # constraints
    fpos={p:i for i,p in enumerate(inst.finished)}
    # Effective horizon demand by scenario, with finished-goods demand
    # propagated upstream through the BOM.  For the nonanticipative current
    # action we later use the maximum across scenarios, so a low-demand
    # scenario cannot incorrectly cap a hedge chosen for a high-demand one.
    need_by_s=[]
    for ss in range(S):
        memo={}
        def _need_ss(mat):
            if mat in memo:
                return memo[mat]
            primary=float(np.sum(scenarios[ss,fpos[mat],:])) if mat in fpos else 0.0
            downstream=sum(inst.ratio[(mat,succ)]*_need_ss(succ) for succ in inst.successors[mat])
            extra_bo=float(state['bo'].get(mat,0.0)) if mat in inst.finished else 0.0
            memo[mat]=max(0.0,primary+downstream+extra_bo)
            return memo[mat]
        for mat in inst.materials:
            _need_ss(mat)
        need_by_s.append(memo)
    common_need={p:max(need_by_s[ss][p] for ss in range(S)) for p in inst.materials}
    for s in range(S):
      for h in range(H):
        # balances
        for p in inst.materials:
            coef={q[(s,p,h)]:1, inv[(s,p,h)]:-1, bo[(s,p,h)]:1}
            rhs=0.0
            if h==0:
                rhs = -state['inv'].get(p,0.0) + state['bo'].get(p,0.0)
            else:
                coef[inv[(s,p,h-1)]]=coef.get(inv[(s,p,h-1)],0)+1
                coef[bo[(s,p,h-1)]]=coef.get(bo[(s,p,h-1)],0)-1
            # demand + successor consumption on RHS: q + invprev - boprev + bo - inv - consumption = demand
            demand=float(scenarios[s,fpos[p],h]) if p in fpos else 0.0
            for succ in inst.successors[p]:
                coef[q[(s,succ,h)]]=coef.get(q[(s,succ,h)],0)-inst.ratio[(p,succ)]
            # For h0: q - inv + bo -cons = demand - invprev + boprev
            # For h>0: q+invprev-boprev-inv+bo-cons=demand
            if h==0:
                b.con(coef, demand - state['inv'].get(p,0.0) + state['bo'].get(p,0.0), demand - state['inv'].get(p,0.0) + state['bo'].get(p,0.0))
            else:
                b.con(coef,demand,demand)
        # capacity and setup-production linkage
        for m in inst.machines:
            coef={}
            for p in inst.pm[m]:
                coef[su[(s,p,h)]]=inst.setuptime[p]
                coef[q[(s,p,h)]]=inst.prodtime[p]
            b.con(coef,-np.inf,inst.weekly_cap[m]*cap_scale)
        # Tight Big-M: the published formulation recommends the smaller
        # of remaining effective demand and the machine-capacity bound.
        for p in inst.materials:
            m=inst.machine_of[p]
            cap=inst.weekly_cap[m]*cap_scale
            demand_bound = common_need[p] if h==0 else need_by_s[s][p]
            M=min(cap/max(inst.prodtime[p],1e-9), max(demand_bound,1e-9))
            coef={q[(s,p,h)]:1, su[(s,p,h)]:-M}
            if h==0:
                rhs=M*float(prev_l.get(p,0))
                b.con(coef,-np.inf,rhs)
            else:
                coef[link[(s,p,h-1)]]=-M
                b.con(coef,-np.inf,0)
        # link constraints for boundaries h->h+1
        if h < H-1:
          for m in inst.machines:
            b.con({link[(s,p,h)]:1 for p in inst.pm[m]},-np.inf,1)
          for p in inst.materials:
            coef={link[(s,p,h)]:1, su[(s,p,h)]:-1}
            if h==0:
                b.con(coef,-np.inf,float(prev_l.get(p,0)))
            else:
                coef[link[(s,p,h-1)]]=-1
                b.con(coef,-np.inf,0)
          # Eq (7): l_q,h + l_q,h-1 - su_q,h + su_r,h <= 2
          for m in inst.machines:
            items=inst.pm[m]
            for qmat in items:
                for rmat in items:
                    if qmat==rmat: continue
                    coef={link[(s,qmat,h)]:1, su[(s,qmat,h)]:-1, su[(s,rmat,h)]:1}
                    if h==0:
                        ub=2-float(prev_l.get(qmat,0))
                    else:
                        coef[link[(s,qmat,h-1)]]=1; ub=2
                    b.con(coef,-np.inf,ub)
    res=b.solve(time_limit=time_limit,mip_gap=mip_gap)
    if res.x is None:
        return {'success':False,'message':str(res.message),'status':int(res.status)}
    x=res.x
    action={'q':{p:float(x[q[(0,p,0)]]) for p in inst.materials},
            'su':{p:int(round(float(x[su[(0,p,0)]]))) for p in inst.materials},
            'link':{p:(int(round(float(x[link[(0,p,0)]]))) if H>1 else 0) for p in inst.materials}}
    return {'success':True,'fun':float(res.fun),'gap':float(getattr(res,'mip_gap',np.nan) if getattr(res,'mip_gap',None) is not None else np.nan),'action':action,'status':int(res.status),'message':str(res.message)}

def execute_week(inst, action, state, actual_demand):
    # actual_demand dict for final goods. Same-period BOM consumption by executed successor production.
    new_inv={}; new_bo={}
    for p in inst.materials:
        cons=sum(inst.ratio[(p,q)]*action['q'][q] for q in inst.successors[p])
        d=actual_demand.get(p,0.0)
        net=state['inv'].get(p,0.0)-state['bo'].get(p,0.0)+action['q'][p]-d-cons
        if p in inst.intermediate:
            # numerical tolerance only; MIP should ensure nonnegative under all scenarios, but actual may differ from scenarios.
            new_inv[p]=max(net,0.0); new_bo[p]=0.0
        else:
            new_inv[p]=max(net,0.0); new_bo[p]=max(-net,0.0)
    setup=sum(inst.setupcost[p]*action['su'][p] for p in inst.materials)
    holding=sum(inst.invcost[p]*new_inv[p] for p in inst.materials)
    backorder=sum(inst.bocost[p]*new_bo[p] for p in inst.finished)
    return {'inv':new_inv,'bo':new_bo}, {'setup':setup,'holding':holding,'backorder':backorder,'total':setup+holding+backorder}

def run_policy(inst, treatment, seed, S=8,H=2,window=8,calibration=20,cap_scale=1.0,time_limit=20,mip_gap=.005,future_binary=True):
    state={'inv':{p:0.0 for p in inst.materials},'bo':{p:0.0 for p in inst.materials}}
    prev_l={p:0 for p in inst.materials}
    costs={'setup':0.0,'holding':0.0,'backorder':0.0,'total':0.0}
    gaps=[]; failures=[]; weekly=[]
    for origin in range(calibration, len(inst.dates)):
        HH=min(H,len(inst.dates)-origin)
        scenarios=make_matched_scenarios(inst,origin,HH,S,seed,window,treatment)
        sol=solve_policy(inst,scenarios,state,prev_l,cap_scale,time_limit,mip_gap,future_binary)
        if not sol['success']:
            failures.append((origin,sol['message'])); return {'success':False,'failures':failures}
        action=sol['action']; gaps.append(sol['gap'])
        actual={p:float(inst.demand[inst.fidx[p],origin]) for p in inst.finished}
        state,cc=execute_week(inst,action,state,actual)
        prev_l=action['link']
        for k in costs: costs[k]+=cc[k]
        weekly.append({'origin':origin,'gap':sol['gap'],**cc,'backlog_qty':sum(state['bo'][p] for p in inst.finished),'inventory_qty':sum(state['inv'][p] for p in inst.materials)})
    return {'success':True,'costs':costs,'max_gap':max(gaps) if gaps else 0,'mean_gap':float(np.nanmean(gaps)) if gaps else 0,'weekly':weekly,'final_backlog':sum(state['bo'][p] for p in inst.finished)}

def paired_run(inst,seed,**kwargs):
    j=run_policy(inst,'joint',seed,**kwargs)
    i=run_policy(inst,'independent',seed,**kwargs)
    if not j.get('success') or not i.get('success'):
        return {'instance':inst.inst,'seed':seed,'success':False,'joint':j,'independent':i}
    cj=j['costs']['total']; ci=i['costs']['total']
    return {'instance':inst.inst,'seed':seed,'success':True,'joint_cost':cj,'ind_cost':ci,'ovd_pct':100*(ci-cj)/cj if cj else np.nan,
            'joint_setup':j['costs']['setup'],'ind_setup':i['costs']['setup'],'joint_holding':j['costs']['holding'],'ind_holding':i['costs']['holding'],
            'joint_backorder':j['costs']['backorder'],'ind_backorder':i['costs']['backorder'],'joint_max_gap':j['max_gap'],'ind_max_gap':i['max_gap'],
            'joint_final_backlog':j['final_backlog'],'ind_final_backlog':i['final_backlog']}

def matched_marginal_check(inst,origin=20,H=2,S=8,seed=1,window=8):
    J=make_matched_scenarios(inst,origin,H,S,seed,window,'joint')
    I=make_matched_scenarios(inst,origin,H,S,seed,window,'independent')
    maxdiff=0.0
    for f in range(J.shape[1]):
        a=sorted([tuple(np.round(J[s,f,:],10)) for s in range(S)])
        b=sorted([tuple(np.round(I[s,f,:],10)) for s in range(S)])
        for aa,bb in zip(a,b): maxdiff=max(maxdiff,max(abs(x-y) for x,y in zip(aa,bb)))
    def mean_abs_corr(X):
        # collapse horizon 0 for cross-item illustration
        A=X[:,:,0]
        if A.shape[1]<2:return 0.0
        C=np.corrcoef(A,rowvar=False)
        vals=[abs(C[a,b]) for a in range(C.shape[0]) for b in range(a+1,C.shape[1]) if np.isfinite(C[a,b])]
        return float(np.mean(vals)) if vals else 0.0
    return {'max_marginal_path_diff':maxdiff,'joint_mean_abs_corr':mean_abs_corr(J),'ind_mean_abs_corr':mean_abs_corr(I)}

def load_all(path=None):
    """Load the prepared workbook JSON from an explicit path or TABLET_DATA_JSON.

    The repository intentionally ships no data files.
    """
    candidate = path or os.environ.get("TABLET_DATA_JSON")
    if not candidate:
        raise FileNotFoundError(
            "No prepared data file configured. Pass a path to load_all() or set TABLET_DATA_JSON."
        )
    with open(candidate, encoding="utf-8") as handle:
        return json.load(handle)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--instance',default='SET4'); ap.add_argument('--seed',type=int,default=1); ap.add_argument('--S',type=int,default=8); ap.add_argument('--H',type=int,default=2); ap.add_argument('--window',type=int,default=8); ap.add_argument('--calibration',type=int,default=20); ap.add_argument('--cap-scale',type=float,default=1.0); ap.add_argument('--time-limit',type=float,default=20); ap.add_argument('--gap',type=float,default=.005); ap.add_argument('--relax-future',action='store_true')
    a=ap.parse_args(); D=load_all(); inst=InstanceData(D,a.instance)
    print('stats',inst.workload_stats()); print('matched',matched_marginal_check(inst,origin=a.calibration,H=min(a.H,2),S=a.S,seed=a.seed,window=a.window))
    t=time.time(); r=paired_run(inst,a.seed,S=a.S,H=a.H,window=a.window,calibration=a.calibration,cap_scale=a.cap_scale,time_limit=a.time_limit,mip_gap=a.gap,future_binary=not a.relax_future); print(json.dumps(r,indent=2)); print('elapsed',time.time()-t)
