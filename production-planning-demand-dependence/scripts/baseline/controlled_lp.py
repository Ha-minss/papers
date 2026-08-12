import json
import sys
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from .ovd_study import InstanceData, load_all, make_matched_scenarios

class LPB:
    def __init__(self):
        self.c=[]; self.lb=[]; self.ub=[]; self.rows=[]; self.cols=[]; self.vals=[]; self.clb=[]; self.cub=[]
    def var(self,lb=0,ub=np.inf,cost=0):
        i=len(self.c); self.c.append(cost); self.lb.append(lb); self.ub.append(ub); return i
    def con(self,coef,lb=-np.inf,ub=np.inf):
        r=len(self.clb)
        for j,v in coef.items():
            if abs(v)>1e-12: self.rows.append(r); self.cols.append(j); self.vals.append(v)
        self.clb.append(lb); self.cub.append(ub)
    def solve(self):
        A=coo_matrix((self.vals,(self.rows,self.cols)),shape=(len(self.clb),len(self.c))).tocsr()
        return milp(c=np.array(self.c),integrality=np.zeros(len(self.c),dtype=int),bounds=Bounds(np.array(self.lb),np.array(self.ub)),constraints=LinearConstraint(A,np.array(self.clb),np.array(self.cub)),options={'presolve':True,'disp':False})

def solve_policy(inst,scen,state,cap_scale=1.0):
    S,F,H=scen.shape; b=LPB(); q={}; inv={}; bo={}; fpos={p:i for i,p in enumerate(inst.finished)}
    for s in range(S):
      for h in range(H):
       for p in inst.materials:
        q[s,p,h]=b.var()
        inv[s,p,h]=b.var(cost=inst.invcost[p]/S)
        bo[s,p,h]=b.var(ub=(0 if p in inst.intermediate else np.inf),cost=inst.bocost[p]/S)
    # nonanticipative current production
    for s in range(1,S):
      for p in inst.materials: b.con({q[s,p,0]:1,q[0,p,0]:-1},0,0)
    for s in range(S):
      for h in range(H):
       for p in inst.materials:
        coef={q[s,p,h]:1,inv[s,p,h]:-1,bo[s,p,h]:1}
        if h>0:
            coef[inv[s,p,h-1]]=1; coef[bo[s,p,h-1]]=-1
        d=float(scen[s,fpos[p],h]) if p in fpos else 0.0
        for succ in inst.successors[p]: coef[q[s,succ,h]]=coef.get(q[s,succ,h],0)-inst.ratio[(p,succ)]
        rhs=d - (state['inv'].get(p,0)-state['bo'].get(p,0) if h==0 else 0)
        b.con(coef,rhs,rhs)
       for m in inst.machines:
        coef={q[s,p,h]:inst.prodtime[p] for p in inst.pm[m]}
        b.con(coef,-np.inf,inst.weekly_cap[m]*cap_scale)
    res=b.solve()
    if res.x is None: return None
    return {p:float(res.x[q[0,p,0]]) for p in inst.materials}

def execute(inst,action,state,actual):
    ni={}; nb={}
    for p in inst.materials:
        cons=sum(inst.ratio[(p,s)]*action[s] for s in inst.successors[p])
        net=state['inv'].get(p,0)-state['bo'].get(p,0)+action[p]-actual.get(p,0)-cons
        if p in inst.intermediate:
            # exact LP balance should keep these nonnegative
            ni[p]=max(net,0); nb[p]=0
        else:
            ni[p]=max(net,0); nb[p]=max(-net,0)
    hold=sum(inst.invcost[p]*ni[p] for p in inst.materials)
    back=sum(inst.bocost[p]*nb[p] for p in inst.finished)
    return {'inv':ni,'bo':nb},{'holding':hold,'backorder':back,'total':hold+back}

def run(inst,treatment,seed,S=20,H=4,window=8,cal=20,cap=1.0):
    state={'inv':{p:0. for p in inst.materials},'bo':{p:0. for p in inst.materials}}
    tot={'holding':0.,'backorder':0.,'total':0.}
    for origin in range(cal,50):
        hh=min(H,50-origin)
        scen=make_matched_scenarios(inst,origin,hh,S,seed,window,treatment)
        act=solve_policy(inst,scen,state,cap)
        if act is None: return None
        actual={p:float(inst.demand[inst.fidx[p],origin]) for p in inst.finished}
        state,c=execute(inst,act,state,actual)
        for k in tot:tot[k]+=c[k]
    return tot

def pair(inst,seed,S=20,H=4,window=8,cal=20,cap=1.0):
    j=run(inst,'joint',seed,S,H,window,cal,cap); i=run(inst,'independent',seed,S,H,window,cal,cap)
    if j is None or i is None:return None
    return {'instance':inst.inst,'seed':seed,'S':S,'H':H,'cap_scale':cap,'joint_cost':j['total'],'ind_cost':i['total'],'joint_holding':j['holding'],'ind_holding':i['holding'],'joint_backorder':j['backorder'],'ind_backorder':i['backorder'],'ovd_pct':100*(i['total']-j['total'])/j['total'] if j['total'] else 0}

if __name__=='__main__':
    D=load_all(); inst=InstanceData(D,sys.argv[1] if len(sys.argv)>1 else 'SET4'); seed=int(sys.argv[2]) if len(sys.argv)>2 else 1
    t=time.time(); print(json.dumps(pair(inst,seed),indent=2)); print('elapsed',time.time()-t)
