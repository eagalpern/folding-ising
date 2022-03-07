import numpy as np
import pandas as pd
import time
import numba
from numba import njit
import scipy.signal as sps
import random


# =============================================================================
#  ISING MODEL FUNCTIONS
# =============================================================================

@njit(inline="always")
def folding_energy_ss_speed(state: np.array,temp: np.float64, 
                            DHarr: numba.types.Array(np.float64,ndim=2,layout='F') ,
                            DS: numba.types.Array(np.float64,ndim=2,layout='F')):
    H=np.sum(DHarr[np.where(state>0)[0]][:,np.where(state>0)[0]])
    S=np.sum(DS[np.where(state==0)[0]][:,np.where(state==0)[0]])
    G=-H-temp*S 
    # Ferreiro et al (2008) : evo_energy is DH_unfolding>0; S is DS_unfolding>0
    return G


@njit(inline="always")
def montecarlo_speed_complete(evo_energy: numba.types.Array(np.float64,ndim=2,layout='F'),
                temp:np.float64 ,DS: numba.types.Array(np.float64,ndim=2,layout='F'),
                        units_len: np.array,k: np.float64 ,nsteps: np.array ,
                transient: numba.int64, save_each: numba.int64, state: np.array):
    n=evo_energy.shape[0]
    m=np.zeros(0)
    q=np.zeros(0)
   # Es=np.zeros(0)
    stat_av=np.zeros(n)
    e0=folding_energy_ss_speed(state,temp,evo_energy,DS) # initial energy

    for i in range(nsteps):
        # MC loop
        x=np.random.randint(n) # choice random spin 
        state[x] = int(not state[x]) # flip it
        ef=folding_energy_ss_speed(state,temp,evo_energy,DS) # energy

        de=ef-e0 # change in energy
    
        # metropolis criterium
        if de<=0: 
            e0=ef
        else:
            if random.uniform(0,1)<np.exp(-de/(k*temp)):
                e0=ef
            else:
                state[x] = int(not state[x]) # flip it back     
        # save fraction folded
        if i%save_each==0 and i>transient:
            m=np.append(m,np.sum(state*units_len)) # pseudomagnetization * residue
            q=np.append(q,np.sum(state)) # pseudomagnetization * residue
            #Es=np.append(Es,e0)
            stat_av+=state # aca sumar estados
        
        q_hist=np.histogram(q,bins=np.arange(-0.5,n+1.5,1))[0]

        
    return state,np.mean(m)/np.sum(units_len),np.std(m/np.sum(units_len))/np.sqrt(m.shape[0]), stat_av/m.shape[0],q_hist
    # last_state,fraction folded average, fraction folded error, state average, histogram  

def temp_denaturation_first_round(DH,DS,units_len,nrep,j,ts,k,ninst,nsteps_,transient_,save_each_):
                                       
    nsteps=nsteps_[j,nrep]
    transient=transient_[j,nrep]
    save_each=save_each_[j,nrep]
    nsave=int((nsteps-transient)/save_each)-1 
    n=DH.shape[0]

    state=np.array([random.randint(0, 1) for _ in range(n)])

    folded_res=np.zeros(len(ts))
    folded_res_err=np.zeros(len(ts))
    states_av=np.zeros((len(ts),n))
    q_hist=np.zeros((len(ts),n+1))
    
    for t,temp in enumerate(ts):
        state,folded_res[t],folded_res_err[t],states_av[t,:],q_hist[t,:]=montecarlo_speed_complete(DH,ts[t],DS,units_len,k,nsteps,transient,save_each,state)

    
    time.sleep(0.2)     

    return folded_res,folded_res_err,states_av,q_hist


def temp_denaturation_second_round(critical_points,folded_res,folded_res_err,states_av,q_hist,
                                   DH,DS,units_len,nrep,ts,j,k,ninst,nsteps_,transient_,
                                   save_each_,order,extrap,cp_factor):
                                       
    nsteps=nsteps_[j,nrep]
    transient=transient_[j,nrep]
    save_each=save_each_[j,nrep]
    save_critical=save_each_[j,nrep]*cp_factor
    nsteps_critical=save_each_[j,nrep]*cp_factor*ninst[j]
    nsave=int((nsteps-transient)/save_each)-1 
    n=DH.shape[0]

    state=np.array([random.randint(0, 1) for _ in range(n)])

    for t in critical_points:
            print(t)
            state,folded_res[t],folded_res_err[t],states_av[t,:],q_hist[t,:]=montecarlo_speed_complete(DH,ts[t],DS,units_len,k,nsteps_critical,save_critical,save_critical,state)
    
    time.sleep(0.2)     

    return folded_res,folded_res_err,states_av,q_hist


def main_fold_1seq_first_round_i(i_,MRA,out_dir_,Jij,Hi,AAdict,replen,rep_frag_len,j,m,gaps_out,si0,
                                 k,nsteps,transient,save_each,ninst,ntsteps,DT,
                                 ff_file,states_file,q_hist_file,ulf_file,DH_file,DS_file,
                                 tini_=None,tfin_=None,ts_=None,custom_ts=False,ts_auto=True):
   
    DH,nrep,breaks=MRA_to_ising(MRA,Jij,Hi,AAdict,replen,rep_frag_len,j,m,gaps_out)
    DS,units_len=si0_to_DS_units_len(si0,MRA,nrep,breaks,rep_frag_len,j,gaps_out)
    
    if i_==0:
        np.savetxt(ulf_file,units_len,fmt='%i')
        np.savetxt(DH_file,DH.values)
        np.savetxt(DS_file,DS.values)

    if custom_ts:
        ts=ts_
         
    else:   
        if ts_auto:
            #4) correr rapido solo ff(T) en los extremos para ver si pasamos de plegado a desplegado 
            tini,tfin=border_check(DT,DH.values,DS.values,units_len,tini_,tfin_,k,
                                          nsteps[j,nrep],transient[j,nrep],save_each[j,nrep])

        else:
            tini=tini_
            tfin=tfin_

        ts=np.linspace(tini,tfin,ntsteps)

    ff=np.zeros((len(ts),2))
    ff_err=np.zeros((len(ts),2))

    ff[:,0]=ts
    ff_err[:,0]=ts

    allstates=np.zeros((len(ts),len(DH)))
     
    args=DH.to_numpy(),DS.to_numpy(),units_len,nrep,j,ts,k,ninst,nsteps,transient,save_each
    ff[:,1],ff_err[:,1],allstates,q_hist=temp_denaturation_first_round(*args)

    np.savetxt(ff_file+'_err_'+str(i_), ff_err)
    np.savetxt(ff_file+'_'+str(i_), ff)
    np.save(states_file+'_'+str(i_),allstates)
    np.save(q_hist_file+'_'+str(i_), q_hist)
    
    del ff,ff_err,allstates,q_hist
    return i_


def main_fold_1seq_second_round_i(i_,DH_file,DS_file,ulf_file,j,k,nsteps,transient,save_each,ninst,ntsteps,
                                  order,extrap,cp_factor,
                                ff_file,states_file,q_hist_file,custom_cp=False,cps_=None,num_cores=None):
    
    DH=np.loadtxt(DH_file)
    DS=np.loadtxt(DS_file)
    ff=np.loadtxt(ff_file+'_'+str(i_)) 
    ff_err=np.loadtxt(ff_file+'_err_'+str(i_))
    allstates=np.load(states_file+'_'+str(i_)+'.npy')
    q_hist=np.load(q_hist_file+'_'+str(i_)+'.npy')

    units_len=np.loadtxt(ulf_file)
    ts=ff[:,0]
    
    if custom_cp:
        cps=cps_
    else:
        ff_errs=np.zeros((len(ff_err),num_cores))

        for fi in range(num_cores):
            ff_err_i=np.loadtxt(ff_file+'_err_'+str(fi))
            ff_errs[:,fi]=ff_err_i[:,1]
        
        ix=sps.argrelmax(ff_errs.mean(axis=1),order=order)[0]
    
        if len(ix>0):
            critical_points=np.concatenate([np.arange((ix_-extrap),(ix_+extrap+1)) for ix_ in ix])
            critical_points=critical_points[critical_points>=0]
            cps=critical_points[critical_points<len(ts)]
        else:
            cps=[]
        del ff_errs,ff_err_i
        
    if(len(cps)>0):
        print(cps)
        DH.shape[0]
        args=cps,ff[:,1],ff_err[:,1],allstates,q_hist,DH,DS,units_len,DH.shape[0],ts,j,k,ninst,nsteps,transient,save_each,order,extrap,cp_factor
        ff[:,1],ff_err[:,1],allstates,q_hist=temp_denaturation_second_round(*args)

        np.savetxt(ff_file+'_err_'+str(i_), ff_err)
        np.savetxt(ff_file+'_'+str(i_), ff)
        np.save(states_file+'_'+str(i_),allstates)
        np.save(q_hist_file+'_'+str(i_), q_hist)

    del ff,ff_err,allstates,q_hist
   
    return i_

# =============================================================================
#  AUXILIARY FUNCTIONS
# =============================================================================

def linear(x,m,b):
    return m*x+b
    
def par_to_d_str(par):
    par_str="{:.4e}".format(par)
    return f"{str(par_str).replace('.','d')}"



def border_check(DT,DH,DS,units_len,tini,tfin,k,nsteps,transient,save_each):
    
    n=DH.shape[0]
    state=np.array([random.randint(0, 1) for _ in range(n)])
    folded_res=np.zeros(4)
    run_=True
    while run_:
        for t,temp in enumerate([tini,tini,tfin,tfin]):
            state,folded_res[t],frer,stav,q_hist=montecarlo_speed_complete(DH,
                                                 temp,DS,units_len,k,nsteps,transient,save_each,state)

              #  st,folded_res[t],frer,stav,tr0,tr1=isf.montecarlo_ss_states_wham_speed2(DH,temp,DS,units_len,k,
                                                                               # nsteps,transient,save_each,state)
        if folded_res[0]<1:
            tini+=-DT
        if folded_res[1]>0:
            tfin+=DT
        if (folded_res==[1,1,0,0]).all():    
            run_=False
        if tini==0:
            tini=10
            if (folded_res[2:]==[0,0]).all():
                run_=False
        if tfin>=600:
            if (folded_res[:1]==[1,1]).all():
                run_=False
        
    del state,folded_res,frer,stav,q_hist
    return tini-5,tfin+5 #le agregue un margen mas porque a veces no sabe ajustar la curva tan cerca

def make_MRA(plain_seq,len_rep,AAdict):

    seq_len=len(list(plain_seq))
    if (seq_len%len_rep)==0:
        nrep=int(seq_len/len_rep)
        print(nrep, 'repeat sequence')
    else:
        print('Error: wrong length')
        return 0

    MSA=np.zeros((1,nrep*len_rep))
    #for i in range(len(aln)):
    MSA[0,:]=[int(AAdict[x]) for x in list(plain_seq)]
    MRA=pd.DataFrame(MSA.reshape((nrep,33)),dtype=int)
    MSA_=np.zeros((1,nrep*len_rep),dtype='object')
    MSA_[0,:]=list(plain_seq)
    MRA_=pd.DataFrame(MSA_.reshape((nrep,33)))

    return MRA,MRA_

def map_to_MRA(i,t_map,df_repeat_info,df_MRA):
    rep_info=df_repeat_info.loc[df_repeat_info['short.name']==t_map['array.name'][i]]
    pini_index=rep_info.index[rep_info.pini==t_map['array.p.ini'][i]][0]
    pfin_index=rep_info.index[rep_info.pfin==t_map['array.p.fin'][i]][0]
    prot_name=t_map['array.name'][i]+'_'+str(t_map['array.p.ini'][i])+'_'+str(t_map['array.p.fin'][i])
    MRA=df_MRA.loc[pini_index:pfin_index]
    return MRA,prot_name



# para el doble heatmap
def MRA_to_evo_and_DH(MRA,Jij,Hi,AAdict,replen,rep_frag_len,j,m,gaps_out):

    evo_energy_full=energy_eval(Jij,Hi,MRA.apply(lambda x: x.map(AAdict)))
    nrep=int(len(evo_energy_full)/replen)
    real_len=replen*nrep  # [res]
    
    if gaps_out:
        gap_index=np.where(MRA.values=='-')[0]*replen+np.where(MRA.values=='-')[1]

        if len(gap_index)>0:
            evo_energy_full.loc[gap_index]=0
            evo_energy_full[gap_index]=0

    
    breaks=np.cumsum(np.array([0]+rep_frag_len[j]*nrep)[:-1])
    evo_energy=energy_submatrix(evo_energy_full,breaks)
      
    DH=evo_energy/m
        
    return evo_energy_full,evo_energy,DH,nrep,breaks

def calculate_evo_features(i,t_map,df_repeat_info,df_MRA,Jij,Hi,AAdict,replen,rep_frag_len,j,m,prot_dir,gaps_out,
                           ank_full_set=False):
    
    if t_map.custom_msa[i]:
            #read custom msa
            prot_name=t_map.prot_name[i]
            MRA=pd.read_csv(prot_dir+'msa/'+prot_name+'.csv',index_col=0)

    else:
        if ank_full_set:
            MRA,prot_name=map_to_MRA(i,t_map,df_repeat_info,df_MRA)
        else:
            MRA=df_MRA.loc[(t_map['abs.init.row'][i]-1):(t_map['abs.fin.row'][i]-1)]
            prot_name=t_map['array.name'][i]


    # 3) correr MRA to ising y si0 to DS
    evo_energy_full,evo_energy,DH,nrep,breaks=MRA_to_evo_and_DH(MRA,Jij,Hi,AAdict,replen,rep_frag_len,j,m,gaps_out)

   # np.savetxt(DH_file,DH.values)

    si0=0.01 # esto no importa, solo para no cambiar la función
    DS,units_len=si0_to_DS_units_len(si0,MRA,nrep,breaks,rep_frag_len,j,gaps_out)

    #np.savetxt(ulf_file,units_len,fmt='%i')
    return evo_energy_full,evo_energy,DH,DS,units_len,breaks


# MRA -> Ising energy
def MRA_to_ising(MRA,Jij,Hi,AAdict,replen,rep_frag_len,j,m,gaps_out):

    evo_energy_full=energy_eval(Jij,Hi,MRA.apply(lambda x: x.map(AAdict)))
    nrep=int(len(evo_energy_full)/replen)
    real_len=replen*nrep  # [res]
    
    if gaps_out:
        gap_index=np.where(MRA.values=='-')[0]*replen+np.where(MRA.values=='-')[1]

        if len(gap_index)>0:
            evo_energy_full.loc[gap_index]=0
            evo_energy_full[gap_index]=0

    
    breaks=np.cumsum(np.array([0]+rep_frag_len[j]*nrep)[:-1])
    evo_energy=energy_submatrix(evo_energy_full,breaks)
      
    DH=evo_energy/m
        
    return DH,nrep,breaks

def si0_to_DS_units_len(si0,MRA,nrep,breaks,rep_frag_len,j,gaps_out,replen=33):
    
    units_len=np.array(rep_frag_len[j]*nrep)
    
    S_2=pd.DataFrame(index=range(replen),columns=range(replen),data=0.0)
    S_1=pd.DataFrame(index=range(replen),columns=range(replen),data=0.0)
    S_1=S_1.where(np.triu(np.ones(S_1.shape)).astype(bool),0)
    np.fill_diagonal(S_1.values,si0)
    
    # los gaps no suman entropía
    DS_all=expand_average_matrix(S_1,S_2,Nrep=nrep)
    if gaps_out:
        gap_index=np.where(MRA.values=='-')[0]*replen+np.where(MRA.values=='-')[1]

        if len(gap_index)>0:
            DS_all.loc[gap_index]=0
            DS_all[gap_index]=0
            
            
            gap_units=np.array([np.argmax(breaks[g>=breaks]) for g in gap_index])
            values, counts = np.unique(gap_units, axis=0, return_counts=True)
            units_len[values]=units_len[values]-counts   
    DS=energy_submatrix(DS_all,breaks=breaks) 
    return DS,units_len


# Experimental data 
def reduce_exp_data(prot_dir,folder,data_file,tsteps):
    exp_data=pd.read_csv(prot_dir+'papers_data/'+folder+'/'+data_file+'.csv',header=None)
    exp_data.columns=['temp','ff']
    rangoT=exp_data.temp.min(),exp_data.temp.max()
    Teqs=np.linspace(rangoT[0],rangoT[1],tsteps)
    ix=[]
    for x in Teqs:
        ix.append(exp_data[exp_data.temp>=x].index[0])
    exp_data_n=exp_data.iloc[ix]
    if len(exp_data)<=tsteps:
        exp_data_n=exp_data
    return exp_data_n


# =============================================================================
#  EVOLUTIONARY ENERGY FUNCTIONS
# =============================================================================

def energy_eval_sum(Jij,Hi,nseq):
    Nrep=len(nseq)
    rep_len=len(nseq.columns)
    E=0
    I=0
    for N in np.arange(Nrep):
        seq1=nseq.values[N]
        pos_1rep=seq1+np.arange(rep_len)*21
        E+=-Jij.iloc[pos_1rep,pos_1rep].sum().sum()/2 -Hi.values[range(rep_len),seq1].sum()

        if N!=Nrep-1:
            seq2=nseq.values[N+1]
            pos_2rep=seq2+np.arange(rep_len,2*rep_len)*21
            I+=-Jij.iloc[pos_1rep,pos_2rep].sum().sum()

    return E,I

def eval_hi(Hi,sequence):
    i=0
    Hi_ev=[]
    for ai in sequence:
        Hi_ev.append(-Hi.values[i,ai])
        i=i+1
    return Hi_ev

def energy_eval(Jij,Hi,nseq):
    Energy=pd.DataFrame()
    Nrep=len(nseq)
    rep_len=len(nseq.columns)
    reps=np.arange(Nrep)
    for N in reps:
        seq1=nseq.values[N]
        pos_1rep=seq1+np.arange(rep_len)*21
        E_1=-Jij.iloc[pos_1rep,pos_1rep]
        E_1.index=np.arange(rep_len)
        E_1.columns=np.arange(rep_len)
        E_1=E_1.where(np.triu(np.ones(E_1.shape)).astype(bool),0)
        np.fill_diagonal(E_1.values,eval_hi(Hi,seq1))

        if N!=Nrep-1:
            seq2=nseq.values[N+1]
            pos_2rep=seq2+np.arange(rep_len,2*rep_len)*21
            E_2=-Jij.iloc[pos_1rep,pos_2rep]
            E_2.index=np.arange(rep_len)
            E_2.columns=np.arange(rep_len)
            E_after = pd.DataFrame(np.zeros((rep_len, (Nrep-N-2)*rep_len)))
            E=pd.concat([E_1,E_2,E_after],axis=1, ignore_index=True)
        else:
            E=E_1

        E_before = pd.DataFrame(np.zeros((rep_len, N*rep_len)))
        E_row=pd.concat([E_before,E],axis=1, ignore_index=True)
        Energy=pd.concat([Energy,E_row], ignore_index=True)
    return Energy

def expand_average_matrix(E1_av,E2_av,Nrep):
    Energy=pd.DataFrame()
    rep_len=len(E1_av)
    for N in np.arange(Nrep):
        if N!=Nrep-1:
            E_after = pd.DataFrame(np.zeros((rep_len, (Nrep-N-2)*rep_len)))
            E=pd.concat([E1_av,E2_av,E_after],axis=1, ignore_index=True)
        else:
            E=E1_av

        E_before = pd.DataFrame(np.zeros((rep_len, N*rep_len)))
        E_row=pd.concat([E_before,E],axis=1, ignore_index=True)
        Energy=pd.concat([Energy,E_row], ignore_index=True)
    return Energy


# REDUCE EVALUATED ENERGY MATRIX ACCORDING TO CUSTOM BREAKS
def energy_submatrix(evo_energy_full,breaks):
    evo_energy_s=pd.DataFrame(index=range(len(breaks)),columns=range(len(breaks)),data=0)
    for n in range(len(breaks)):
        if n==len(breaks)-1:
            pos_n=range(breaks[n],len(evo_energy_full))
        else:
            pos_n=range(breaks[n],breaks[(n+1)])
        for m in range(n,len(breaks)):

            if m==len(breaks)-1:
                pos_m=range(breaks[m],len(evo_energy_full))
            else:
                pos_m=range(breaks[m],breaks[(m+1)])
            
            evo_energy_s.loc[n,m]=evo_energy_full.loc[pos_n,pos_m].values.sum()
    return evo_energy_s


def energy_average_matrix(evo_energy_full,breaks):

    evo_energy_s=pd.DataFrame(np.zeros((len(evo_energy_full),len(evo_energy_full))))
    evo_energy_av=pd.DataFrame(np.zeros((len(evo_energy_full),len(evo_energy_full))))
    evo_energy_s[:]=np.nan
    evo_energy_av[:]=np.nan

    for n in range(len(breaks)):
        if n==len(breaks)-1:
            pos_n=range(breaks[n],len(evo_energy_full))
        else:
            pos_n=range(breaks[n],breaks[(n+1)])
        for m in range(n,len(breaks)):

            if m==len(breaks)-1:
                pos_m=range(breaks[m],len(evo_energy_full))
            else:
                pos_m=range(breaks[m],breaks[(m+1)])

            evo_energy_s.loc[pos_n,pos_m]=evo_energy_full.loc[pos_n,pos_m].values.sum()
            evo_energy_av.loc[pos_n,pos_m]=evo_energy_full.loc[pos_n,pos_m].values.sum()/len(pos_n)/len(pos_m)

            if n==m:

                evo_energy_av.loc[pos_n,pos_m]=evo_energy_av.loc[pos_n,pos_m]*2
    return evo_energy_s,evo_energy_av
