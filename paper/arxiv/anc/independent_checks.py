"""Finite checks of analytic identities and scope-sensitive certificates."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import json
import math
from fractions import Fraction as R
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info

HERE=Path(__file__).resolve().parent
rng=np.random.default_rng(20261029)
assertions=0
def check(condition):
    global assertions
    assert condition
    assertions+=1

def path(n):
    a=np.zeros((n,n))
    for i in range(n-1):
        a[i,i]+=1;a[i+1,i+1]+=1
        a[i,i+1]-=1;a[i+1,i]-=1
    return a

def grid(T,k):
    n=T+1
    a=np.zeros((n**k,n**k))
    for j in range(k):
        a+=np.kron(np.kron(np.eye(n**j),path(n)),np.eye(n**(k-j-1)))
    return a

def green(T,k):
    n=T+1
    mu=4*np.sin(np.pi*np.arange(n)/(2*n))**2
    w=2/n*np.cos(np.pi*np.arange(n)/(2*n))**2
    w[0]=1/n
    val=0.
    for index in np.ndindex(*([n]*k)):
        if any(index):val+=np.prod(w[list(index)])/np.sum(mu[list(index)])
    return val

def entropy(p):
    p=np.asarray(p);p=p[p>1e-15]
    return float(-np.sum(p*np.log2(p)))

r=R(1,16)
for c in [R(0),R(1,100),R(1,20),R(1),R(100)]:
    p0=math.ceil(4*c/r)+2
    def e(h):return c*R(1 if h%2 else 0,h**3)
    for p in range(p0,p0+125):
        g=R(1,p+11)
        fp=-g/p+(r+e(p-1))/p**2
        fq=-g/(p+1)+(r+e(p))/(p+1)**2
        tie=r*(R(1,p)+R(1,p+1))+R(p+1,p)*e(p-1)-R(p,p+1)*e(p)
        nexttie=r*(R(1,p+1)+R(1,p+2))+R(p+2,p+1)*e(p)-R(p+1,p+2)*e(p+1)
        check(fp-fq==(tie-g)/(p*(p+1)))
        check(tie>nexttie)

for k in [1,2,3]:
    for T in [1,2,3,4]:
        lap=grid(T,k); pinv=np.linalg.pinv(lap,hermitian=True)
        check(abs(green(T,k)-pinv[0,0])<1e-10)
        if k==1:
            check(abs(green(T,1)-T*(2*T+1)/(6*(T+1)))<1e-12)
        if k==3:check(green(T,3)<14)
        if k==2:
            n=T+1
            lp=np.linalg.pinv(path(n),hermitian=True)
            for j in range(n):
                flow=(sum(i*i for i in range(1,j+1))+sum(i*i for i in range(1,n-j)))/n**2
                check(abs(lp[j,j]-flow)<1e-10)
                w=np.full(n*n,-1/(n*n));w[j*n:(j+1)*n]+=1/n
                check(abs(w@pinv@w-lp[j,j]/n)<1e-10)
                check(w@pinv@w<=1/3+1e-12)

max_ratio=0.
minimum_slack=math.inf
for k in [1,2,3]:
    for T in [1,2,3]:
        n=T+1;M=n**k;lap=grid(T,k);G=green(T,k)
        mu1=4*math.sin(math.pi/(2*n))**2
        constant=2*(3+4*math.sqrt(G))**2+2/(M*mu1)
        P=np.diag([1.,1.,0.,0.])
        for rank in [1,2,3]:
            raw=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
            u,_=np.linalg.qr(raw);Q=u[:,:rank]@u[:,:rank].conj().T
            F=np.kron(np.eye(2),Q[:2,:2])
            H=np.kron(lap,np.eye(4)).astype(complex)
            H[:4,:4]+=np.eye(4)-P;H[-4:,-4:]+=Q
            op=np.kron(np.eye(M),F)
            slack=np.linalg.eigvalsh(constant*M*H-op)[0]
            minimum_slack=min(minimum_slack,float(slack))
            check(slack>-1e-8)
            ev,vec=np.linalg.eigh(H)
            for j in range(min(4,len(ev))):
                psi=vec[:,j];lam=float(np.vdot(psi,H@psi).real)
                loss=float(np.vdot(psi,op@psi).real)
                check(loss<=constant*M*lam+1e-9)
                if lam>1e-10:max_ratio=max(max_ratio,loss/(M*lam))
                blocks=psi.reshape(M,4);mean=np.mean(blocks,axis=0)
                deviation=blocks-mean
                check(np.linalg.norm(deviation)**2<=lam/mu1+1e-9)
                if k>=2:
                    z=np.tile(P@mean,(M,1)).reshape(-1)
                    b=(1.5+math.sqrt(G))*math.sqrt(max(M*lam,0))
                    check(np.linalg.norm(psi-z)<=b+1e-9)

adversary=[]
for T in [2,4,8,16]:
    lap=grid(T,2);M=len(lap);pinv=np.linalg.pinv(lap,hermitian=True)
    G=pinv[-1,-1];f=np.ones(M)-pinv[:,-1]/G
    energy=float(f@lap@f/(f@f));ratio=1/(M*energy)
    check(abs(f[-1])<1e-10)
    check(ratio>=G-1e-9)
    adversary.append(dict(T=T,resistance=float(G),ratio=float(ratio)))

for d in [2,4,8]:
    phi=np.eye(d).reshape(-1)/math.sqrt(d)
    H0=np.eye(d*d)-np.outer(phi,phi)
    for eta in [.01,.1,.249]:
        raw=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d))
        v=raw+raw.conj().T;v*=eta/np.linalg.norm(v,2)
        H=H0+np.kron(v,np.eye(d))+np.kron(np.eye(d),v.T)
        ev,vec=np.linalg.eigh(H);X=vec[:,0].reshape(d,d)
        inverse=np.linalg.inv((1-ev[0])*np.eye(d)+2*v)
        test=inverse.reshape(-1);test/=np.linalg.norm(test)
        check(abs(np.vdot(test,vec[:,0]))>1-1e-10)
        check(np.linalg.svd(X,compute_uv=False)[-1]>1e-8)
        check(ev[1]-ev[0]>=1-4*eta-1e-10)
    eta=.9/(4+2*math.sqrt(d))
    for trial in range(5):
        fields=[]
        for j in range(2):
            raw=rng.normal(size=(d,d))+1j*rng.normal(size=(d,d))
            v=raw+raw.conj().T;v*=eta/np.linalg.norm(v,2);fields.append(v)
        H=H0+np.kron(fields[0],np.eye(d))+np.kron(np.eye(d),fields[1])
        ev,vec=np.linalg.eigh(H)
        check(np.linalg.svd(vec[:,0].reshape(d,d),compute_uv=False)[-1]>1e-9)
    for beta in [.1,math.log(d+1),2*math.log(d+1)]:
        Z=math.exp(beta)+d*d-1
        rho=(np.eye(d*d)+(math.exp(beta)-1)*np.outer(phi,phi))/Z
        F=math.exp(beta)/Z
        mi=2*math.log2(d)-entropy(np.linalg.eigvalsh(rho))
        formula=2*math.log2(d)+F*math.log2(F)+(1-F)*math.log2((1-F)/(d*d-1))
        check(abs(mi-formula)<1e-10)
        check(mi>0)
        pt=rho.reshape(d,d,d,d).transpose(0,3,2,1).reshape(d*d,d*d)
        check(abs(np.linalg.eigvalsh(pt)[0]-(1-(math.exp(beta)-1)/d)/Z)<1e-10)
        check((np.linalg.eigvalsh(pt)[0]>=-1e-10)==(beta<=math.log(d+1)+1e-10))

N=100;LA=10000;a=2.**-40;p=3
depths=[j for j in range(2,N+1,2) if 2*16.**-j<=a*LA**-p]
check(min(depths)==22)
check(max(depths)==100)
for pool in threadpool_info():check(pool['num_threads']==1)
pool_records=[{k:p[k] for k in ('internal_api','num_threads','version') if k in p}
              for p in threadpool_info()]
record=dict(status='PASS',assertions=assertions,max_extraction_ratio=max_ratio,
            minimum_operator_slack=minimum_slack,green_adversary=adversary,
            depth_example=min(depths),threadpools=pool_records)
(HERE/'independent_checks.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:v for k,v in record.items() if k!='threadpools'},indent=2))
print('ALL OK')
