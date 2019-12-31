# taken from https://github.com/jjgoings/McMurchie-Davidson/

import numpy as np
from scipy.special import factorial2 as fact2
from scipy.special import hyp1f1
from math import sqrt

def E(i,j,t,Qx,a,b):
    ''' Recursive definition of Hermite Gaussian coefficients.
        Returns a float.
        a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
        i,j: orbital angular momentum number on Gaussian 'a' and 'b'
        t: number nodes in Hermite (depends on type of integral, 
           e.g. always zero for overlap integrals)
        Qx: distance between origins of Gaussian 'a' and 'b'
    '''
    p = a + b
    q = a*b/p
    if (t < 0) or (t > (i + j)):
        # out of bounds for t  
        return 0.0
    elif i == j == t == 0:
        # base case
        return np.exp(-q*Qx*Qx) # K_AB
    elif j == 0:
        # decrement index i
        return (1/(2*p))*E(i-1,j,t-1,Qx,a,b) - \
               (q*Qx/a)*E(i-1,j,t,Qx,a,b)    + \
               (t+1)*E(i-1,j,t+1,Qx,a,b)
    else:
        # decrement index j
        return (1/(2*p))*E(i,j-1,t-1,Qx,a,b) + \
               (q*Qx/b)*E(i,j-1,t,Qx,a,b)    + \
               (t+1)*E(i,j-1,t+1,Qx,a,b)

def overlap(a,lmn1,A,b,lmn2,B):
    ''' Evaluates overlap integral between two Gaussians
        Returns a float.
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
              for Gaussian 'a'
        lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
        A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
        B:    list containing origin of Gaussian 'b'
    '''
    l1,m1,n1 = lmn1 # shell angular momentum on Gaussian 'a'
    l2,m2,n2 = lmn2 # shell angular momentum on Gaussian 'b'
    S1 = E(l1,l2,0,A[0]-B[0],a,b) # X
    S2 = E(m1,m2,0,A[1]-B[1],a,b) # Y
    S3 = E(n1,n2,0,A[2]-B[2],a,b) # Z
    return S1*S2*S3*np.power(np.pi/(a+b),1.5) 

def S(a,b):
    '''Evaluates overlap between two contracted Gaussians
       Returns float.
       Arguments:
       a: contracted Gaussian 'a', BasisFunction object
       b: contracted Gaussian 'b', BasisFunction object
    '''
    s = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            s += a.norm[ia]*b.norm[ib]*ca*cb*\
                     overlap(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin)
    return s

class BasisFunction(object):
    ''' A class that contains all our basis function data
        Attributes:
        origin: array/list containing the coordinates of the Gaussian origin
        shell:  tuple of angular momentum
        exps:   list of primitive Gaussian exponents
        coefs:  list of primitive Gaussian coefficients
        norm:   list of normalization factors for Gaussian primitives
    '''
    def __init__(self,origin=[0.0,0.0,0.0],shell=(0,0,0),num_exps=None,exps=[],coefs=[]):
        self.origin = np.asarray(origin)
        self.shell = shell
        self.exps  = exps
        self.coefs = coefs
        self.num_exps = len(self.exps)
        self.norm = None
        self.normalize()

    def normalize(self):
        ''' Routine to normalize the basis functions, in case they
            do not integrate to unity.
        '''
        l,m,n = self.shell
        L = l+m+n
        # self.norm is a list of length equal to number primitives
        # normalize primitives first (PGBFs)
        self.norm = np.sqrt(np.power(2,2*(l+m+n)+1.5)*
                        np.power(self.exps,l+m+n+1.5)/
                        fact2(2*l-1)/fact2(2*m-1)/
                        fact2(2*n-1)/np.power(np.pi,1.5))

        # now normalize the contracted basis functions (CGBFs)
        # Eq. 1.44 of Valeev integral whitepaper
        prefactor = np.power(np.pi,1.5)*\
            fact2(2*l - 1)*fact2(2*m - 1)*fact2(2*n - 1)/np.power(2.0,L)

        N = 0.0
        num_exps = len(self.exps)
        for ia in range(num_exps):
            for ib in range(num_exps):
                N += self.norm[ia]*self.norm[ib]*self.coefs[ia]*self.coefs[ib]/\
                         np.power(self.exps[ia] + self.exps[ib],L+1.5)

        N *= prefactor
        N = np.power(N,-0.5)
        for ia in range(num_exps):
            pass # self.coefs[ia] *= N

def kinetic(a,lmn1,A,b,lmn2,B):
    ''' Evaluates kinetic energy integral between two Gaussians
        Returns a float.
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
              for Gaussian 'a'
        lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
        A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
        B:    list containing origin of Gaussian 'b'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    term0 = b*(2*(l2+m2+n2)+3)*\
                            overlap(a,(l1,m1,n1),A,b,(l2,m2,n2),B)
    term1 = -2*np.power(b,2)*\
                           (overlap(a,(l1,m1,n1),A,b,(l2+2,m2,n2),B) +
                            overlap(a,(l1,m1,n1),A,b,(l2,m2+2,n2),B) +
                            overlap(a,(l1,m1,n1),A,b,(l2,m2,n2+2),B))
    term2 = -0.5*(l2*(l2-1)*overlap(a,(l1,m1,n1),A,b,(l2-2,m2,n2),B) +
                  m2*(m2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2-2,n2),B) +
                  n2*(n2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2,n2-2),B))
    return term0+term1+term2

def T(a,b):
    '''Evaluates kinetic energy between two contracted Gaussians
       Returns float.
       Arguments:
       a: contracted Gaussian 'a', BasisFunction object
       b: contracted Gaussian 'b', BasisFunction object
    '''
    t = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            t += a.norm[ia]*b.norm[ib]*ca*cb*\
                     kinetic(a.exps[ia],a.shell,a.origin,\
                     b.exps[ib],b.shell,b.origin)
    return t

def R(t,u,v,n,p,PCx,PCy,PCz,RPC):
    ''' Returns the Coulomb auxiliary Hermite integrals 
        Returns a float.
        Arguments:
        t,u,v:   order of Coulomb Hermite derivative in x,y,z
                 (see defs in Helgaker and Taylor)
        n:       order of Boys function 
        PCx,y,z: Cartesian vector distance between Gaussian 
                 composite center P and nuclear center C
        RPC:     Distance between P and C
    '''

    val = 0.0
    if t == u == v == 0:
        T = p*RPC*RPC
        val += np.power(-2*p,n)*boys(n,T)
    elif t == u == 0:
        if v > 1:
            val += (v-1)*R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)
        val += PCz*R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
    elif t == 0:
        if u > 1:
            val += (u-1)*R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCy*R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
    else:
        if t > 1:
            val += (t-1)*R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCx*R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    return val

def boys(n,T):
    return hyp1f1(n+0.5,n+1.5,-T)/(2.0*n+1.0) 

def gaussian_product_center(a,A,b,B):
    return (a*A+b*B)/(a+b)

def nuclear_attraction(a,lmn1,A,b,lmn2,B,C):
    ''' Evaluates kinetic energy integral between two Gaussians
         Returns a float.
         a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
         b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
         lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
               for Gaussian 'a'
         lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
         A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
         B:    list containing origin of Gaussian 'b'
         C:    list containing origin of nuclear center 'C'
    '''
    l1,m1,n1 = lmn1 
    l2,m2,n2 = lmn2
    p = a + b
    P = gaussian_product_center(a,A,b,B) # Gaussian composite center
    RPC = np.linalg.norm(P-C)

    val = 0.0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                val += E(l1,l2,t,A[0]-B[0],a,b) * \
                       E(m1,m2,u,A[1]-B[1],a,b) * \
                       E(n1,n2,v,A[2]-B[2],a,b) * \
                       R(t,u,v,0,p,P[0]-C[0],P[1]-C[1],P[2]-C[2],RPC)
    val *= 2*np.pi/p 
    return val

def V(a,b,C):
    '''Evaluates overlap between two contracted Gaussians
       Returns float.
       Arguments:
       a: contracted Gaussian 'a', BasisFunction object
       b: contracted Gaussian 'b', BasisFunction object
       C: center of nucleus
    '''
    v = 0.0
    for ia, ca in enumerate(a.coefs):
        for ib, cb in enumerate(b.coefs):
            v += a.norm[ia]*b.norm[ib]*ca*cb*\
                     nuclear_attraction(a.exps[ia],a.shell,a.origin,
                     b.exps[ib],b.shell,b.origin,C)
    return v

def electron_repulsion(a,lmn1,A,b,lmn2,B,c,lmn3,C,d,lmn4,D):
    ''' Evaluates kinetic energy integral between two Gaussians
        Returns a float.
        a,b,c,d:   orbital exponent on Gaussian 'a','b','c','d'
        lmn1,lmn2
        lmn3,lmn4: int tuple containing orbital angular momentum
                   for Gaussian 'a','b','c','d', respectively
        A,B,C,D:   list containing origin of Gaussian 'a','b','c','d'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    l3,m3,n3 = lmn3
    l4,m4,n4 = lmn4
    p = a+b # composite exponent for P (from Gaussians 'a' and 'b')
    q = c+d # composite exponent for Q (from Gaussians 'c' and 'd')
    alpha = p*q/(p+q)
    P = gaussian_product_center(a,A,b,B) # A and B composite center
    Q = gaussian_product_center(c,C,d,D) # C and D composite center

    #print (alpha, p, q, P, Q)
    
    RPQ = np.linalg.norm(P-Q)

    val = 0.0
    for t in range(l1+l2+1):
        for u in range(m1+m2+1):
            for v in range(n1+n2+1):
                for tau in range(l3+l4+1):
                    for nu in range(m3+m4+1):
                        for phi in range(n3+n4+1):
                            val += E(l1,l2,t,A[0]-B[0],a,b) * \
                                   E(m1,m2,u,A[1]-B[1],a,b) * \
                                   E(n1,n2,v,A[2]-B[2],a,b) * \
                                   E(l3,l4,tau,C[0]-D[0],c,d) * \
                                   E(m3,m4,nu ,C[1]-D[1],c,d) * \
                                   E(n3,n4,phi,C[2]-D[2],c,d) * \
                                   np.power(-1,tau+nu+phi) * \
                                   R(t+tau,u+nu,v+phi,0,\
                                       alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ)

    val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q))
    return val

def ERI(a,b,c,d):
    '''Evaluates overlap between two contracted Gaussians
        Returns float.
        Arguments:
        a: contracted Gaussian 'a', BasisFunction object
        b: contracted Gaussian 'b', BasisFunction object
        c: contracted Gaussian 'b', BasisFunction object
        d: contracted Gaussian 'b', BasisFunction object
    '''
    eri = 0.0
    for ja, ca in enumerate(a.coefs):
        for jb, cb in enumerate(b.coefs):
            for jc, cc in enumerate(c.coefs):
                for jd, cd in enumerate(d.coefs):
                    er = electron_repulsion(a.exps[ja],a.shell,a.origin,\
                                            b.exps[jb],b.shell,b.origin,\
                                            c.exps[jc],c.shell,c.origin,\
                                            d.exps[jd],d.shell,d.origin)
                    #print (er)
                    eri += a.norm[ja]*b.norm[jb]*c.norm[jc]*d.norm[jd]*\
                             ca*cb*cc*cd*\
                             er
    return eri

def xR(xyz,n,PQ,F):
    for i in [ 0, 1, 2 ]:
        if not xyz[i]: continue
        r1 = list(xyz)
        r1[i] -= 1
        val = PQ[i]*xR(r1,n+1,PQ,F)
        if xyz[i] > 1:
            r2 = list(xyz)
            r2[i] -= 2
            val += (xyz[i]-1)*xR(r2,n+1,PQ,F)
        return val
    return F(n)

def x_electron_repulsion(p,lmn1,P,q,lmn2,Q):
    ''' Evaluates kinetic energy integral between two Gaussians
        Returns a float.
        a,b,c,d:   orbital exponent on Gaussian 'a','b','c','d'
        lmn1,lmn2
        lmn3,lmn4: int tuple containing orbital angular momentum
                   for Gaussian 'a','b','c','d', respectively
        A,B,C,D:   list containing origin of Gaussian 'a','b','c','d'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    alpha = p*q/(p+q)

    #print (alpha, p, q, P, Q)
    
    RPQ = np.linalg.norm(P-Q)

    def F(n):
        return np.power(-2*alpha,n)*boys(n,alpha*RPQ**2)

    val = 0.0
    for t in range(l1+1):
        for u in range(m1+1):
            for v in range(n1+1):
                for tau in range(l2+1):
                    for nu in range(m2+1):
                        for phi in range(n2+1):
                            val += E(l1,0,t,P[0],p,0) * \
                                   E(m1,0,u,P[1],p,0) * \
                                   E(n1,0,v,P[2],p,0) * \
                                   E(l2,0,tau,Q[0],q,0) * \
                                   E(m2,0,nu ,Q[1],q,0) * \
                                   E(n2,0,phi,Q[2],q,0) * \
                                   np.power(-1,tau+nu+phi) * \
                                   xR([t+tau,u+nu,v+phi],0,\
                                      [P[0]-Q[0],P[1]-Q[1],P[2]-Q[2]],
                                      F)

    val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q))
    return val


def h_electron_repulsion(p,lmn1,P,q,lmn2,Q):
    ''' Evaluates kinetic energy integral between two Gaussians
        Returns a float.
        a,b,c,d:   orbital exponent on Gaussian 'a','b','c','d'
        lmn1,lmn2
        lmn3,lmn4: int tuple containing orbital angular momentum
                   for Gaussian 'a','b','c','d', respectively
        A,B,C,D:   list containing origin of Gaussian 'a','b','c','d'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    alpha = p*q/(p+q)

    #print (alpha, p, q, P, Q)
    
    RPQ = np.linalg.norm(P-Q)

    def On(n):
        assert n >= (l1+l2+m1+m2+n1+n2+1)/2
        return np.power(-2*alpha,n)*boys(n,alpha*RPQ**2)
    
    val = 0.0
    val = xR([l1+l2,m1+m2,n1+n2],0,\
             [P[0]-Q[0],P[1]-Q[1],P[2]-Q[2]],
             On
          )

    val *= np.power(-2*q,-(l2+m2+n2))
    val *= np.power(+2*p,-(l1+m1+n1))
    val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q))

    return val


def xERI2(p,q):
    r0 = np.asarray([0,0,0])
    eri = 0.0
    for jp, cp in enumerate(p.coefs):
        for jq, cq in enumerate(q.coefs):
            v = x_electron_repulsion(
                p.exps[jp],p.shell,p.origin,
                q.exps[jq],q.shell,q.origin
            )
            #print (v)
            eri += cp*cq*v#*p.norm[jp]*q.norm[jq]
    return eri

def hERI2(p,q):
    r0 = np.asarray([0,0,0])
    eri = 0.0
    for jp, cp in enumerate(p.coefs):
        for jq, cq in enumerate(q.coefs):
            v = h_electron_repulsion(
                p.exps[jp],p.shell,p.origin,
                q.exps[jq],q.shell,q.origin
            )
            #print (v)
            eri += cp*cq*v#*p.norm[jp]*q.norm[jq]
    return eri

def orbitals(L):
    s = set()
    for i in range(L+1):
        for j in range(L+1):
            for k in range(L+1):
                if i+j+k == L:
                    s.update([(i,j,k)])
    return sorted(s, reverse=True)
                
def cart_to_spher_transform(L):
    t = np.zeros([2*L+1, int((L+1)*(L+2)/2)])
    if L == 0:
        t[0,0] = 1
    if L == 1:
        t[0,2] = 1
        t[1,0] = 1
        t[2,1] = 1
    if L == 2:
        t[0] = [ -1/2., 0, 0, -1/2., 0, 1 ]
        t[1] = [ 0, 0, 1, 0, 0, 0 ]
        t[2] = [ 0, 0, 0, 0, 1, 0 ]
        t[3] = [ 1/2.*sqrt(3), 0, 0, -1/2.*sqrt(3), 0, 0 ]
        t[4] = [ 0, 1, 0, 0, 0, 0 ]
    return t

class Shell:
    def __init__(self, L, prims, r = [0,0,0]):
        self.L = L
        self.orbitals = orbitals(L)
        self.prims = prims
        self.r = np.array(r)


def R_transform(orbitals,PQ):
    T = np.zeros([len(orbitals), sum(orbitals[0])+1])
    def R_transform_recursive(T,L,n=0,t=1):
        for i in [ 0, 1, 2 ]:
            if not L[i]: continue
            r1 = list(L)
            r1[i] -= 1
            R_transform_recursive(T,r1,n+1,t*PQ[i])
            if L[i] > 1:
                r2 = list(L)
                r2[i] -= 2
                R_transform_recursive(T,r2,n+1,t*(L[i]-1))
            return
        T[n] += t
    for k,L in enumerate(orbitals):
        R_transform_recursive(T[k],L)
    return T

def On(p,Lp,P,q,Lq,Q):

    P = np.array(P)
    Q = np.array(Q)
    
    L = Lp+Lq
    alpha = p*q/(p+q)
    PQ = np.linalg.norm(P-Q)

    O = [ 0 ]*(L+1)
    #print O
    
    for n in range(L/2, L+1):
        #assert n >= (l1+l2+m1+m2+n1+n2+1)/2
        O[n] = np.power(-2*alpha,n)*boys(n,alpha*PQ**2)
    
    val = 1
    val *= np.power(  -1,-Lq)
    val *= np.power(+2*q,-Lq)
    val *= np.power(+2*p,-Lp)
    val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q))

    return ( val, O )

def mmd2(P,Q):
    eri = 0.0
    O = np.zeros([P.L+Q.L+1])
    for (p,cp) in P.prims:
        for (q,cq) in Q.prims:
            (scale,Opq) = On(p, P.L, P.r, q, Q.L, Q.r)
            #O += cp*cq*scale*np.array(Opq)
            O += cp*cq*scale*np.array(Opq)
    PQ1 = np.dot(
        R_transform(orbitals(P.L+Q.L), P.r-Q.r),
        O
    )
    #print O
    
    index = dict()
    for (i,f) in enumerate(orbitals(P.L+Q.L)):
        index[f] = i

    PQ = np.ndarray([len(P.orbitals),len(Q.orbitals)])
        
    for i,p in enumerate(P.orbitals):
        for j,q in enumerate(Q.orbitals):
            ij = index[tuple(map(sum, zip(p,q)))]
            PQ[i,j] = PQ1[ij]

    print "ERI(h) = ",PQ
            
    PQ = np.dot(cart_to_spher_transform(P.L),PQ)
    PQ = np.dot(PQ, cart_to_spher_transform(Q.L).T)

    return PQ
