from integrals import xERI2, BasisFunction
from integrals import orbitals, cart_to_spher_transform, hERI2, On, R_transform, mmd2, Shell
import numpy as np

P = np.array([ 3.1, 6, 1 ])
Q = np.array([ 1.3, -2, 2 ])

L=2
nbf=((L+1)*(L+2))/2

# print (orbitals(2))

eri = np.ndarray([nbf,nbf])
h = np.ndarray([nbf,nbf])

for i,p in enumerate(orbitals(L)):
    for j,q in enumerate(orbitals(L)):
        eri[i,j] = xERI2(
            BasisFunction(P, p, exps=[0.1, 1], coefs=[10, 0.5]),
            BasisFunction(Q, q, exps=[0.25, 3.1], coefs=[5, 1.37]),
        )
        h[i,j] = hERI2(
            BasisFunction(P, p, exps=[0.1], coefs=[10]),
            BasisFunction(Q, q, exps=[0.25], coefs=[5]),
        )
    

print(eri)
eri = np.dot(cart_to_spher_transform(L), eri)
eri = np.dot(eri, cart_to_spher_transform(L).T)


#print "H: ", h
#h = np.dot(cart_to_spher_transform(L), h)
#h = np.dot(h, cart_to_spher_transform(L).T)
#print "H': ", h

#print ("diff: %f" % (eri-h).sum())

# print On(0.1, 1, [3,2,1], 0.013, 2, [0, 1, 0])
# print R_transform(orbitals(2), P-Q)

del h

h = mmd2(Shell(L, [(0.1, 10), (1, 0.5)], P), Shell(L, [(0.25, 5), (3.1, 1.37)], Q))
print "H': ", h
print ("diff: %f" % (eri-h).sum())

