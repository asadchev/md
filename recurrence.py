import sympy
from sympy import Symbol

x = sympy.Symbol('x')
y = sympy.Symbol('y')
z = sympy.Symbol('z')
Q = [x,y,z]

def R(r,m=0):
  #print r
  if min(r) < 0: return 0
  i = None
  if r[0]: i = 0
  if r[1]: i = 1
  if r[2]: i = 2
  if i is None: return Symbol("s(%i)" % m)
  r1 = list(r); r1[i] -= 1
  r2 = list(r); r2[i] -= 2
  #print r,r1,r2
  return sympy.expand(Q[i]*R(r1,m+1) - (r[i]-1)*R(r2,m+1))

# print R([4,0,0])
# print R([3,1,0])

s0s = [ Symbol("s(%i)" % m) for m in range(20) ]

print R([2,1,0])
print R([3,0,0])
print R([4,3,2])
print sympy.collect(R([12,0,0]), s0s)
