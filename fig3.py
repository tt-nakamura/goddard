import numpy as np
import matplotlib.pyplot as plt
from max_range import max_range,eq_of_motion
from max_range import l_unit,t_unit,v_unit
from scipy.constants import foot

zf = 144138*foot/l_unit

label = ['z = altitude  / km',
         'v = velocity  / km/s',
         r'm = mass  / $m_{\rm f}$',
         r'u = thrust  / $m_{\rm f}g$',
         r'D = drag  / $m_{\rm f}g$']

th0,tf,v = np.pi/4, 1, np.ones(64)
for z in [0.5*zf, 0.99*zf, zf]:
    th0,tf,v = max_range(z, th0, tf, v, iter=400)

_,_,z,m,F,D = eq_of_motion(th0,tf,v)

km,km_s = 1e3/l_unit, 1e3/v_unit
t = np.linspace(0, tf*t_unit, len(v))

plt.figure(figsize=(5,10))

for i,y in enumerate([z/km, v/km_s, m, F, D]):
    plt.subplot(5,1,i+1)
    plt.plot(t,y)
    plt.xlim(t[[0,-1]])
    plt.ylabel(label[i])

plt.xlabel('t = time  / sec')
plt.tight_layout()
plt.savefig('fig3.eps')
plt.show()
