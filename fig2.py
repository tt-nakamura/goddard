import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from max_range import max_range,eq_of_motion,l_unit
from scipy.constants import foot,degree

zf = [0, 50000, 100000, 120000, 138000, 144138]
zf = np.asarray(zf)*foot/l_unit
km = 1e3/l_unit

xe,ze = [],[]
th0,tf,v = np.pi/4, 1, np.ones(64)
for h in zf:
    th0,tf,v = max_range(h, th0, tf, v, iter=400)
    _,x,z,_,_,_ = eq_of_motion(th0,tf,v)
    xe.append(x[-1])
    ze.append(z[-1])
    plt.plot(x/km, z/km)
    print('initial path angle: {} deg'.format(th0/degree))

env = interp1d(xe,ze,'cubic') # envelope
xe = np.linspace(xe[-1], xe[0], 64)
ze = env(xe)

plt.axis('equal')
plt.plot(xe/km, ze/km, 'k:')
plt.ylim([0,46])
plt.xlabel('x = range  / km', fontsize=14)
plt.ylabel('z = altitude  / km', fontsize=14)
plt.tight_layout()
plt.savefig('fig2.eps')
plt.show()
