import matplotlib.pyplot as plt
from goddard import goddard,NotGoddard
from goddard import l_unit,t_unit,v_unit

label = ['z = altitude / km',
         'v = velocity / km/s',
         r'm = mass / $m_0$',
         r'u = thrust / $m_0g_0$',
         r'D = drag / $m_0g_0$']

t,y = goddard(3.5, 0.6, 0.5)
u,x = NotGoddard(3.5, 0.6, 0.5)

km,km_s = 1e3/l_unit, 1e3/v_unit
t *= t_unit; y[0] /= km; y[1] /= km_s
u *= t_unit; x[0] /= km; x[1] /= km_s

print('optimal control')
print('max altitude {} at time {}:'.format(y[0,-1],t[-1]))
print('greedy control')
print('max altitude {} at time {}:'.format(x[0,-1],u[-1]))

plt.figure(figsize=(5,10))

for i in range(5):
    plt.subplot(5,1,i+1)
    plt.plot(t, y[i], 'b', label='optimal')
    plt.plot(u, x[i], 'r', label='greedy')
    plt.xlim(t[[0,-1]])
    plt.ylabel(label[i])
    plt.legend()

plt.xlabel('t = time / sec')
plt.tight_layout()
plt.savefig('fig1.eps')
plt.show()
