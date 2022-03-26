# reference: A. E. Bryson
#  "Dynamic Optimization" problem 10.3.2

import numpy as np
from scipy.integrate import cumtrapz
from scipy.optimize import fmin_slsqp
from scipy.constants import foot,pound,slug,g

m_init = 1000*pound # initial weight / kg
m_final = 500*pound # final weight / kg
rho0 = 0.0021*slug/foot**3 # density of air at ground / kg/m^3
H = 30000*foot # scale height of atmosphere / m
CDS = 0.47*foot**2 # drag coeff * reference area / m^2
fsi = 6698*foot # fuel specific impulse / m/s

m_unit = m_final # mass unit / kg
l_unit = 2*m_final/CDS/rho0 # length unit l0 / m
t_unit = (l_unit/g)**0.5 # time unit t0 / sec
v_unit = l_unit/t_unit # velocity unit v0 / m/s

beta = 1/(H/l_unit)
c = fsi/v_unit
m0 = m_init/m_unit

def eq_of_motion(th0, tf, v):
    """
    th0 = initial path angle / radian
    tf = flight time / t0
    v = velocity history (1d-array) / v0
    """
    dt = tf/(len(v)-1) # time interval
    th = np.arcsinh(np.tan(th0)) - cumtrapz(1/v, dx=dt, initial=0)
    th = np.arctan(np.sinh(th)) # flight path angle
    x = cumtrapz(v*np.cos(th), dx=dt, initial=0) # horizontal range
    h = cumtrapz(v*np.sin(th), dx=dt, initial=0) # vertical altitude
    s = cumtrapz(np.sin(th), dx=dt, initial=0)
    ev = np.exp((v+s)/c) # integrating factor
    D = v**2*np.exp(-beta*h) # drag
    De = cumtrapz(D*ev, dx=dt, initial=0)
    m = (m0 - De/c)/ev # mass
    F = -np.gradient(m,dt)*c # thrust
    return th,x,h,m,F,D

def max_range(hf, th0, tf, v, **kw):
    """
    hf = final altitude / l0 (l0 = 2*m_final/(CDS*rho0))
    th0 = initial guess for initial path angle / radian
    tf = initial guess for flight time / t0
    v = initial guess for velocity history (1d-array) / v0
    kw = keyword arguments passed to fmin_slsqp
    return th0,tf,v that maximize xf for given hf
    """
    def f_min(p):# horizontal range
        return -eq_of_motion(p[0],p[1],p[2:])[1][-1]
    def eqcon(p):# height and fuel constraints
        _,_,h,m,_,_ = eq_of_motion(p[0],p[1],p[2:])
        return [h[-1] - hf, m[-1] - 1]
    def ieqcon(p):# thrust constraints
        _,_,_,m,_,_ = eq_of_motion(p[0],p[1],p[2:])
        return -np.diff(m)

    p = fmin_slsqp(f_min, np.r_[th0,tf,v],
                   f_eqcons=eqcon, f_ieqcons=ieqcon, **kw)
    return p[0],p[1],p[2:]
