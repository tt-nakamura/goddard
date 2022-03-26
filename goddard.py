# reference: A. E. Bryson
#  "Dynamic Optimization" example 10.3.1

import numpy as np
from scipy.integrate import solve_ivp
from scipy.constants import g as g0

l_unit = 6378140 # length unit r0 (earth radius) / m
t_unit = (l_unit/g0)**0.5 # time unit t0 / second
v_unit = l_unit/t_unit # velocity unit v0 / m/s

rtol = 1e-9 # tolerance for relative error

def goddard(F_max, mf, c, C_D=620, z0=2e-3, n_grid=128):
    """ Goddard problem to maximize
    the altitude of a souding rocket
    F_max = maximum thrust / m0*g0
    mf = mass when empty fuel / m0
    c =  specific fuel impulse / (g0*r0)**0.5
    C_D =  drag coeff / m0/(rho0*S*r0)
    z0 =  scale height of atmosphere / r0
    n_grid = number of grid points on time axis
    return t,y where
      t = time from lauch to max altitude
      y[0:5] = height, velocity, mass, thrust, drag
      t.shape = (n_grid,), y.shape = (5,n_grid)
    length unit r0 = earth radius
    time unit (r0/g0)**0.5 where g0 = 9.8m/s^2
    mass unit m0 = initial mass of rocket
    rho0 = mass density of air at earth surface
    S = rocket sectional area (perpendicular to axis)
    """
    def drag(v,h):
        return C_D*v**2/2*np.exp(-h/z0)

    def eom_boost(t,y):# equation of motion for boost phase
        h,v,m = y
        return [v, (F_max - drag(v,h))/m - 1/(1+h)**2, -F_max/c]

    def eom_sustain(t,y):# equation of motion for sustain phase
        h,v,m = y
        g,gh = 1/(1+h)**2, -2/(1+h)**3
        D = drag(v,h)
        cv = c/v
        f = (c*(c+v)/z0 - (1+2*cv)*g + m*gh*c*c/D)/(1+2*cv*(2+cv))
        return [v, f, -(D + m*(f+g))/c]

    def eom_coast(t,y):# equation of motion for coast phase
        h,v,m = y
        return [v, -drag(v,h)/m - 1/(1+h)**2, 0]

    def thrust_sustain(y):
        h,v,m = y
        g,gh = 1/(1+h)**2, -2/(1+h)**3
        D = drag(v,h)
        cv = c/v
        f = (c*(c+v)/z0 - (1+2*cv)*g + m*gh*c*c/D)/(1+2*cv*(2+cv))
        return D + m*(f+g)

    t,t1 = 0, 0.01 # initial guess
    h,v,m = 0,0,1
    while True: # newton-raphson iteration
        s = solve_ivp(eom_boost, [t,t1], [h,v,m])
        t = t1
        h,v,m = s.y[:,-1]
        g,gh = 1/(1+h)**2, -2/(1+h)**3
        D = drag(v,h)
        Dh,Dv = -D/z0, 2*D/v
        dh,dv,dm = eom_boost(t,[h,v,m])
        f = m*g - D*(1 + v/c)
        df = dm*g + m*gh*dh - (Dv*dv + Dh*dh)*(1 + v/c) - D*dv/c
        dt = -f/df
        t1 += dt
        if np.abs(dt) < rtol*np.abs(t): break

    print('end of boost phase: {} sec'.format(t1*t_unit))

    t,t2 = t1, t1+0.01 # initial guess
    while True: # newton-raphson iteration
        s = solve_ivp(eom_sustain, [t,t2], [h,v,m])
        t = t2
        h,v,m = s.y[:,-1]
        _,_,dm = eom_sustain(t,[h,v,m])
        dt = (mf-m)/dm
        t2 += dt
        if np.abs(dt) < rtol*np.abs(t): break

    print('end of sustain phase: {} sec'.format(t2*t_unit))

    t,tf = t2, t2+0.01 # initial guess
    while True: # newton-raphson iteration
        s = solve_ivp(eom_coast, [t,tf], [h,v,m])
        t = tf
        h,v,m = s.y[:,-1]
        _,dv,_ = eom_coast(t,[h,v,m])
        dt = -v/dv
        tf += dt
        if np.abs(dt) < rtol*np.abs(t): break

    print('end of coast phase: {} sec'.format(tf*t_unit))

    dt = tf/(n_grid-1)# grid interval
    t01 = np.linspace(0,t1,int(t1/dt)+1)
    t12 = np.linspace(t1,t2,int((t2-t1)/dt)+1)
    t23 = np.linspace(t2,tf,n_grid-len(t01)-len(t12)+2)
    s1 = solve_ivp(eom_boost, [0,t1], [0,0,1], t_eval=t01)
    s2 = solve_ivp(eom_sustain, [t1,t2], s1.y[:,-1], t_eval=t12)
    s3 = solve_ivp(eom_coast, [t2,tf], s2.y[:,-1], t_eval=t23)

    t = np.r_[s1.t, s2.t, s3.t]# don't remove overlap
    y = np.c_[s1.y, s2.y ,s3.y]
    F = thrust_sustain(s2.y)
    F = np.r_[np.full(len(s1.t), F_max), F, np.zeros(len(s3.t))]
    D = drag(y[1],y[0])
    y = np.vstack((y,F,D))
    return t,y


def NotGoddard(F_max, mf, c, C_D=620, z0=2e-3, n_grid=128):
    """ F = F_max always """
    def drag(v,h):
        return C_D*v**2/2*np.exp(-h/z0)

    def eom_boost(t,y):# equation of motion for boost phase
        h,v,m = y
        return [v, (F_max - drag(v,h))/m - 1/(1+h)**2, -F_max/c]

    def eom_coast(t,y):# equation of motion for coast phase
        h,v,m = y
        return [v, -drag(v,h)/m - 1/(1+h)**2, 0]

    t,t1 = 0, 0.01 # initial guess
    h,v,m = 0,0,1
    while True: # newton-raphson iteration
        s = solve_ivp(eom_boost, [t,t1], [h,v,m])
        t = t1
        h,v,m = s.y[:,-1]
        _,_,dm = eom_boost(t,[h,v,m])
        dt = (mf-m)/dm
        t1 += dt
        if np.abs(dt) < rtol*np.abs(t): break

    print('end of boost phase: {} sec'.format(t1*t_unit))

    t,tf = t1, t1+0.01 # initial guess
    while True: # newton-raphson iteration
        s = solve_ivp(eom_coast, [t,tf], [h,v,m])
        t = tf
        h,v,m = s.y[:,-1]
        _,dv,_ = eom_coast(t,[h,v,m])
        dt = -v/dv
        tf += dt
        if np.abs(dt) < rtol*np.abs(t): break

    print('end of coast phase: {} sec'.format(tf*t_unit))
    
    dt = tf/(n_grid-1)# grid interval
    t01 = np.linspace(0,t1,int(t1/dt)+1)
    t12 = np.linspace(t1,tf,n_grid-len(t01)+1)
    s1 = solve_ivp(eom_boost, [0,t1], [0,0,1], t_eval=t01)
    s2 = solve_ivp(eom_coast, [t1,tf], s1.y[:,-1], t_eval=t12)

    t = np.r_[s1.t, s2.t]# don't remove overlap
    y = np.c_[s1.y, s2.y]
    F = np.r_[np.full(len(s1.t), F_max), np.zeros(len(s2.t))]
    D = drag(y[1],y[0])
    y = np.vstack((y,F,D))
    return t,y
