def RK2(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt*k1, u)
    return x + dt/2*(k1 + k2)

def RK4(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt/2*k1, u)
    k3 = f(x + dt/2*k2, u)
    k4 = f(x + dt*k3, u)
    return x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
