import numpy as np
from scipy.constants import m_e, e, c, pi



def propagate_beam(p_data,z,same_time=False):
    p_data_out = np.copy(p_data)
    g = np.sqrt(1+(p_data_out[:,3]*1e6*e/(m_e*c**2))**2)
    b = np.sqrt(1-1/g**2)
    beta_z = b*p_data_out[:,10]
    dz = 1e-2*(z-p_data[:,6])
    t_0 = p_data[:,7]
    
    if same_time:
        iSel = np.argmin(np.abs(dz/(beta_z*c)+t_0))
        t_1 = dz/(beta_z*c)+t_0
        t_1 = t_1[iSel]
        #print(t_1)
    else:
        t_1 = dz/(beta_z*c)+t_0
    dt = (t_1-t_0)
    p_data_out[:,4] = p_data[:,4] + dt*b*p_data_out[:,8]*c*1e2
    p_data_out[:,5] = p_data[:,5] + dt*b*p_data_out[:,9]*c*1e2
    p_data_out[:,6] = p_data[:,6] + dt*b*p_data_out[:,10]*c*1e2
    p_data_out[:,7] = p_data[:,7] + dt
    return p_data_out