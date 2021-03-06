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


def radial_beam_hist(p_data,r_axis=None,z_axis=None,weight=None):
    if r_axis is None:
        r_axis = 100
    if z_axis is None:
        z_axis = 100
    
    e_r = np.sqrt(p_data[:,4]**2+p_data[:,5]**2) # cm
    e_z = p_data[:,6]
    H,r_axis,z_axis = np.histogram2d(e_r, e_z,bins=(r_axis,z_axis), #, bins=(100,100), #
                                 range=None, normed=None, weights=np.ones_like(e_r)*weight, density=None)

    dr = np.mean(np.diff(r_axis))
    dz = np.mean(np.diff(z_axis))
    r_area = pi*(r_axis[1:]**2-r_axis[:-1]**2).reshape(1,-1)
    n_e_zr = (H.T)/(r_area*dz) # particles per cm3

    return n_e_zr, r_axis, z_axis

def radial_spectrum(p_data,r_axis=None,E_axis=None,weight=None):
    if r_axis is None:
        r_axis = 100
    if E_axis is None:
        E_axis = 100
    
    e_r = np.sqrt(p_data[:,4]**2+p_data[:,5]**2) # cm
    e_pz = p_data[:,11]
    H,r_axis,E_axis = np.histogram2d(e_r, e_pz,bins=(r_axis,E_axis), #, bins=(100,100), #
                                 range=None, normed=None, weights=np.ones_like(e_r)*weight, density=None)

    dr = np.mean(np.diff(r_axis))
    dE = (np.diff(E_axis))[:,np.newaxis]
    r_area = pi*(r_axis[1:]**2-r_axis[:-1]**2).reshape(1,-1)
    n_e_Er = (H.T)/(r_area*dE) # particles per cm3

    return n_e_Er, r_axis, E_axis

def sim_beam_dist(p_data_in,r_axis,z_axis,p_weight=1,primary_energy=1,
                  primary_sigma_r = 0.1, primary_div= 1e-3,primary_sigma_t = 10e-15,
                  N_rand = 10,r_max = 300e-4):

    p_data = np.copy(p_data_in)
    N_p = np.size(p_data,axis=0)

    g_axis = np.linspace(0,primary_energy,num=400)
    
    E_MeV_log = np.logspace(-3,0,num=100,endpoint=True)*primary_energy

    n_zr_sum = 0
    n_e_Er_sum = 0
    for n in range(N_rand):
        # random jiggle transverse position and divergence
        x_rand = np.random.randn(N_p)
        p_data[:,4] = p_data[:,4] + x_rand*primary_sigma_r
        p_data[:,8] = p_data[:,8] + x_rand*primary_div
        y_rand = np.random.randn(N_p)
        p_data[:,5] = p_data[:,5] + y_rand*primary_sigma_r
        p_data[:,9] = p_data[:,9] + y_rand*primary_div
        # random jiggle longitudinal position
        z_rand = np.random.randn(N_p)
        p_data[:,6] = p_data[:,6] + z_rand*primary_sigma_t*c*1e2

        n_zr,r,z = radial_beam_hist(p_data,r_axis=r_axis,z_axis=z_axis,weight=p_weight/N_rand)
        n_zr_sum = n_zr_sum+n_zr
        
        e_r = np.sqrt(p_data[:,4]**2+p_data[:,5]**2) # cm
        e_z = p_data[:,6]
        
        n_e_Er, r, E_axis = radial_spectrum(p_data,r_axis=r_axis,E_axis=E_MeV_log,weight=p_weight/N_rand)

        n_e_Er_sum = n_e_Er_sum+n_e_Er
    
    part_dist = dict(n_zr= n_zr,r=r,z=z, E_MeV_log=E_MeV_log,n_Er=n_e_Er_sum)
    return part_dist