import numpy as np
from scipy.constants import m_e, e, c, pi
def d(x):
    return np.nanmedian(np.diff(x))


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

def emittance(x,x_dash):
    return np.sqrt(np.nanvar(x)*np.nanvar(x_dash) - (np.nanmean(x*x_dash))**2)

def fluka_beam_emittance(p_data,p_sel=None,phase_space=False,norm=True):
    if p_sel is None:
        N_p = np.shape(p_data)[0]
        p_sel = np.ones(N_p)>0
    x_m = 1e-2*p_data[p_sel,4]
    y_m = 1e-2*p_data[p_sel,5]
    px_norm = p_data[p_sel,11]*p_data[p_sel,8]/0.511*1e3
    py_norm = p_data[p_sel,11]*p_data[p_sel,9]/0.511*1e3
    pz_norm = p_data[p_sel,11]*p_data[p_sel,10]/0.511*1e3
    px_dash = p_data[p_sel,8]/p_data[p_sel,10]
    py_dash = p_data[p_sel,9]/p_data[p_sel,10]
    i_sel = np.abs(x_m-np.median(x_m))<(5*np.std(x_m))
    i_sel = i_sel*(np.abs(y_m-np.median(y_m))<(5*np.std(y_m)))
    pz_norm_mean = np.mean(pz_norm[i_sel])
    if phase_space:
        
        eta_x = emittance(x_m[i_sel] ,px_norm[i_sel] )
        eta_y = emittance(y_m[i_sel] ,py_norm[i_sel] )
        if not norm:
            eta_x = eta_x/pz_norm_mean
            eta_y = eta_y/pz_norm_mean
    
    else:
        eta_x = emittance(x_m[i_sel],px_dash[i_sel])
        eta_y = emittance(y_m[i_sel],py_dash[i_sel])
        if norm:
            eta_x = eta_x*pz_norm_mean
            eta_y = eta_y*pz_norm_mean
    
    return eta_x,eta_y

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


def jiggle_beams(part_data,N_rand,primary_div,primary_source_r, z_drift,
                    primary_sigma_t=0,z_prop=0,same_time=False):
    primary_sigma_r = z_drift*primary_div
    selected_p = np.copy(part_data)
    p_zero = propagate_beam(selected_p,z_prop,same_time=same_time)
    N_p = np.shape(p_zero)[0]
    part_dist = None
    for n in range(N_rand):
        
        p_plane = np.copy(p_zero)
        x_rand = np.random.randn(N_p)
        p_plane[:,4] = p_plane[:,4] + x_rand*primary_sigma_r
        p_plane[:,8] = p_plane[:,8] + x_rand*primary_div
        x_rand = np.random.randn(N_p)
        p_plane[:,4] = p_plane[:,4] + x_rand*primary_source_r

        y_rand = np.random.randn(N_p)
        p_plane[:,5] = p_plane[:,5] + y_rand*primary_sigma_r
        p_plane[:,9] = p_plane[:,9] + y_rand*primary_div
        y_rand = np.random.randn(N_p)
        p_plane[:,5] = p_plane[:,5] + y_rand*primary_source_r

        z_rand = np.random.randn(N_p)
        p_plane[:,6] = p_plane[:,6] + z_rand*primary_sigma_t*c*1e2

        if part_dist is None:
            part_dist = np.copy(p_plane)
        else:
            part_dist = np.append(part_dist,p_plane,axis=0)
    
    return part_dist

def get_part_dist_div_mrad(p_data,p_sel):
    if p_sel is None:
        N_p = np.shape(p_data)[0]
        p_sel = np.ones(N_p)>0
    px_dash = p_data[p_sel,8]/p_data[p_sel,10]
    py_dash = p_data[p_sel,9]/p_data[p_sel,10]
    source_mrad = np.array([np.std(px_dash)*1e3,np.std(py_dash)*1e3 ])
    return source_mrad


def get_part_dist_source_um(p_data,p_sel):
    if p_sel is None:
        N_p = np.shape(p_data)[0]
        p_sel = np.ones(N_p)>0
    x_m = 1e-2*p_data[p_sel,4]
    y_m = 1e-2*p_data[p_sel,5]
    source_um = np.array([np.std(x_m)*1e6,np.std(y_m)*1e6])
    return source_um

def get_part_dist_properties(part_dist,bins=None,p_weight=1,b_width=0.05,phase_space=False):
    if bins is None:
        bins = np.logspace(np.log10(5),np.log10(2000),100,endpoint=True)
    bin_width_MeV = np.diff(bins)
    pp_MeV = part_dist[:,10]*part_dist[:,11]*1e3
    H,Hx = np.histogram(part_dist[:,3],bins=bins)
    E_MeV_axis = (Hx[:-1] + d(Hx)/2)
    dN_p_per_5pcBW = p_weight*H/bin_width_MeV*E_MeV_axis*0.05*e*1e12
    # spec÷_list.append(dN_p_per_5pcBW)
    eta_p_g = []
    source_um = []
    source_mrad = []
    for ec in E_MeV_axis:
        E_min = ec*(1-b_width/2)
        E_max = ec*(1+b_width/2)
        p_sel = (pp_MeV>=E_min)*(pp_MeV<=E_max)
        eta_p = fluka_beam_emittance(part_dist,p_sel,phase_space=phase_space,norm=False)
        eta_p_g.append(eta_p)
        source_um.append(get_part_dist_source_um(part_dist,p_sel))
        source_mrad.append(get_part_dist_div_mrad(part_dist,p_sel))
    eta_p_g= np.array(eta_p_g)
    source_um = np.array(source_um)
    source_mrad = np.array(source_mrad)
    return E_MeV_axis,dN_p_per_5pcBW,eta_p_g,source_um,source_mrad