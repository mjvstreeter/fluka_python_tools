import numpy as np
from scipy.constants import m_e, e, c, pi
import re
from physics_tools.general_tools import load_object
import pickle5

def load_object5(filename):
    with open(filename, 'rb') as fid:
        return pickle5.load(fid)

def read_usrbin_dat(file_path):

    with open(file_path, "r") as f:
        f_lines = f.readlines()
    x_edges = []
    Nxc = []
    for s in f_lines[2:5]:
        match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee][+-]\ *-?\ *[0-9]+)?')
        nums = [float(x) for x in re.findall(match_number, s)]
        x_edges.append(np.linspace(nums[0],nums[1],num=int(nums[2])+1,endpoint=True))
        Nxc.append(int(nums[2]))

    full_data = []
    N_group = int(Nxc[0]*Nxc[2]/10)
    for iy in range(Nxc[1]):
        data =[]
        n_l = []
        for s in f_lines[(9+iy*N_group):(9+((iy+1)*N_group))]:
            nums = [float(x) for x in re.findall(match_number, s)]
            n_l.append(len(nums))
            data.append(nums)
        data = np.array(data)
        data = data.reshape((Nxc[2],Nxc[0])).T
        full_data.append(data)
    return full_data, x_edges


def load_geometry_elements(file_path):
    with open(file_path, "r") as f:
        f_lines = f.readlines()
    geo_lines=[]
    geo_flag = False
    match_number = re.compile(' -?\ *[0-9]+\.?[0-9]*(?:[Ee][+-]\ *-?\ *[0-9]+)?')

    for line in f_lines:    
        if line.startswith('END'):
            geo_flag = False
        if geo_flag:
            if line.startswith(('SPH','RPP','XYP','RCC')):
                geo_lines.append(line)
            
        if line.startswith('GEOBEGIN'):
            geo_flag = True
    geo_list = []
    for line in geo_lines:
        nums = [float(x) for x in re.findall(match_number, line)]
        geo_list.append(
            dict(
                Type=line.split()[0],
                Name=line.split()[1],
                x=nums
                ))

    
    return geo_list, geo_lines

def get_particle_data(data_array_file,species_list = [3,4,7],
                from_select = [5], to_select = None,
                z_lims = None, pz_lims = [0,None]):
    try:
        d_obj  = load_object(data_array_file)
    except:
        d_obj  = load_object5(data_array_file)
    try:
        data_array=d_obj['data_array']
        N_files=d_obj['N_files']
    except:
        data_array = d_obj
        N_files = -1
    p_list = []
    for s in species_list:
        p_sel = data_array.astype(int)[:,2]==int(s)

        if from_select is not None:
            f_sel = p_sel*0.0
            for f in from_select:
                f_sel += data_array.astype(int)[:,0]==int(f)
            p_sel = p_sel*f_sel

        if to_select is not None:
            f_sel = p_sel*0.0
            for f in to_select:
                f_sel += data_array.astype(int)[:,1]==int(f)
            p_sel = p_sel*f_sel

        if z_lims is not None:
            f_sel = np.ones_like(p_sel)
            if z_lims[0] is not None:
                f_sel = f_sel*(data_array[:,6]>=z_lims[0])
            if z_lims[1] is not None:
                f_sel = f_sel*(data_array[:,6]<=z_lims[1])
            p_sel = p_sel*f_sel

        if pz_lims is not None:
            f_sel = np.ones_like(p_sel)
            if pz_lims[0] is not None:
                f_sel = f_sel*(data_array[:,11]>=pz_lims[0])
            if pz_lims[1] is not None:
                f_sel = f_sel*(data_array[:,11]<=pz_lims[1])
            p_sel = p_sel*f_sel

        p_list.append(data_array[p_sel>0,:])
    return p_list, N_files

def get_primary_energy(inp_file_path):
    with open(inp_file_path, "r") as f:
        text = f.readlines()
  
    s = [x for x in text if x.startswith('BEAM ')]
    primary_energy = float(s[0].strip().split()[1])*-1
    return primary_energy


