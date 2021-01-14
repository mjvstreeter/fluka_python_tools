import numpy as np
from scipy.constants import m_e, e, c, pi

# def read_usrbin_dat(file_path):

#     with open(file_path, "r") as f:
#         f_lines = f.readlines()
#     x_edges = []
#     Nxc = []
#     for s in f_lines[2:5]:
#         match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee][+-]\ *-?\ *[0-9]+)?')
#         nums = [float(x) for x in re.findall(match_number, s)]
#         x_edges.append(np.linspace(nums[0],nums[1],num=int(nums[2])+1,endpoint=True))
#         Nxc.append(int(nums[2]))
#     data =[]
#     n_l = []
#     for s in f_lines[9:80009]:
#         nums = [float(x) for x in re.findall(match_number, s)]
#         n_l.append(len(nums))
#         data.append(nums)
#     data = np.array(data)
#     data = data.reshape((Nxc[2],Nxc[0])).T;


def load_geometry_elements(file_path):
    with open(file_path, "r") as f:
        f_lines = f.readlines()
    geo_lines=[]
    geo_flag = False

    for line in f_lines:    
        if line.startswith('END'):
            geo_flag = False
        if geo_flag:
            if line.startswith(('SPH','RPP','XYP')):
                geo_lines.append(line)
            
        if line.startswith('GEOBEGIN'):
            geo_flag = True
    return geo_lines
