import torch
import numpy as np
from plyfile import PlyData, PlyElement

def construct_list_of_attributes(model):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(model.features_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(model.scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(model.rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l

def save_ply(model, path):
    xyz = model.xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = model.features_dc.detach().contiguous().cpu().numpy()
    opacities = model.opacity.detach().cpu().numpy()
    scale = model.scaling.detach().cpu().numpy()
    rotation = model.rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(model)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el], text=True).write(path)
