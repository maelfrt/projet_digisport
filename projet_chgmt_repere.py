import numpy as np
import matplotlib.pyplot as plt
from pyomeca import Markers
from scipy.spatial.transform import Rotation

marqueurs_rom = Markers.from_c3d("data/ROM_S1.c3d")
marqueurs = Markers.from_c3d("data/REEL_0008.c3d")
temps = np.arange(marqueurs.shape[2]) / marqueurs.attrs["rate"]
print(marqueurs.channel.values.tolist())


def position(essai, nom):
    return essai.sel(channel=nom).values[0:3, :]  # (3, n_frames), en mm


def normaliser(v):
    return v / np.linalg.norm(v)