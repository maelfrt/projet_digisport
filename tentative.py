import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyomeca import Markers
from scipy.signal import butter, filtfilt, find_peaks


SUJET1_PATH  = "/home/onyxia/work/data"

for nom, chemin in [
    ("SUJET1_PATH", SUJET1_PATH)
]:
    if os.path.isdir(chemin):
        print(f"{nom} : OK -> {chemin}")
    else:
        print(f"{nom} : INTROUVABLE -> {chemin}  (a corriger ci-dessus)")

sujets = {"Sujet1": SUJET1_PATH}

for nom, chemin in sujets.items():
    fichiers = []
    for f in os.listdir(chemin):
        if f.endswith(".c3d"):
            fichiers.append(f)
    fichiers = sorted(fichiers)

    print(f"\n{nom} : {len(fichiers)} fichiers")
    for fichier in fichiers:
        print("   -", fichier)
