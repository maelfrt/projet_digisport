# -*- coding: utf-8 -*-
"""
Analyse de donnees de capture de mouvement (C3D) — Digisport M1 — CORRIGE

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt, find_peaks
from pyomeca import Markers

# %%
# ### Chemins du projet (code fourni)

# %%
# À MODIFIER : chemin complet vers le dossier de donnees de chaque sujet
#

SUJET1_PATH = "E:/2_ATER/S1/UEF2_Sciences du sport/UE1_Modelisation Biomecanique/Seance_2/Donnees_C3D/Sujet1"
SUJET2_PATH = "E:/2_ATER/S1/UEF2_Sciences du sport/UE1_Modelisation Biomecanique/Seance_2/Donnees_C3D/Sujet2"

for nom, chemin in [("SUJET1_PATH", SUJET1_PATH), ("SUJET2_PATH", SUJET2_PATH)]:
    if os.path.isdir(chemin):
        print(f"{nom} : OK -> {chemin}")
    else:
        print(f"{nom} : INTROUVABLE -> {chemin}  (a corriger ci-dessus)")

# %%
# **A savoir sur les donnees (deja verifie) :**
# - 64 marqueurs corps entier, nommage `p...` (ex. `pRightASI`,
#   `pRightHeelFoot`).
# - Aucune occlusion/trou dans les marqueurs, unites mm.
# - Axe vertical = **Z** (indice 2).
# - `pHipOrigin` est un marqueur technique quasi immobile : ne pas l'utiliser
#   comme centre du bassin — preferer la moyenne de `pRightASI`, `pLeftASI`,
#   `pRightCSI`, `pLeftCSI`.
# - Pas de plateforme de force / voie analogique exploitable.

# %%
# ### Exercice — explorer les donnees disponibles
#
# Pour chacun des 2 sujets (`SUJET1_PATH` et `SUJET2_PATH`), listez les
# fichiers `.c3d` presents dans son dossier.

# %%
sujets = {"Sujet1": SUJET1_PATH, "Sujet2": SUJET2_PATH}

for nom, chemin in sujets.items():
    fichiers = []
    for f in os.listdir(chemin):
        if f.endswith(".c3d"):
            fichiers.append(f)
    fichiers = sorted(fichiers)

    print(f"\n{nom} : {len(fichiers)} fichiers")
    for fichier in fichiers:
        print("   -", fichier)

# %%
# ### Exercice — premier chargement
#
# Chargez `data/Sujet1/Carbone12.c3d` avec `Markers.from_c3d()` pour verifier
# que tout fonctionne. On reutilise cet essai (`rep`) dans toute la suite du
# notebook jusqu'a la partie 4.

# %%
fichier_test = os.path.join(SUJET1_PATH, "Carbone12.c3d")
rep = Markers.from_c3d(fichier_test)

print("Installation OK, fichier de test chargé.")
print("Fréquence d'échantillonnage :", rep.attrs["rate"], "Hz")
print("Nombre de marqueurs :", len(rep.channel))
print("Nombre d'images (frames) :", rep.shape[-1])

# %%
# ## Partie 2 — Lecture d'un fichier `.c3d`
#
# On continue avec l'essai charge en partie 1 (`rep` = Sujet1/Carbone12).

# %%
# ### Exercice — construire un chemin de fichier a partir de ses "composants"
#
# Plutot que d'ecrire le nom du fichier en dur, construisez-le a partir de la
# condition et du niveau (utile pour les boucles des parties 6-7). Vous devez
# retrouver exactement le meme fichier que `fichier_test` plus haut.

# %%
condition_tmp = "Carbone"
niveau_tmp = "12"

file_path_tmp = os.path.join(SUJET1_PATH, f"{condition_tmp}{niveau_tmp}.c3d")
print("Le nom complet est :", file_path_tmp)
print("Identique à fichier_test :", file_path_tmp == fichier_test)

# %%
# ### Exercice — duree de l'enregistrement
#
# Tout est dans `rep.attrs` : la frequence, la premiere et la derniere frame.

# %%
sample_rate = rep.attrs["rate"]
mocap_time_length = (rep.attrs["last_frame"] - rep.attrs["first_frame"]) / sample_rate
print("La durée de la capture est de", round(mocap_time_length, 2), "s")

time = np.array(rep.time)
rep_mklist = np.array(rep.channel)
print("Nombre de marqueurs :", len(rep_mklist))
print("Premiers marqueurs :", rep_mklist[:5])

# %%
# ### Exercice — longueur de cuisse droite (hanche -> genou)
#
# 1. Recuperez les marqueurs de hanche (`pRightGreaterTrochanter`) et de genou
#    (moyenne de `pRightKneeLatEpicondyle` et `pRightKneeMedEpicondyle`).
# 2. Calculez la distance entre les deux a chaque instant, par 2 methodes
#    differentes (elles doivent donner le meme resultat).
# 3. Calculez la moyenne et l'ecart-type de cette longueur.

# %%
hip_right = np.array(rep.sel(channel="pRightGreaterTrochanter", axis=["x", "y", "z"]))
knee_lat = np.array(rep.sel(channel="pRightKneeLatEpicondyle", axis=["x", "y", "z"]))
knee_med = np.array(rep.sel(channel="pRightKneeMedEpicondyle", axis=["x", "y", "z"]))
knee_right = (knee_lat + knee_med) / 2

# Méthode 1 : calcul manuel avec np.sqrt() et np.sum()
thigh_length_1 = np.sqrt(np.sum((hip_right - knee_right) ** 2, axis=0))

# Méthode 2 : np.linalg.norm()
thigh_length_2 = np.linalg.norm(hip_right - knee_right, axis=0)

print("Vérification méthode 1 vs méthode 2 :", np.allclose(thigh_length_1, thigh_length_2))

mean_thigh_length = np.nanmean(thigh_length_1)
std_thigh_length = np.nanstd(thigh_length_1)
print("Longueur de cuisse droite :", round(mean_thigh_length, 2), "±", round(std_thigh_length, 2), "mm")

# %%
# ### Exercice — verifier la rigidite du segment jambe (genou -> cheville)
#
# La distance entre 2 marqueurs d'un meme segment osseux rigide ne devrait
# quasiment pas varier au cours du temps. Marqueurs de cheville :
# `pRightLatMalleolus`, `pRightMedMalleolus`. Calculez le coefficient de
# variation (CV = ecart-type / moyenne x 100).

# %%
ankle_lat = np.array(rep.sel(channel="pRightLatMalleolus", axis=["x", "y", "z"]))
ankle_med = np.array(rep.sel(channel="pRightMedMalleolus", axis=["x", "y", "z"]))
ankle_right = (ankle_lat + ankle_med) / 2

shank_length = np.linalg.norm(knee_right - ankle_right, axis=0)
mean_shank = np.nanmean(shank_length)
std_shank = np.nanstd(shank_length)
cv_shank = std_shank / mean_shank * 100

print("Longueur de jambe (genou-cheville) :", round(mean_shank, 2), "±",
      round(std_shank, 2), "mm  (CV =", round(cv_shank, 2), "%)")

# %%
# ## Partie 3 — Visualisation graphique
#
# On continue avec `rep` (Carbone12.c3d, chargé en Partie 1), `time` et
# `sample_rate` déjà disponibles. Objectif : tracer une trajectoire, une
# cinématique complète (position/vitesse/accélération), et la hauteur du
# bassin au cours du temps.

# %%
# ### Exercice 3.1 — trajectoire (x, y, z) d'un marqueur
#
# Tracez les 3 coordonnées du talon droit (`pRightHeelFoot`) en fonction du
# temps, une courbe par axe, avec légende et titres. Convertissez les mm en m
# (diviser par 1000) pour des valeurs plus lisibles.

# %%
heel_right = np.array(rep.sel(channel="pRightHeelFoot", axis=["x", "y", "z"]))
heel_right = heel_right / 1000  # mm -> m

plt.figure()
plt.plot(time, heel_right[0, :], label="X (avant/arrière)")
plt.plot(time, heel_right[1, :], label="Y (latéral)")
plt.plot(time, heel_right[2, :], label="Z (vertical)")
plt.xlabel("Temps (s)")
plt.ylabel("Position du talon droit (m)")
plt.title("Trajectoire du marqueur pRightHeelFoot")
plt.legend()
plt.show()

# %%
# ### Exercice 3.2 — cinématique complète (position / vitesse / accélération)
#
# Figure à 3 sous-repères empilés (`plt.subplots(3, 1, sharex=True)`) affichant
# position, vitesse (`np.diff(...) * fréquence`) et accélération du talon
# droit, un axe (X, Y, Z) par couleur.

# %%
heel_velocity = np.diff(heel_right, axis=1) * sample_rate
heel_acceleration = np.diff(heel_velocity, axis=1) * sample_rate

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 10))

colors = {"X": "blue", "Y": "orange", "Z": "brown"}

for i, axis_name in enumerate(["X", "Y", "Z"]):
    ax[0].plot(time, heel_right[i, :], label=axis_name, c=colors[axis_name])
    ax[1].plot(time[1:], heel_velocity[i, :], c=colors[axis_name])
    ax[2].plot(time[2:], heel_acceleration[i, :], c=colors[axis_name])

ax[0].set_ylabel("Position (m)")
ax[0].legend(loc="upper right")
ax[1].set_ylabel("Vitesse (m/s)")
ax[2].set_ylabel("Accélération (m/s²)")
ax[2].set_xlabel("Temps (s)")

fig.suptitle("Cinématique du marqueur pRightHeelFoot (talon droit)", fontsize=16)
plt.show()

# %%
# ### Exercice 3.3 — hauteur du bassin au cours du temps
#
# Le centre du bassin est approché par la moyenne de 4 marqueurs des crêtes
# iliaques : `pRightASI`, `pLeftASI`, `pRightCSI`, `pLeftCSI`.
# **Ne pas utiliser `pHipOrigin`** : c'est un marqueur technique quasi immobile
# (voir README.md). Tracez la position verticale (Z) du bassin, avec grille et
# légende.

# %%
rasi = np.array(rep.sel(channel="pRightASI", axis=["x", "y", "z"]))
lasi = np.array(rep.sel(channel="pLeftASI", axis=["x", "y", "z"]))
rcsi = np.array(rep.sel(channel="pRightCSI", axis=["x", "y", "z"]))
lcsi = np.array(rep.sel(channel="pLeftCSI", axis=["x", "y", "z"]))

pelvis = (rasi + lasi + rcsi + lcsi) / 4
pelvis = pelvis / 1000  # mm -> m

plt.figure()
plt.plot(time, pelvis[2, :], label="Hauteur du bassin (Z)")
plt.xlabel("Temps (s)")
plt.ylabel("Position verticale (m)")
plt.title("Déplacement vertical du bassin")
plt.xlim(0, time[-1])
plt.legend()
plt.grid()
plt.show()

print("Amplitude verticale du bassin sur cet essai :",
      round((pelvis[2, :].max() - pelvis[2, :].min()) * 1000, 1), "mm")

# %%
# ## Partie 4 — Filtrage des données
#
# Pourquoi filtrer ? Le bruit de mesure, peu visible en position, est amplifié
# par la dérivation lors du calcul de la vitesse et surtout de l'accélération.
# On applique un filtre passe-bas de Butterworth pour l'atténuer.

# %%
# ### Démonstration sur un signal simulé
#
# On simule une oscillation "propre" (1 cycle par seconde, comme une foulée)
# à laquelle on ajoute un bruit haute fréquence (20 Hz, comme une erreur de
# mesure) — code fourni, à exécuter tel quel.

# %%
freq_sample_demo = 100
time_demo = np.linspace(0, 10, 10 * freq_sample_demo)
signal_clean = 0.1 * np.sin(2 * np.pi * time_demo * 1)
signal_noise = 0.03 * np.sin(2 * np.pi * time_demo * 20)
signal_noisy = signal_clean + signal_noise

velocity_clean = np.diff(signal_clean) * freq_sample_demo
velocity_noisy = np.diff(signal_noisy) * freq_sample_demo
accel_clean = np.diff(velocity_clean) * freq_sample_demo
accel_noisy = np.diff(velocity_noisy) * freq_sample_demo

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].set_xlim(0, 5)
ax[0].plot(time_demo, signal_noisy, label="avec bruit")
ax[0].plot(time_demo, signal_clean, label="sans bruit")
ax[0].set_ylabel("Position")
ax[1].plot(time_demo[1:], velocity_noisy)
ax[1].plot(time_demo[1:], velocity_clean)
ax[1].set_ylabel("Vitesse")
ax[2].plot(time_demo[2:], accel_noisy, label="avec bruit")
ax[2].plot(time_demo[2:], accel_clean, label="sans bruit")
ax[2].set_ylabel("Accélération")
ax[0].legend()
fig.suptitle("Un bruit discret en position devient une forte distorsion en accélération")
plt.show()

# %%
# ### Exercice 4.1 — écrire la fonction de filtrage passe-bas (Butterworth)
#
# Complétez les deux fonctions ci-dessous (`scipy.signal.butter` et
# `scipy.signal.filtfilt`).

# %%
def butter_lowpass(cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=2):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data, axis=-1)

# %%
# ### Application sur les données simulées (code fourni)
#
# `CUTOFF = 10` Hz est la fréquence de coupure classique en analyse du
# mouvement humain : on la réutilisera dans toutes les parties suivantes.

# %%
CUTOFF = 10  # Hz, fréquence de coupure classique pour l'analyse de mouvement
signal_filt = lowpass_filter(signal_noisy, CUTOFF, freq_sample_demo)
velocity_filt = np.diff(signal_filt) * freq_sample_demo
accel_filt = np.diff(velocity_filt) * freq_sample_demo

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].set_xlim(0, 5)
ax[0].plot(time_demo, signal_clean, c="orange", label="signal propre")
ax[0].plot(time_demo, signal_filt, c="green", label="signal filtré")
ax[1].plot(time_demo[1:], velocity_clean, c="orange")
ax[1].plot(time_demo[1:], velocity_filt, c="green")
ax[2].plot(time_demo[2:], accel_clean, c="orange", label="propre")
ax[2].plot(time_demo[2:], accel_filt, c="green", label="filtré")
ax[0].legend()
ax[2].legend()
fig.suptitle("Signal filtré vs signal propre : très proches, même en accélération")
plt.show()

# %%
# ### Exercice 4.2 — filtrage d'un vrai marqueur (talon droit)
#
# On réutilise `rep`, `sample_rate`, `time` et `heel_right` (en mètres, déjà
# calculés en Partie 3). `pyomeca` propose un filtrage intégré, équivalent au
# filtre scipy ci-dessus, directement applicable à l'objet `Markers` complet
# (tous les marqueurs à la fois) : `rep.meca.low_pass(...)`.

# %%
repf = rep.meca.low_pass(order=2, cutoff=CUTOFF, freq=sample_rate)
heel_right_filt = np.array(repf.sel(channel="pRightHeelFoot", axis=["x", "y", "z"])) / 1000

heel_vel = np.diff(heel_right, axis=1) * sample_rate
heel_vel_filt = np.diff(heel_right_filt, axis=1) * sample_rate
heel_acc = np.diff(heel_vel, axis=1) * sample_rate
heel_acc_filt = np.diff(heel_vel_filt, axis=1) * sample_rate

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
ax[0].plot(time, heel_right[2, :], label="données brutes", c="blue")
ax[0].plot(time, heel_right_filt[2, :], label="données filtrées", c="orange")
ax[0].set_ylabel("Position Z (m)")
ax[0].legend()
ax[1].plot(time[1:], heel_vel[2, :], c="blue")
ax[1].plot(time[1:], heel_vel_filt[2, :], c="orange")
ax[1].set_ylabel("Vitesse Z (m/s)")
ax[2].plot(time[2:], heel_acc[2, :], c="blue")
ax[2].plot(time[2:], heel_acc_filt[2, :], c="orange")
ax[2].set_ylabel("Accélération Z (m/s²)")
ax[2].set_xlabel("Temps (s)")
fig.suptitle("pRightHeelFoot — brut vs filtré (cutoff = 10 Hz)")
plt.show()

# %%
# ### Exercice 4.3 — quantifier l'effet du filtrage
#
# Calculez l'écart-type de l'accélération verticale brute et filtrée, et le
# pourcentage de réduction. Le résultat vous semble-t-il cohérent avec la
# démonstration simulée du début (indice : la propreté du signal d'origine) ?

# %%
std_brut = np.std(heel_acc[2, :])
std_filt = np.std(heel_acc_filt[2, :])
print("Écart-type accélération brute  :", round(std_brut, 2))
print("Écart-type accélération filtrée:", round(std_filt, 2))
print("Réduction :", round((1 - std_filt / std_brut) * 100, 1), "%")
print("-> l'effet est ici plus modeste que sur la démonstration simulée : ce jeu")
print("   de données est déjà assez propre (peu de bruit haute fréquence).")
print("   Le filtrage reste néanmoins une bonne pratique systématique avant de")
print("   dériver un signal (vitesse, accélération), en particulier sur des")
print("   données réelles moins propres que celles-ci.")

# %%
# ### Pour la suite : une fonction réutilisable
#
# On regroupe chargement + filtrage dans une fonction `charger_et_filtrer`,
# réutilisée dans toutes les parties suivantes (code fourni).

# %%
def charger_et_filtrer(file_path, cutoff=CUTOFF):
    """Charge un essai .c3d et renvoie l'objet pyomeca filtré + la fréquence."""
    rep_ = Markers.from_c3d(file_path)
    freq_ = rep_.attrs["rate"]
    repf_ = rep_.meca.low_pass(order=2, cutoff=cutoff, freq=freq_)
    return repf_, freq_

# %%
# ## Partie 5 — Cycles et normalisation temporelle
#
# Contrairement à un mouvement discret (un seul saut, segmenté par un seuil de
# vitesse), les mouvements étudiés ici sont **cycliques** (course, pédalage) :
# on ne compare pas un événement isolé, mais un **cycle moyen** obtenu par
# normalisation temporelle — la méthode standard en analyse du mouvement
# cyclique. On réutilise `charger_et_filtrer` (Partie 4) sur Carbone12.

# %%
repf, freq_samp = charger_et_filtrer(os.path.join(SUJET1_PATH, "Carbone12.c3d"))
time = np.array(repf.time)

# %%
# ### Exercice 5.1 — fonction générique de calcul d'angle articulaire
#
# Angle (en degrés) entre le segment p1->p2 et le segment p3->p4, où p1, p2,
# p3, p4 sont des tableaux (3, n_frames). Cette fonction sera réutilisée dans
# toutes les parties suivantes.

# %%
def calculer_angle(p1, p2, p3, p4):
    """Angle (en degrés) entre le segment p1->p2 et le segment p3->p4.

    p1, p2, p3, p4 : tableaux (3, n_frames)
    """
    vecteur1 = p2 - p1
    vecteur2 = p4 - p3
    vecteur1_norm = vecteur1 / np.linalg.norm(vecteur1, axis=0)
    vecteur2_norm = vecteur2 / np.linalg.norm(vecteur2, axis=0)
    produit_scalaire = np.sum(vecteur1_norm * vecteur2_norm, axis=0)
    produit_scalaire = np.clip(produit_scalaire, -1.0, 1.0)  # sécurité arrondis
    return np.degrees(np.arccos(produit_scalaire))

# %%
# ### Exercice 5.2 — détection automatique des cycles
#
# Utilisez la trajectoire verticale (Z) du talon droit, filtrée, et
# `scipy.signal.find_peaks()` pour détecter chaque maximum local (un maximum =
# un même instant du cycle, répété à chaque cycle). Astuce : imposez un écart
# minimal entre 2 pics (paramètre `distance`, en nombre de frames) pour éviter
# de détecter du bruit comme un cycle — 0.4 s convertis en frames est un bon
# point de départ. On formalise ça en une fonction `detecter_cycles`,
# réutilisée dans les parties suivantes.

# %%
def detecter_cycles(heel_z, freq, min_period_s=0.4):
    """Détecte les cycles à partir d'une trajectoire verticale (1D, filtrée)
    et renvoie les indices des pics (= 1 instant par cycle)."""
    distance_min = int(min_period_s * freq)
    peaks, _ = find_peaks(heel_z, distance=distance_min)
    return peaks

heel_z = np.array(repf.sel(channel="pRightHeelFoot", axis=["z"])).flatten()
peaks = detecter_cycles(heel_z, freq_samp)

print("Nombre de cycles détectés :", len(peaks) - 1)

periodes = np.diff(peaks) / freq_samp
print("Durée moyenne d'un cycle :", round(np.mean(periodes), 3), "s",
      "(cadence :", round(60 / np.mean(periodes), 1), "cycles/min)")
print("Coefficient de variation de la durée de cycle :",
      round(np.std(periodes) / np.mean(periodes) * 100, 1), "%")

plt.figure()
plt.plot(time, heel_z, label="pRightHeelFoot (Z, filtré)")
plt.plot(time[peaks], heel_z[peaks], "rx", label="cycles détectés")
plt.xlabel("Temps (s)")
plt.ylabel("Position verticale (mm)")
plt.legend()
plt.title("Détection des cycles sur la trajectoire verticale du talon droit")
plt.show()

# %%
# ### Exercice 5.3 — angle du genou droit au cours de l'essai
#
# hanche (`pRightGreaterTrochanter`) -> genou -> cheville (genou et cheville :
# moyenne des 2 marqueurs, comme en Partie 2).

# %%
hip = np.array(repf.sel(channel="pRightGreaterTrochanter", axis=["x", "y", "z"]))
knee = (np.array(repf.sel(channel="pRightKneeLatEpicondyle", axis=["x", "y", "z"]))
        + np.array(repf.sel(channel="pRightKneeMedEpicondyle", axis=["x", "y", "z"]))) / 2
ankle = (np.array(repf.sel(channel="pRightLatMalleolus", axis=["x", "y", "z"]))
         + np.array(repf.sel(channel="pRightMedMalleolus", axis=["x", "y", "z"]))) / 2

angle_genou = calculer_angle(hip, knee, knee, ankle)

plt.figure()
plt.plot(time, angle_genou)
plt.xlabel("Temps (s)")
plt.ylabel("Angle du genou droit (°)")
plt.title("Évolution de l'angle du genou droit sur l'essai complet")
plt.show()

# %%
# ### Exercice 5.4 — normalisation temporelle des cycles (0-100 %)
#
# Chaque cycle a une durée différente : on interpole chacun sur une base de
# 101 points (0 à 100 %) pour pouvoir les moyenner entre eux (`np.interp`).
# Cette fonction sera réutilisée dans toutes les parties suivantes.

# %%
def normaliser_cycles(signal, peaks, n_points=101):
    """Découpe `signal` entre pics successifs et renvoie une matrice
    (n_cycles, n_points) de cycles normalisés à 0-100 %."""
    cycles = []
    for i in range(len(peaks) - 1):
        segment = signal[peaks[i]:peaks[i + 1]]
        if len(segment) < 5:
            continue
        x_old = np.linspace(0, 100, len(segment))
        x_new = np.linspace(0, 100, n_points)
        cycles.append(np.interp(x_new, x_old, segment))
    return np.array(cycles)


cycles_genou = normaliser_cycles(angle_genou, peaks)
base_temps = np.linspace(0, 100, cycles_genou.shape[1])
moyenne_genou = cycles_genou.mean(axis=0)
ecart_type_genou = cycles_genou.std(axis=0)

plt.figure()
plt.plot(base_temps, moyenne_genou, label="cycle moyen", color="blue")
plt.fill_between(base_temps, moyenne_genou - ecart_type_genou,
                  moyenne_genou + ecart_type_genou, alpha=0.2, color="blue",
                  label="± écart-type")
plt.xlabel("Cycle (%)")
plt.ylabel("Angle du genou droit (°)")
plt.title(f"Cycle moyen du genou droit (n={cycles_genou.shape[0]} cycles) — Carbone12")
plt.legend()
plt.show()

print("Amplitude articulaire moyenne (ROM) du genou :",
      round(moyenne_genou.max() - moyenne_genou.min(), 1), "°")

# %%
# ### Exercice 5.5 — comparer 2 conditions (Carbone12 vs Mousse12)
#
# Rassemblez les étapes précédentes (chargement+filtrage, angle du genou,
# détection de cycles, normalisation) dans une fonction unique, puis
# appliquez-la aux 2 fichiers à comparer.

# %%
def cycle_moyen_angle_genou(file_path):
    """Pipeline complet : charge, filtre, calcule l'angle du genou, détecte
    les cycles et renvoie le cycle moyen ± écart-type."""
    repf_local, freq_local = charger_et_filtrer(file_path)
    heel_z_local = np.array(repf_local.sel(channel="pRightHeelFoot", axis=["z"])).flatten()
    peaks_local = detecter_cycles(heel_z_local, freq_local)

    hip_local = np.array(repf_local.sel(channel="pRightGreaterTrochanter", axis=["x", "y", "z"]))
    knee_local = (np.array(repf_local.sel(channel="pRightKneeLatEpicondyle", axis=["x", "y", "z"]))
                  + np.array(repf_local.sel(channel="pRightKneeMedEpicondyle", axis=["x", "y", "z"]))) / 2
    ankle_local = (np.array(repf_local.sel(channel="pRightLatMalleolus", axis=["x", "y", "z"]))
                   + np.array(repf_local.sel(channel="pRightMedMalleolus", axis=["x", "y", "z"]))) / 2
    angle_genou_local = calculer_angle(hip_local, knee_local, knee_local, ankle_local)

    cycles_local = normaliser_cycles(angle_genou_local, peaks_local)
    return cycles_local.mean(axis=0), cycles_local.std(axis=0)


moyenne_carbone, std_carbone = cycle_moyen_angle_genou(
    os.path.join(SUJET1_PATH, "Carbone12.c3d"))
moyenne_mousse, std_mousse = cycle_moyen_angle_genou(
    os.path.join(SUJET1_PATH, "Mousse12.c3d"))

base_temps = np.linspace(0, 100, len(moyenne_carbone))

plt.figure()
plt.plot(base_temps, moyenne_carbone, label="Carbone12", color="black")
plt.fill_between(base_temps, moyenne_carbone - std_carbone, moyenne_carbone + std_carbone,
                  alpha=0.2, color="black")
plt.plot(base_temps, moyenne_mousse, label="Mousse12", color="green")
plt.fill_between(base_temps, moyenne_mousse - std_mousse, moyenne_mousse + std_mousse,
                  alpha=0.2, color="green")
plt.xlabel("Cycle (%)")
plt.ylabel("Angle du genou droit (°)")
plt.title("Comparaison Carbone vs Mousse (Sujet1, niveau 12)")
plt.legend()
plt.show()

print("ROM genou Carbone12 :", round(moyenne_carbone.max() - moyenne_carbone.min(), 1), "°")
print("ROM genou Mousse12  :", round(moyenne_mousse.max() - moyenne_mousse.min(), 1), "°")

# %%
# ## Partie 6 — Automatisation avec des boucles
#
# Objectif : comprendre et utiliser les boucles `for` pour automatiser des
# opérations répétitives, des boucles simples aux boucles imbriquées, et
# automatiser le chargement de plusieurs fichiers `.c3d` — en réutilisant les
# fonctions des parties précédentes.

# %%
# ### Rappel — boucle for simple (code fourni)

# %%
for i in range(0, 5):
    print(f"Le carré de {i} est {i**2}")

# %%
# ### Exercice 6.1 — lister les fichiers d'un sujet avec une boucle

# %%
sujet1_path = SUJET1_PATH

fichiers_sujet1 = []
for f in os.listdir(sujet1_path):
    if f.endswith(".c3d"):
        fichiers_sujet1.append(f)
fichiers_sujet1 = sorted(fichiers_sujet1)

for fichier in fichiers_sujet1:
    print(f"Traitement du fichier : {fichier}")

# %%
# ### Exercice 6.2 — boucle imbriquée : toutes les combinaisons condition x niveau
#
# Complétez le dictionnaire `sujets_config` pour le Sujet2 (conditions
# `"Chrono"`/`"Route"`, niveaux `"250"`/`"300"`/`"350"`), puis écrivez la
# triple boucle imbriquée (sujet, condition, niveau) qui affiche, pour CHAQUE
# combinaison, le sujet et le nom de fichier correspondant. Résultat attendu :
# 2 sujets x 2 conditions x 3 niveaux = 12 lignes.

# %%
sujets_config = {
    "Sujet1": {"conditions": ["Carbone", "Mousse"], "niveaux": ["12", "14", "16"]},
    "Sujet2": {"conditions": ["Chrono", "Route"], "niveaux": ["250", "300", "350"]},
}

for sujet, config in sujets_config.items():
    for condition in config["conditions"]:
        for niveau in config["niveaux"]:
            nom_fichier = f"{condition}{niveau}.c3d"
            print(sujet, nom_fichier)

# %%
# ### Exercice 6.3 — charger automatiquement chaque fichier et afficher sa durée
#
# On réutilise `charger_et_filtrer` (Partie 4).

# %%
for fichier in fichiers_sujet1:
    file_path = os.path.join(sujet1_path, fichier)
    repf, freq = charger_et_filtrer(file_path)

    num_frames = len(repf.time)
    duree = num_frames / freq

    print(f"{fichier:15s} -> durée {round(duree, 2)} s")

# %%
# ### Exercice 6.4 — automatiser un calcul : cadence de chaque essai (Sujet1)
#
# Combinez boucle + calcul (repris de la Partie 5, via `detecter_cycles`) pour
# obtenir la cadence de chaque essai du Sujet1, et stockez les résultats dans
# une liste de dictionnaires.

# %%
resultats_cadence = []

for fichier in fichiers_sujet1:
    file_path = os.path.join(sujet1_path, fichier)
    repf, freq = charger_et_filtrer(file_path)

    heel_z = np.array(repf.sel(channel="pRightHeelFoot", axis=["z"])).flatten()
    peaks = detecter_cycles(heel_z, freq)
    periodes = np.diff(peaks) / freq
    cadence = 60 / np.mean(periodes)

    resultats_cadence.append({"fichier": fichier, "cadence_cyc_min": round(float(cadence), 1)})
    print(f"{fichier:15s} -> cadence {round(float(cadence), 1)} cycles/min "
          f"({len(peaks) - 1} cycles détectés)")

print("\nRésumé :", resultats_cadence)

# %%
# ## Partie 7 — Pipeline complet (bonus)
#
# Objectif : rassembler tout ce qui précède (`charger_et_filtrer`,
# `calculer_angle`, `detecter_cycles`, `normaliser_cycles`) dans un pipeline
# complet, traitant automatiquement l'ensemble des essais des 2 sujets, avec
# un DataFrame Pandas et des graphiques Seaborn.
#
# Cette partie va plus loin que les précédentes sur 2 points :
#
# - **Asymétrie droite/gauche** : les centres articulaires et angles sont
#   calculés des 2 côtés, puis un indice de symétrie `SI_<articulation>_pct`
#   est calculé par essai (0 = parfaitement symétrique).
# - **Coordination inter-articulaire** : différence de phase entre 2
#   articulations (genou-hanche, cheville-genou, épaule-hanche), calculée sur
#   le cycle normalisé (0-100 %).
#
# Partie optionnelle : donne une base de travail pour une analyse plus
# poussée (mémoire, projet de recherche).

# %%
# pandas et seaborn ont deja ete importes en Partie 1 (import os, numpy,
# pandas, matplotlib, seaborn, scipy.signal, pyomeca — voir tout en haut du
# notebook).
SUJETS_CONFIG = {
    "Sujet1": {"path": SUJET1_PATH, "conditions": ["Carbone", "Mousse"], "niveaux": ["12", "14", "16"]},
    "Sujet2": {"path": SUJET2_PATH, "conditions": ["Chrono", "Route"], "niveaux": ["250", "300", "350"]},
}

# %%
# ### Exercice 7.1 — dictionnaire de centres articulaires (2 côtés)
#
# On réutilise `charger_et_filtrer` (Partie 4). Complétez le dictionnaire de
# centres articulaires, **pour les 2 côtés** (nécessaire à l'analyse
# d'asymétrie D/G plus loin). Marqueurs utiles :
# - tronc haut : `pC7SpinalProcess`, `pIJ`
# - tronc milieu : `pPX`, `pT8SpinalProcess`
# - bassin : `pRightASI`, `pLeftASI`, `pRightCSI`, `pLeftCSI` (moyenne des 4)
# - épaule droite/gauche : `pRightAcromion` / `pLeftAcromion`
# - coude droit/gauche : `pRightArmLatEpicondyle`+`pRightArmMedEpicondyle` /
#   `pLeftArmLatEpicondyle`+`pLeftArmMedEpicondyle`
# - genou droit/gauche : `pRightKneeLatEpicondyle`+`pRightKneeMedEpicondyle` /
#   `pLeftKneeLatEpicondyle`+`pLeftKneeMedEpicondyle`
# - cheville droite/gauche : `pRightLatMalleolus`+`pRightMedMalleolus` /
#   `pLeftLatMalleolus`+`pLeftMedMalleolus`
# - avant-pied droit/gauche : `pRightFirstMetatarsal`+`pRightFifthMetatarsal` /
#   `pLeftFirstMetatarsal`+`pLeftFifthMetatarsal`
# - talon droit/gauche : `pRightHeelFoot` / `pLeftHeelFoot`

# %%
def collect_and_clean_markers(file_path):
    """Charge+filtre un essai (via charger_et_filtrer) et renvoie un
    dictionnaire de centres articulaires, la fréquence et le vecteur temps."""
    repf, freq = charger_et_filtrer(file_path)
    time = np.array(repf.time)

    def m(name):
        return np.array(repf.sel(channel=name, axis=["x", "y", "z"]))

    def mid(n1, n2):
        return (m(n1) + m(n2)) / 2

    markers = {
        "trunkTop": mid("pC7SpinalProcess", "pIJ"),
        "trunkMid": mid("pPX", "pT8SpinalProcess"),
        "trunkBase": (m("pRightASI") + m("pLeftASI") + m("pRightCSI") + m("pLeftCSI")) / 4,
        "shoulder_right": m("pRightAcromion"),
        "elbow_right": mid("pRightArmLatEpicondyle", "pRightArmMedEpicondyle"),
        "knee_right": mid("pRightKneeLatEpicondyle", "pRightKneeMedEpicondyle"),
        "ankle_right": mid("pRightLatMalleolus", "pRightMedMalleolus"),
        "foot_right": (m("pRightHeelFoot") + m("pRightFirstMetatarsal") + m("pRightFifthMetatarsal")) / 3,
        "heel_right": m("pRightHeelFoot"),
        "shoulder_left": m("pLeftAcromion"),
        "elbow_left": mid("pLeftArmLatEpicondyle", "pLeftArmMedEpicondyle"),
        "knee_left": mid("pLeftKneeLatEpicondyle", "pLeftKneeMedEpicondyle"),
        "ankle_left": mid("pLeftLatMalleolus", "pLeftMedMalleolus"),
        "foot_left": (m("pLeftHeelFoot") + m("pLeftFirstMetatarsal") + m("pLeftFifthMetatarsal")) / 3,
        "heel_left": m("pLeftHeelFoot"),
    }
    return markers, freq, time

# %%
# ### Exercice 7.2 — tous les angles articulaires (2 côtés)
#
# On réutilise `calculer_angle` (Partie 5). Pour chaque articulation, l'angle
# est calculé entre le segment "proximal" et le segment "distal" (ex : genou =
# segment bassin->genou vs segment genou->cheville).

# %%
def calculate_all_joint_angles(markers):
    angles = {}
    # côté droit
    angles["hip_right"] = calculer_angle(
        markers["trunkMid"], markers["trunkBase"], markers["trunkBase"], markers["knee_right"])
    angles["knee_right"] = calculer_angle(
        markers["trunkBase"], markers["knee_right"], markers["knee_right"], markers["ankle_right"])
    angles["ankle_right"] = calculer_angle(
        markers["knee_right"], markers["ankle_right"], markers["ankle_right"], markers["foot_right"])
    angles["shoulder_right"] = calculer_angle(
        markers["trunkTop"], markers["shoulder_right"], markers["shoulder_right"], markers["elbow_right"])
    # côté gauche (même construction, marqueurs gauches) : sert de base à
    # l'analyse d'asymétrie droite/gauche plus bas
    angles["hip_left"] = calculer_angle(
        markers["trunkMid"], markers["trunkBase"], markers["trunkBase"], markers["knee_left"])
    angles["knee_left"] = calculer_angle(
        markers["trunkBase"], markers["knee_left"], markers["knee_left"], markers["ankle_left"])
    angles["ankle_left"] = calculer_angle(
        markers["knee_left"], markers["ankle_left"], markers["ankle_left"], markers["foot_left"])
    angles["shoulder_left"] = calculer_angle(
        markers["trunkTop"], markers["shoulder_left"], markers["shoulder_left"], markers["elbow_left"])
    return angles

# %%
# ### Exercice 7.3 — différence de phase (coordination)
#
# Complétez `calculer_difference_phase`, utilisée plus loin pour l'analyse de
# coordination inter-articulaire.

# %%
def calculer_difference_phase(angle1, angle2):
    """Différence point par point entre 2 courbes angulaires (même base de
    temps, ex : 2 angles sur le même cycle normalisé 0-100 %). Renseigne sur
    la coordination entre 2 articulations."""
    return angle2 - angle1

# %%
# ### Exercice 7.4 — pipeline complet sur un seul fichier -> une ligne de résultats
#
# À partir des cycles normalisés de chaque angle (via `normaliser_cycles`,
# Partie 5), calculez son ROM (amplitude : max - min de la courbe moyenne) et
# son angle maximal moyen, et ajoutez-les au dictionnaire `resultat` avec des
# clés `f"ROM_{{nom_angle}}"` et `f"angle_max_{{nom_angle}}"`.

# %%
def analyser_essai(sujet, condition, niveau, file_path):
    markers, freq, time = collect_and_clean_markers(file_path)
    angles = calculate_all_joint_angles(markers)
    peaks = detecter_cycles(markers["heel_right"][2, :], freq)

    periodes = np.diff(peaks) / freq
    cadence = 60 / np.mean(periodes)
    cv_periode = np.std(periodes) / np.mean(periodes) * 100
    duree_s = len(time) / freq

    resultat = {
        "sujet": sujet,
        "condition": condition,
        "niveau": niveau,
        "fichier": os.path.basename(file_path),
        "duree_s": round(duree_s, 2),
        "n_cycles": len(peaks) - 1,
        "cadence_cyc_min": round(float(cadence), 1),
        "cv_periode_pct": round(float(cv_periode), 1),
    }

    courbes_normalisees = {}
    for nom_angle, valeurs in angles.items():
        cycles = normaliser_cycles(valeurs, peaks)
        moyenne = cycles.mean(axis=0)
        resultat[f"ROM_{nom_angle}"] = round(float(moyenne.max() - moyenne.min()), 1)
        resultat[f"angle_max_{nom_angle}"] = round(float(moyenne.max()), 1)
        courbes_normalisees[nom_angle] = moyenne

    # --- Asymétrie droite / gauche : indice de symétrie (SI, en %) ---
    for articulation in ["hip", "knee", "ankle", "shoulder"]:
        rom_d = resultat[f"ROM_{articulation}_right"]
        rom_g = resultat[f"ROM_{articulation}_left"]
        si = abs(rom_d - rom_g) / (0.5 * (rom_d + rom_g)) * 100
        resultat[f"SI_{articulation}_pct"] = round(float(si), 1)

    # --- Coordination inter-articulaire (différence de phase, côté droit) ---
    diff_genou_hanche = calculer_difference_phase(
        courbes_normalisees["knee_right"], courbes_normalisees["hip_right"])
    diff_cheville_genou = calculer_difference_phase(
        courbes_normalisees["ankle_right"], courbes_normalisees["knee_right"])
    diff_epaule_hanche = calculer_difference_phase(
        courbes_normalisees["shoulder_right"], courbes_normalisees["hip_right"])

    resultat["coord_genou_hanche_moy"] = round(float(np.mean(diff_genou_hanche)), 1)
    resultat["coord_cheville_genou_moy"] = round(float(np.mean(diff_cheville_genou)), 1)
    resultat["coord_epaule_hanche_moy"] = round(float(np.mean(diff_epaule_hanche)), 1)

    courbes_normalisees["phase_genou_hanche"] = diff_genou_hanche
    courbes_normalisees["phase_cheville_genou"] = diff_cheville_genou
    courbes_normalisees["phase_epaule_hanche"] = diff_epaule_hanche

    return resultat, courbes_normalisees

# %%
# ### Exercice 7.5 — boucle principale : traiter tous les essais des 2 sujets
#
# Complétez la triple boucle (comme en Partie 6) : pour chaque sujet, chaque
# condition, chaque niveau, construisez le chemin du fichier, vérifiez qu'il
# existe (`os.path.exists`), puis appelez `analyser_essai()` et stockez les
# résultats.

# %%
resultats = []
courbes_genou = {}  # pour la figure "petits multiples" plus bas
toutes_courbes = {}  # dictionnaire complet, pour les figures D/G et coordination

for sujet, config in SUJETS_CONFIG.items():
    for condition in config["conditions"]:
        for niveau in config["niveaux"]:
            fichier = f"{condition}{niveau}.c3d"
            file_path = os.path.join(config["path"], fichier)

            if os.path.exists(file_path):
                resultat, courbes = analyser_essai(sujet, condition, niveau, file_path)
                resultats.append(resultat)
                courbes_genou[(sujet, condition, niveau)] = courbes["knee_right"]
                toutes_courbes[(sujet, condition, niveau)] = courbes
            else:
                print(f"Fichier introuvable : {file_path}")

df = pd.DataFrame(resultats)
print(df)

# %%
# ### Exercice 7.6 — statistiques descriptives par sujet x condition
#
# Pour chaque sujet, calculez moyenne/écart-type/min/max du ROM du genou
# (colonne `"ROM_knee_right"`) groupé par condition (`df.groupby`).

# %%
for sujet in SUJETS_CONFIG:
    print(f"\n--- {sujet} : ROM genou (°) par condition ---")
    print(df[df["sujet"] == sujet].groupby("condition")["ROM_knee_right"]
          .agg(["mean", "std", "min", "max"]).round(1))

# %%
# ### Visualisation — comparaison des conditions (boxplots, code fourni)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, sujet in zip(axes, SUJETS_CONFIG):
    df_sujet = df[df["sujet"] == sujet]
    sns.boxplot(x="condition", y="ROM_knee_right", data=df_sujet, ax=ax)
    sns.stripplot(x="condition", y="ROM_knee_right", data=df_sujet, ax=ax,
                  color="black", size=8)
    ax.set_title(sujet)
    ax.set_ylabel("Amplitude articulaire genou droit (°)")
    ax.set_xlabel("Condition")

fig.suptitle("Amplitude du genou droit selon la condition (tous niveaux confondus)",
             fontsize=14)
plt.tight_layout()
plt.show()

# %%
# ### Visualisation — effet du niveau sur la cadence (code fourni)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, sujet in zip(axes, SUJETS_CONFIG):
    df_sujet = df[df["sujet"] == sujet]
    sns.lineplot(x="niveau", y="cadence_cyc_min", hue="condition", data=df_sujet,
                 marker="o", ax=ax)
    ax.set_title(sujet)
    ax.set_ylabel("Cadence (cycles/min)")
    ax.set_xlabel("Niveau")

fig.suptitle("Cadence en fonction du niveau et de la condition", fontsize=14)
plt.tight_layout()
plt.show()

# %%
# ### Visualisation — petits multiples : cycle moyen du genou (code fourni)

# %%
for sujet, config in SUJETS_CONFIG.items():
    fig, axs = plt.subplots(1, len(config["niveaux"]), figsize=(15, 4), sharey=True)
    base_temps = np.linspace(0, 100, 101)

    for j, niveau in enumerate(config["niveaux"]):
        for condition, couleur in zip(config["conditions"], ["black", "green"]):
            cle = (sujet, condition, niveau)
            if cle in courbes_genou:
                axs[j].plot(base_temps, courbes_genou[cle], label=condition, color=couleur)
        axs[j].set_title(f"Niveau {niveau}")
        axs[j].set_xlabel("Cycle (%)")

    axs[0].set_ylabel("Angle du genou droit (°)")
    axs[0].legend()
    fig.suptitle(f"{sujet} — cycle moyen du genou par condition et par niveau", fontsize=14)
    plt.tight_layout()
    plt.show()

# %%
# ### Exercice 7.7 — asymétrie droite/gauche : indice de symétrie (SI)
#
# `SI_knee_pct` (par ex.) = 0 si le genou droit et le genou gauche ont
# exactement la même amplitude articulaire sur cet essai ; plus il est grand,
# plus l'asymétrie D/G est marquée. Utile par ex. pour repérer une éventuelle
# compensation liée à une condition (semelle, position vélo). Affichez la
# moyenne des colonnes `SI_*_pct` groupées par sujet x condition.

# %%
print("\n--- Indice de symétrie (SI, %) — moyenne par sujet x condition ---")
colonnes_si = []
for c in df.columns:
    if c.startswith("SI_"):
        colonnes_si.append(c)
print(df.groupby(["sujet", "condition"])[colonnes_si].mean().round(1))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, sujet in zip(axes, SUJETS_CONFIG):
    df_sujet = df[df["sujet"] == sujet]
    sns.boxplot(x="condition", y="SI_knee_pct", data=df_sujet, ax=ax)
    sns.stripplot(x="condition", y="SI_knee_pct", data=df_sujet, ax=ax, color="black", size=8)
    ax.set_title(sujet)
    ax.set_ylabel("Indice de symétrie du genou (%)")
    ax.set_xlabel("Condition")
fig.suptitle("Asymétrie D/G du genou selon la condition (tous niveaux confondus)",
             fontsize=14)
plt.tight_layout()
plt.show()

# %%
# ### Exercice 7.8 — asymétrie D/G : comparaison visuelle des 2 côtés
#
# Tracez le cycle moyen du genou droit ET du genou gauche sur un même
# graphique, pour l'essai Sujet1/Carbone12 (dans `toutes_courbes`).

# %%
sujet_ex, condition_ex, niveau_ex = "Sujet1", "Carbone", "12"
courbes_ex = toutes_courbes[(sujet_ex, condition_ex, niveau_ex)]
base_temps = np.linspace(0, 100, 101)

plt.figure()
plt.plot(base_temps, courbes_ex["knee_right"], label="genou droit", color="blue")
plt.plot(base_temps, courbes_ex["knee_left"], label="genou gauche", color="red")
plt.xlabel("Cycle (%)")
plt.ylabel("Angle du genou (°)")
plt.title(f"Genou droit vs gauche — {sujet_ex}/{condition_ex}{niveau_ex}")
plt.legend()
plt.show()

# %%
# ### Exercice 7.9 — coordination inter-articulaire (différence de phase)
#
# Une différence de phase qui varie peu traduit une coordination stable entre
# 2 articulations au cours du cycle. Affichez la moyenne de
# `coord_genou_hanche_moy` groupée par sujet x condition.

# %%
print("\n--- Coordination genou-hanche (différence angulaire moyenne, °) ---")
print(df.groupby(["sujet", "condition"])["coord_genou_hanche_moy"].mean().round(1))

plt.figure()
plt.plot(base_temps, courbes_ex["phase_genou_hanche"], label="genou - hanche", color="purple")
plt.plot(base_temps, courbes_ex["phase_cheville_genou"], label="cheville - genou", color="teal")
plt.plot(base_temps, courbes_ex["phase_epaule_hanche"], label="épaule - hanche", color="brown")
plt.xlabel("Cycle (%)")
plt.ylabel("Différence angulaire (°)")
plt.title(f"Coordination inter-articulaire — {sujet_ex}/{condition_ex}{niveau_ex}")
plt.legend()
plt.show()