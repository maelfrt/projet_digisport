# -*- coding: utf-8 -*-

# %%
# # Projet — Angle du genou droit lors d'une frappe de ballon
#
# Correction du script basé sur le TD "changement de repère" :
# - repere_cuisse recevait "mollet" (marqueur de jambe) au lieu de "cuisse"
#   pour construire l'axe x de la CUISSE -> corrigé.
# - repere_jambe mélangeait cuisse et cheville pour définir l'origine et
#   l'axe z de la JAMBE -> corrigé en s'appuyant sur les marqueurs
#   réellement disponibles (pas de paire ext/int à la cheville ici).
# - Ajout d'une vérification des données manquantes (NaN), fréquentes sur
#   une vraie capture (occlusions de marqueurs), qui provoquaient le crash
#   "SVD did not converge" en propageant des NaN dans les matrices de repère.

# %%
import numpy as np
import matplotlib.pyplot as plt
from pyomeca import Markers
from scipy.spatial.transform import Rotation

marqueurs_rom = Markers.from_c3d("projet_digisport/data2/ANAT_REF.c3d")
marqueurs = Markers.from_c3d("projet_digisport/data2/CAPTURE0002_ballon_v5.c3d")
temps = np.arange(marqueurs.shape[2]) / marqueurs.attrs["rate"]
print(marqueurs.channel.values.tolist())


def position(essai, nom):
    return essai.sel(channel=nom).values[0:3, :]  # (3, n_frames), en mm


def normaliser(v):
    norme = np.linalg.norm(v)
    if norme < 1e-8:
        # évite une division par ~0 (vecteurs quasi colinéaires) qui
        # produirait des NaN et ferait planter Rotation.from_matrix ensuite
        raise ValueError("Vecteur quasi nul : marqueurs colinéaires ou données manquantes sur cette frame.")
    return v / norme


# %%
# ## Étape 0 — Vérifier l'absence de données manquantes (NaN)
#
# Sur une vraie capture (contrairement aux fichiers "propres" ROM_S1/REEL_0008
# du TD), il arrive qu'un marqueur soit occulté sur certaines frames : les
# coordonnées valent alors NaN. Une seule frame avec un NaN suffit à propager
# des NaN dans toute la matrice de rotation -> crash "SVD did not converge".

marqueurs_genou_droit = [
    "genou exterieur droit", "genou interieur droite",
    "cuisse droite", "mollet droite", "cheville droite",
]

def compter_nan(essai, noms):
    for nom in noms:
        p = position(essai, nom)
        n_nan = np.isnan(p).any(axis=0).sum()
        if n_nan > 0:
            print(f"  ATTENTION : {n_nan} frame(s) avec NaN pour '{nom}'")

print("Vérification NaN - capture :")
compter_nan(marqueurs, marqueurs_genou_droit)
print("Vérification NaN - référence :")
compter_nan(marqueurs_rom, marqueurs_genou_droit)

# %%
# ## Étape 1 — Repère de la cuisse droite
#
# - origine : centre du genou = milieu de genou_exterieur / genou_interieur
# - z (médio-latéral) : genou_interieur - genou_exterieur, normalisé
# - x (avant) : (cuisse - origine) x z   <- le marqueur "cuisse" sert de
#   point de référence non colinéaire, exactement comme LTHI dans le TD
# - y : z x x

def repere_cuisse(cuisse, gnext, gnint):
    origine = (gnext + gnint) / 2
    z = normaliser(gnint - gnext)
    x = normaliser(np.cross(cuisse - origine, z))
    return np.column_stack((x, np.cross(z, x), z))


# %%
# ## Étape 2 — Repère de la jambe droite
#
# Le TD construit le repère de la jambe à partir d'une paire de marqueurs à
# la cheville (ANE/ANI), symétrique de la paire au genou. Ici, il n'y a
# qu'un seul marqueur de cheville ("cheville droite") : on ne peut donc pas
# calculer un axe médio-latéral propre à la cheville.
#
# Adaptation retenue :
# - origine : le marqueur de cheville directement (pas de moyenne possible)
# - z : on réutilise l'axe médio-latéral du genou (faute de paire à la
#   cheville) — c'est une approximation nécessaire avec ce jeu de marqueurs
# - x : (mollet - origine) x z   <- "mollet" joue ici le même rôle que
#   "cuisse" pour le repère proximal : un marqueur du segment lui-même

def repere_jambe(gnext, gnint, mollet, cheville):
    origine = cheville
    z = normaliser(gnint - gnext)
    x = normaliser(np.cross(mollet - origine, z))
    return np.column_stack((x, np.cross(z, x), z))


# %%
# ## Étape 3 — Angle du genou droit
#
# Même principe que le TD : R_rel = R_cuisse^T . R_jambe, décomposé en
# séquence de Cardan Z-X-Y (Grood & Suntay 1983).
# Pour le genou DROIT (contrairement au genou gauche du TD), on ne met PAS
# de signe (-1) : la convention de signe dépend de l'orientation de vos
# axes locaux. Si vos courbes de flexion sortent inversées par rapport à
# ce qui est physiologiquement attendu (flexion positive pendant l'armé de
# la frappe), ajoutez le signe (-1) comme dans le TD.

def angles_genou_droit(essai):
    cuisse, gnext, gnint, mollet, cheville = [
        position(essai, nom) for nom in
        ["cuisse droite", "genou exterieur droit", "genou interieur droite",
         "mollet droite", "cheville droite"]
    ]
    n_frames = cuisse.shape[1]
    angles = np.full((3, n_frames), np.nan)
    for i in range(n_frames):
        try:
            R_cuisse = repere_cuisse(cuisse[:, i], gnext[:, i], gnint[:, i])
            R_jambe = repere_jambe(gnext[:, i], gnint[:, i], mollet[:, i], cheville[:, i])
            R_rel = R_cuisse.T @ R_jambe
            angles[:, i] = Rotation.from_matrix(R_rel).as_euler("ZXY", degrees=True)
        except ValueError as e:
            # frame avec marqueur manquant / configuration dégénérée : on
            # laisse un NaN plutôt que de faire planter tout le script
            print(f"  frame {i} ignorée : {e}")
    return angles


angles = angles_genou_droit(marqueurs)

# %%
# ## Étape 4 — Calibration avec ANAT_REF
#
# Comme dans le TD (essai ROM), on calcule les angles sur l'essai de
# référence et on prend l'instant où le sujet bouge le moins, en se basant
# sur la vitesse du marqueur "cheville droite" (disponible ici).

angles_ref_tous = angles_genou_droit(marqueurs_rom)

cheville_ref = position(marqueurs_rom, "cheville droite")
vitesse = np.convolve(
    np.linalg.norm(np.diff(cheville_ref, axis=1), axis=0),
    np.ones(50) / 50, mode="same"
)
idx_ref = np.nanargmin(vitesse)
angles_ref = angles_ref_tous[:, idx_ref]
angles_cal = angles - angles_ref[:, None]
print("angles de référence (deg) :", np.round(angles_ref, 1))

# %%
# ## Étape 5 — Affichage

fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
titres = ["Flexion (+) / extension (-)",
          "Abduction (+) / adduction (-)",
          "Rotation interne (+) / externe (-)"]
for k, couleur in enumerate(["tab:red", "tab:green", "tab:blue"]):
    axes[k].plot(temps, angles[k], color=couleur, alpha=0.4, label="brut")
    axes[k].plot(temps, angles_cal[k], color=couleur, label="calibré")
    axes[k].set_title(titres[k], fontsize=10)
    axes[k].set_ylabel("deg")
axes[0].legend(fontsize=8)
axes[-1].set_xlabel("Temps (s)")
plt.tight_layout()
plt.show()
