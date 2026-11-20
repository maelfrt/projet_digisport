import numpy as np
import matplotlib.pyplot as plt
from pyomeca import Markers
from scipy.spatial.transform import Rotation

import matplotlib
print("Backend utilisé :", matplotlib.get_backend())

marqueurs_rom = Markers.from_c3d("data2/ANAT_REF.c3d")
marqueurs2 = Markers.from_c3d("data2/CAPTURE0002_ballon_v5.c3d")
mar
temps = np.arange(marqueurs2.shape[2]) / marqueurs2.attrs["rate"]
print(marqueurs2.channel.values.tolist())


def position(essai, nom):
    return essai.sel(channel=nom).values[0:3, :]  # (3, n_frames), en mm


def normaliser(v):
    return v / np.linalg.norm(v)



def vers_repere_local(p_global, origine, R):
    return R.T @ (p_global - origine)


marqueurs_genou_droit = [
    "genou exterieur droit", "genou interieur droite",
    "cuisse droite", "mollet droite", "cheville droite",
]


def compter_nan(essai, noms):   #pour les na
    for nom in noms:
        p = position(essai, nom)
        n_nan = np.isnan(p).any(axis=0).sum()
        if n_nan > 0:
            print(f"  ATTENTION : {n_nan} frame(s) avec NaN pour '{nom}'")


print("Vérification NaN - capture :")
compter_nan(marqueurs2, marqueurs_genou_droit)
print("Vérification NaN - référence :")
compter_nan(marqueurs_rom, marqueurs_genou_droit)


def repere_cuisse(cuisse, gnext, gnint):
    origine = (gnext + gnint) / 2
    z = normaliser(gnint - gnext)
    x = normaliser(np.cross(cuisse - origine, z))
    return np.column_stack((x, np.cross(z, x), z))


def repere_jambe(gnext, gnint, mollet, cheville): #on prend la cheville comme origine, l'axe du genoux pour Z, et l'ecart mollet Z pour X
    origine = cheville
    z = normaliser(gnint - gnext)
    x = normaliser(np.cross(mollet - origine, z))
    return np.column_stack((x, np.cross(z, x), z))


def angles_genou_droit(essai):
    cuisse, gnext, gnint, mollet, cheville = [position(essai, nom) for nom in ["cuisse droite", "genou exterieur droit", "genou interieur droite","mollet droite", "cheville droite"]]
    angles = np.full((3, cuisse.shape[1]), np.nan)
    for i in range(cuisse.shape[1]):
        try:
            R_cuisse = repere_cuisse(cuisse[:, i], gnext[:, i], gnint[:, i])
            R_jambe = repere_jambe(gnext[:, i], gnint[:, i], mollet[:, i], cheville[:, i])
            R_rel = R_cuisse.T @ R_jambe
            angles[:, i] = Rotation.from_matrix(R_rel).as_euler("ZXY", degrees=True)
        except ValueError as e:
            print(f"  frame {i} ignorée : {e}")  #pour les na
    return angles


angles2 = angles_genou_droit(marqueurs2)

angles_ref_tous = angles_genou_droit(marqueurs_rom)

cheville_ref = position(marqueurs_rom, "cheville droite")
vitesse = np.convolve(
    np.linalg.norm(np.diff(cheville_ref, axis=1), axis=0),
    np.ones(50) / 50, mode="same"
)
idx_ref = np.nanargmin(vitesse)
angles_ref = angles_ref_tous[:, idx_ref]
angles_cal = angles2 - angles_ref[:, None]
print("angles de référence (deg) :", np.round(angles_ref, 1))

fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
titres = ["Flexion (+) / extension (-)",
          "Abduction (+) / adduction (-)",
          "Rotation interne (+) / externe (-)"]
for k, couleur in enumerate(["tab:red", "tab:green", "tab:blue"]):
    axes[k].plot(temps, angles2[k], color=couleur, alpha=0.4, label="brut")
    axes[k].plot(temps, angles_cal[k], color=couleur, label="calibré")
    axes[k].set_title(titres[k], fontsize=10)
    axes[k].set_ylabel("deg")
axes[0].legend(fontsize=8)
axes[-1].set_xlabel("Temps (s)")
plt.tight_layout()
plt.show()


TITRES = ["Flexion (+) / extension (-)",
          "Abduction (+) / adduction (-)",
          "Rotation interne (+) / externe (-)"]
COULEURS = ["tab:red", "tab:green", "tab:blue"]


# %%
# ## 1. Angle en fonction du temps
#
# Affiche les 3 composantes de l'angle du genou (flexion, abduction,
# rotation) en fonction du temps, sur une même figure (3 sous-graphiques
# empilés, axe des temps partagé).

def afficher_angles(temps, angles, titre_general=None, fichier=None):
    """
    temps   : array (n_frames,) en secondes
    angles  : array (3, n_frames) en degrés, typiquement `angles` (brut)
              ou `angles_cal` (calibré)
    fichier : si fourni (ex. "angle_brut.png"), sauvegarde la figure dans
              ce fichier en plus (ou à la place) de l'afficher à l'écran —
              indispensable si plt.show() n'ouvre aucune fenêtre (environnement
              distant / headless, cas fréquent sur Onyxia/JupyterHub).
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for k, couleur in enumerate(COULEURS):
        axes[k].plot(temps, angles[k], color=couleur)
        axes[k].set_title(TITRES[k], fontsize=10)
        axes[k].set_ylabel("deg")
        axes[k].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[-1].set_xlabel("Temps (s)")
    if titre_general:
        fig.suptitle(titre_general)
    plt.tight_layout()
    if fichier:
        fig.savefig(fichier, dpi=150)
        print(f"Figure enregistrée : {fichier}")
    plt.show()


def afficher_effet_calibration(temps, angles, angles_cal, angles_ref, fichier=None):
    """
    temps       : array (n_frames,) en secondes
    angles      : array (3, n_frames) en degrés — angles bruts
    angles_cal  : array (3, n_frames) en degrés — angles calibrés
                  (= angles - angles_ref)
    angles_ref  : array (3,) en degrés — angle(s) de référence retiré(s)
                  (calculé sur l'essai ROM / ANAT_REF)
    fichier     : si fourni, sauvegarde la figure dans ce fichier (voir
                  afficher_angles)
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for k, couleur in enumerate(COULEURS):
        axes[k].plot(temps, angles[k], color=couleur, alpha=0.4, label="brut")
        axes[k].plot(temps, angles_cal[k], color=couleur, label="calibré")
        axes[k].axhline(angles_ref[k], color=couleur, linewidth=0.8,
                         linestyle=":", alpha=0.7,
                         label=f"référence ROM = {angles_ref[k]:.1f}°")
        axes[k].set_title(TITRES[k], fontsize=10)
        axes[k].set_ylabel("deg")
        axes[k].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Temps (s)")
    fig.suptitle("Effet de la calibration (retrait de la ROM)")
    plt.tight_layout()
    if fichier:
        fig.savefig(fichier, dpi=150)
        print(f"Figure enregistrée : {fichier}")
    plt.show()



afficher_angles(temps, angles, "Angle du genou droit (brut)", fichier="angle_brut.png")
afficher_angles(temps, angles_cal, "Angle du genou droit (calibré)", fichier="angle_calibre.png")
afficher_effet_calibration(temps, angles, angles_cal, angles_ref, fichier="effet_calibration.png")
