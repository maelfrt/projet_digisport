CAPTUREimport warnings
import numpy as np
import matplotlib.pyplot as plt
from pyomeca import Markers
from scipy.spatial.transform import Rotation

plt.style.use("default")

DOSSIER = "data2/"
FICHIER_REF = DOSSIER + "ANAT_REF.c3d"

CAPTURES = {
    "Arrière": ["arriere1_pas_ball.c3d", "arriere2_pas_ball.c3d"],
    "Neutre":  ["droit1_pas_ball.c3d", "droit2_pas_ball.c3d"],
    "Avant":   ["avant1_pas_ball.c3d", "avant2_pas_ball.c3d"],
}
COULEURS = {"Arrière": "tab:red", "Neutre": "tab:green", "Avant": "tab:blue"}
NOMS_ANGLES = ["Flexion (+) / extension (-)",
               "Abduction (+) / adduction (-)",
               "Rotation interne (+) / externe (-)"]

N_POINTS = 200      # nombre de points de l'échelle commune 0-100 %
TROU_MAX_PCT = 3.0  # on ne comble pas un trou de données plus long que ça (en %)


# %%
# ## Fonctions de base (identiques à votre script qui fonctionne)

def position(essai, nom):
    return essai.sel(channel=nom).values[0:3, :]  # (3, n_frames), en mm


def normaliser(v):
    norme = np.linalg.norm(v)
    if norme < 1e-8:
        raise ValueError("Vecteur quasi nul")
    return v / norme


def repere_cuisse(cuisse, gnext, gnint):
    origine = (gnext + gnint) / 2
    z = normaliser(gnint - gnext)
    x = normaliser(np.cross(cuisse - origine, z))
    return np.column_stack((x, np.cross(z, x), z))


def repere_jambe(gnext, gnint, mollet, cheville):
    origine = cheville
    z = normaliser(gnint - gnext)
    x = normaliser(np.cross(mollet - origine, z))
    return np.column_stack((x, np.cross(z, x), z))


NOMS_MARQUEURS = ["cuisse droite", "genou exterieur droit", "genou interieur droite",
                  "mollet droite", "cheville droite"]


def angles_genou_droit(essai):
    cuisse, gnext, gnint, mollet, cheville = [position(essai, nom) for nom in NOMS_MARQUEURS]
    n_frames = cuisse.shape[1]
    angles = np.full((3, n_frames), np.nan)
    for i in range(n_frames):
        pts = [m[:, i] for m in (cuisse, gnext, gnint, mollet, cheville)]
        if any(np.isnan(p).any() for p in pts):
            continue  # marqueur manquant sur cette image : on laisse NaN
        try:
            R_rel = repere_cuisse(pts[0], pts[1], pts[2]).T @ repere_jambe(pts[1], pts[2], pts[3], pts[4])
            angles[:, i] = Rotation.from_matrix(R_rel).as_euler("ZXY", degrees=True)
        except ValueError:
            continue
    return angles


# %%
# ## Calibration : angle de référence calculé une seule fois (ANAT_REF)

def angle_reference(fichier_ref):
    marqueurs_ref = Markers.from_c3d(fichier_ref)
    angles_ref_tous = angles_genou_droit(marqueurs_ref)
    cheville_ref = position(marqueurs_ref, "cheville droite")
    vitesse = np.convolve(np.linalg.norm(np.diff(cheville_ref, axis=1), axis=0),
                          np.ones(50) / 50, mode="same")
    return angles_ref_tous[:, np.nanargmin(vitesse)]


# %%
# ## Mise à l'échelle commune 0-100 % (réechantillonnage)
#
# On interpole chaque courbe sur N_POINTS points répartis de 0 à 100 % de
# la durée de la capture. Les trous (marqueur manquant) ne sont pas comblés
# s'ils sont plus longs que TROU_MAX_PCT : on garde un NaN à la place.

def reechantillonner(y, n_points=N_POINTS, trou_max_pct=TROU_MAX_PCT):
    n = len(y)
    pct = np.linspace(0, 100, n)
    cible = np.linspace(0, 100, n_points)
    valides = np.where(~np.isnan(y))[0]
    if len(valides) < 2:
        return np.full(n_points, np.nan)

    resultat = np.interp(cible, pct[valides], y[valides])

    # masquer les zones situées dans un trou trop long
    pos = np.searchsorted(pct[valides], cible)
    gauche = pct[valides][np.clip(pos - 1, 0, len(valides) - 1)]
    droite = pct[valides][np.clip(pos, 0, len(valides) - 1)]
    resultat[(droite - gauche) > trou_max_pct] = np.nan
    resultat[(cible < pct[valides[0]]) | (cible > pct[valides[-1]])] = np.nan
    return resultat


def moyenne_captures(essais):
    """essais : liste de tableaux (3, n_points). Moyenne en ignorant les NaN."""
    pile = np.stack(essais)                       # (n_essais, 3, n_points)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(pile, axis=0)           # (3, n_points)


# %%
# ## Traitement de toutes les captures

def traiter_tout():
    angles_ref = angle_reference(FICHIER_REF)
    print("angles de référence (deg) :", np.round(angles_ref, 1))

    essais_par_mouvement = {}   # mouvement -> liste de (3, N_POINTS)
    moyennes = {}               # mouvement -> (3, N_POINTS)

    for mouvement, fichiers in CAPTURES.items():
        essais = []
        for f in fichiers:
            marqueurs = Markers.from_c3d(DOSSIER + f)
            duree = marqueurs.shape[2] / marqueurs.attrs["rate"]
            angles_cal = angles_genou_droit(marqueurs) - angles_ref[:, None]
            n_nan = int(np.isnan(angles_cal[0]).sum())
            print(f"{mouvement:8s} {f} : {marqueurs.shape[2]} images, "
                  f"{duree:.2f} s, {n_nan} images sans angle")
            essais.append(np.array([reechantillonner(a) for a in angles_cal]))
        essais_par_mouvement[mouvement] = essais
        moyennes[mouvement] = moyenne_captures(essais)

    return essais_par_mouvement, moyennes


# %%
# ## Graphique : une courbe moyenne par mouvement
#
# Trait épais = moyenne des 2 captures ; traits fins = chaque capture.
# `composante` : 0 = flexion (la seule vraiment exploitable avec nos
# marqueurs, voir plus haut), 1 = abduction, 2 = rotation.

def tracer_moyennes(essais_par_mouvement, moyennes, composante=0, fichier=None):
    x = np.linspace(0, 100, N_POINTS)
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    for mouvement, couleur in COULEURS.items():
        for essai in essais_par_mouvement[mouvement]:
            ax.plot(x, essai[composante], color=couleur, alpha=0.25, linewidth=1)
        ax.plot(x, moyennes[mouvement][composante], color=couleur,
                linewidth=2.5, label=f"{mouvement} (moyenne)")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"Genou droit : {NOMS_ANGLES[composante]}")
    ax.set_xlabel("Avancement du mouvement (%)")
    ax.set_ylabel("deg")
    ax.legend()
    plt.tight_layout()
    if fichier:
        fig.savefig(fichier, dpi=150, facecolor="white", transparent=False)
        print(f"Figure enregistrée : {fichier}")
    plt.show()


# %%
if __name__ == "__main__":
    essais_par_mouvement, moyennes = traiter_tout()
    tracer_moyennes(essais_par_mouvement, moyennes, composante=0,
                    fichier="moyenne_flexion_par_mouvement.png")
