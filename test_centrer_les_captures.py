
# -*- coding: utf-8 -*-

import warnings
import numpy as np
import matplotlib.pyplot as plt
from pyomeca import Markers
from scipy.spatial.transform import Rotation

plt.style.use("default")

DOSSIER = "data2/"
FICHIER_REF = DOSSIER + "ANAT_REF.c3d"

# Fichiers contenant les marqueurs du joueur
CAPTURES = {
    "Arrière": ["arriere1_pas_ball.c3d", "arriere2_pas_ball.c3d"],
    "Neutre":  ["droit1_pas_ball.c3d", "droit2_pas_ball.c3d"],
    "Avant":   ["avant1_pas_ball.c3d", "avant2_pas_ball.c3d"],
}

# Fichiers contenant le marqueur moyen du ballon
CAPTURES_BALL = {
    "Arrière": ["arriere1_ball_precis.c3d", "arriere2_ball_precis.c3d"],
    "Neutre":  ["droit1_ball_precis.c3d", "droit2_ball_precis.c3d"],
    "Avant":   ["avant1_ball_precis.c3d", "avant2_ball_precis.c3d"],
}

COULEURS = {
    "Arrière": "tab:red",
    "Neutre": "tab:green",
    "Avant": "tab:blue"
}

NOMS_ANGLES = [
    "Flexion (+) / extension (-)",
    "Abduction (+) / adduction (-)",
    "Rotation interne (+) / externe (-)"
]

# --- Détection de la frappe ---

MARQUEURS_PIED = ["pied droit"]

# Noms possibles du marqueur moyen du ballon
MARQUEURS_BALLON = [
    "ball_precis",
    "ballon_moyenne",
    "ballon moyenne",
    "ballon"
]

SEUIL_BALLON_MM = 50

# --- Fenêtre analysée autour de la frappe ---

DUREE_APRES_S = None
TROU_MAX_S = 0.05


# ============================================================
# FONCTIONS DE BASE
# ============================================================

def position(essai, nom):
    valeurs = essai.sel(channel=nom).values
    valeurs = valeurs[:3, ...]

    # Si plusieurs points portent le même nom,
    # on calcule leur position moyenne
    if valeurs.ndim == 3:
        valeurs = np.nanmean(valeurs, axis=1)

    return valeurs


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


NOMS_MARQUEURS = [
    "cuisse droite",
    "genou exterieur droit",
    "genou interieur droite",
    "mollet droite",
    "cheville droite"
]


def angles_genou_droit(essai):
    cuisse, gnext, gnint, mollet, cheville = [
        position(essai, nom) for nom in NOMS_MARQUEURS
    ]

    n_frames = cuisse.shape[1]
    angles = np.full((3, n_frames), np.nan)

    for i in range(n_frames):
        pts = [
            m[:, i]
            for m in (cuisse, gnext, gnint, mollet, cheville)
        ]

        if any(np.isnan(p).any() for p in pts):
            continue

        try:
            R_rel = (
                repere_cuisse(pts[0], pts[1], pts[2]).T
                @ repere_jambe(
                    pts[1], pts[2], pts[3], pts[4]
                )
            )

            angles[:, i] = Rotation.from_matrix(
                R_rel
            ).as_euler("ZXY", degrees=True)

        except ValueError:
            continue

    return angles


def angle_reference(fichier_ref):
    marqueurs_ref = Markers.from_c3d(fichier_ref)

    angles_ref_tous = angles_genou_droit(marqueurs_ref)
    cheville_ref = position(marqueurs_ref, "cheville droite")

    vitesse = np.convolve(
        np.linalg.norm(np.diff(cheville_ref, axis=1), axis=0),
        np.ones(50) / 50,
        mode="same"
    )

    return angles_ref_tous[:, np.nanargmin(vitesse)]


# ============================================================
# CHOIX DU MARQUEUR
# ============================================================

def choisir_marqueur(marqueurs, candidats, role):
    presents = marqueurs.channel.values.tolist()

    for nom in candidats:
        if nom in presents:
            return nom

    raise KeyError(
        f"Aucun marqueur '{role}' trouvé parmi {candidats}. "
        f"Marqueurs disponibles : {presents}"
    )


# ============================================================
# ALIGNEMENT TEMPOREL DU BALLON SUR LA CAPTURE DU JOUEUR
# ============================================================

def aligner_ballon_sur_joueur(
    ballon,
    rate_ballon,
    n_joueur,
    rate_joueur
):
    """
    Place les positions du ballon sur les images du joueur.

    Hypothèse : les deux fichiers commencent au même instant.
    """

    t_ballon = np.arange(ballon.shape[1]) / rate_ballon
    t_joueur = np.arange(n_joueur) / rate_joueur

    ballon_aligne = np.full((3, n_joueur), np.nan)

    for axe in range(3):
        valides = np.isfinite(ballon[axe])

        if np.sum(valides) < 2:
            continue

        ballon_aligne[axe] = np.interp(
            t_joueur,
            t_ballon[valides],
            ballon[axe, valides],
            left=np.nan,
            right=np.nan
        )

    return ballon_aligne


# ============================================================
# DÉTECTION DE L'INSTANT DE FRAPPE
# ============================================================

def detecter_impact(marqueurs_joueur, marqueurs_ballon):

    nom_pied = choisir_marqueur(
        marqueurs_joueur,
        MARQUEURS_PIED,
        "pied"
    )

    nom_ballon = choisir_marqueur(
        marqueurs_ballon,
        MARQUEURS_BALLON,
        "ballon"
    )

    print(
        f"  marqueurs utilisés : {nom_pied} / "
        f"{nom_ballon} (fichier ball_precis)"
    )

    # Position du pied dans le fichier du joueur
    pied = position(marqueurs_joueur, nom_pied)

    # Position du ballon dans son fichier spécifique
    ballon_brut = position(marqueurs_ballon, nom_ballon)

    # Fréquences et nombre d'images
    rate_joueur = float(marqueurs_joueur.attrs["rate"])
    rate_ballon = float(marqueurs_ballon.attrs["rate"])
    n_joueur = pied.shape[1]

    # Mise du ballon sur la grille temporelle du joueur
    ballon = aligner_ballon_sur_joueur(
        ballon_brut,
        rate_ballon,
        n_joueur,
        rate_joueur
    )

    # Vérification des dimensions
    if pied.shape != ballon.shape:
        raise ValueError(
            f"Dimensions incompatibles après alignement : "
            f"pied {pied.shape}, ballon {ballon.shape}"
        )

    # Distance pied-ballon
    distance = np.linalg.norm(
        pied - ballon,
        axis=0
    )

    distance[~np.isfinite(distance)] = np.nan

    # Détection du départ du ballon
    images_ballon = np.where(
        np.isfinite(ballon).all(axis=0)
    )[0]

    if len(images_ballon) == 0:
        raise ValueError(
            "Marqueur du ballon absent sur toute la capture"
        )

    depart = np.median(
        ballon[:, images_ballon[:20]],
        axis=1
    )

    deplacement = np.linalg.norm(
        ballon - depart[:, None],
        axis=0
    )

    parti = np.where(
        deplacement > SEUIL_BALLON_MM
    )[0]

    if len(parti) > 0:

        i_parti = int(parti[0])
        fenetre = distance[:i_parti + 1]

    else:

        i_parti = None

        print(
            "  ATTENTION : le ballon ne bouge pas de plus de "
            f"{SEUIL_BALLON_MM} mm."
            " Recherche du minimum sur toute la capture."
        )

        fenetre = distance

    if np.all(np.isnan(fenetre)):
        raise ValueError(
            "Distance pied-ballon inconnue avant le départ du ballon"
        )

    i_imp = int(np.nanargmin(fenetre))

    return i_imp, i_parti, distance


# ============================================================
# MISE SUR UNE GRILLE DE TEMPS COMMUNE
# ============================================================

def sur_grille(t, y, grille, trou_max_s=TROU_MAX_S):

    valides = np.where(~np.isnan(y))[0]

    if len(valides) < 2:
        return np.full(len(grille), np.nan)

    tv = t[valides]

    res = np.interp(
        grille,
        tv,
        y[valides]
    )

    pos = np.searchsorted(tv, grille)

    gauche = tv[
        np.clip(pos - 1, 0, len(tv) - 1)
    ]

    droite = tv[
        np.clip(pos, 0, len(tv) - 1)
    ]

    trou = (
        ((droite - gauche) > trou_max_s)
        & ~np.isclose(grille, droite)
    )

    res[trou] = np.nan

    res[
        (grille < tv[0] - 1e-9)
        | (grille > tv[-1] + 1e-9)
    ] = np.nan

    return res


def moyenne_captures(essais):

    pile = np.stack(essais)

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning
        )

        return np.nanmean(pile, axis=0)


# ============================================================
# TRAITEMENT DE TOUTES LES CAPTURES
# ============================================================

def traiter_tout():

    angles_ref = angle_reference(FICHIER_REF)

    print(
        "Angles de référence (deg) :",
        np.round(angles_ref, 1)
    )

    captures = []

    for mouvement, fichiers in CAPTURES.items():

        fichiers_ball = CAPTURES_BALL[mouvement]

        if len(fichiers) != len(fichiers_ball):
            raise ValueError(
                f"Nombre de fichiers différent pour {mouvement}"
            )

        for f, f_ball in zip(fichiers, fichiers_ball):

            print(f"\nTraitement : {f}")

            # Chargement du fichier du joueur
            marqueurs = Markers.from_c3d(DOSSIER + f)

            # Chargement du fichier ball_precis correspondant
            marqueurs_ballon = Markers.from_c3d(
                DOSSIER + f_ball
            )

            rate = float(marqueurs.attrs["rate"])
            n = marqueurs.shape[2]

            # Calcul des angles uniquement avec le fichier joueur
            angles_cal = (
                angles_genou_droit(marqueurs)
                - angles_ref[:, None]
            )

            # Détection de la frappe avec le pied du joueur
            # et le ballon du fichier ball_precis
            i_imp, i_parti, distance = detecter_impact(
                marqueurs,
                marqueurs_ballon
            )

            captures.append(
                dict(
                    mouvement=mouvement,
                    fichier=f,
                    fichier_ballon=f_ball,
                    rate=rate,
                    n=n,
                    angles=angles_cal,
                    i_imp=i_imp,
                    i_parti=i_parti,
                    distance=distance
                )
            )

            print(
                f"{mouvement:8s} {f} : "
                f"{n/rate:.2f} s, "
                f"impact à t = {i_imp/rate:.2f} s "
                f"(image {i_imp}), "
                f"{i_imp/rate:.2f} s avant / "
                f"{(n-1-i_imp)/rate:.2f} s après"
            )

    # --------------------------------------------------------
    # Durée commune : la plus courte avant la frappe
    # --------------------------------------------------------

    limitante = min(
        captures,
        key=lambda c: c["i_imp"] / c["rate"]
    )

    duree_avant = (
        limitante["i_imp"] / limitante["rate"]
    )

    print(
        f"\nDurée avant impact retenue : {duree_avant:.3f} s "
        f"(limitée par {limitante['fichier']})"
    )

    # Grille temporelle commune
    rate = max(c["rate"] for c in captures)

    nb_avant = int(np.floor(duree_avant * rate))

    # On conserve la durée après frappe jusqu'à la fin
    # de la capture la plus longue
    duree_apres_max = max(
        (c["n"] - 1 - c["i_imp"]) / c["rate"]
         for c in captures
    )
    nb_apres = int(np.ceil(duree_apres_max * rate))

    grille = np.arange(-nb_avant, nb_apres + 1) / rate
    # Interpolation des angles
    essais_par_mouvement = {
        m: [] for m in CAPTURES
    }

    for c in captures:

        t_rel = (
            np.arange(c["n"]) - c["i_imp"]
        ) / c["rate"]

        c["sur_grille"] = np.array([
            sur_grille(t_rel, a, grille)
            for a in c["angles"]
        ])

        essais_par_mouvement[
            c["mouvement"]
        ].append(c["sur_grille"])

    # Moyenne des deux captures par mouvement
    moyennes = {
        m: moyenne_captures(e)
        for m, e in essais_par_mouvement.items()
    }

    return grille, essais_par_mouvement, moyennes, captures


# ============================================================
# GRAPHIQUE PRINCIPAL
# ============================================================

def tracer_moyennes(
    grille,
    essais_par_mouvement,
    moyennes,
    composante=0,
    fichier=None
):

    fig, ax = plt.subplots(
        figsize=(9, 5),
        facecolor="white"
    )

    for mouvement, couleur in COULEURS.items():

       

        ax.plot(
            grille,
            moyennes[mouvement][composante],
            color=couleur,
            linewidth=2.5,
            label=f"{mouvement} (moyenne)"
            )

        ax.plot(
            grille,
            moyennes[mouvement][composante],
            color=couleur,
            linewidth=2.5,
            label=f"{mouvement} (moyenne)"
        )

    ax.axvline(
        0,
        color="black",
        linewidth=1,
        linestyle="--"
    )

    ax.axhline(
        0,
        color="black",
        linewidth=0.5,
        linestyle=":"
    )

    ax.set_title(
        f"Genou droit : {NOMS_ANGLES[composante]}"
    )

    ax.set_xlabel("Temps par rapport à la frappe (s)")
    ax.set_ylabel("deg")

    ax.legend()

    plt.tight_layout()

    if fichier:
        fig.savefig(
            fichier,
            dpi=150,
            facecolor="white",
            transparent=False
        )

        print(f"Figure enregistrée : {fichier}")

    plt.show()


# ============================================================
# GRAPHIQUE DE CONTRÔLE DES IMPACTS
# ============================================================

def tracer_controle_impacts(captures, grille, fichier=None):

    fig, axes = plt.subplots(
        3, 2,
        figsize=(11, 8),
        facecolor="white",
        sharey=True
    )

    for ax, c in zip(axes.ravel(), captures):

        t = np.arange(c["n"]) / c["rate"]
        t_imp = c["i_imp"] / c["rate"]

        ax.plot(
            t,
            c["distance"],
            color=COULEURS[c["mouvement"]]
        )

        ax.axvspan(
            t_imp + grille[0],
            t_imp + grille[-1],
            color="grey",
            alpha=0.2
        )

        ax.axvline(
            t_imp,
            color="black",
            linewidth=1.2
        )

        if c["i_parti"] is not None:

            ax.axvline(
                c["i_parti"] / c["rate"],
                color="orange",
                linestyle=":",
                linewidth=1.2
            )

        ax.set_title(
            f"{c['mouvement']} - {c['fichier'][:11]}",
            fontsize=9
        )

        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Distance pied-ballon (mm)")

    fig.suptitle(
        "Contrôle : noir = impact, orange = ballon parti, "
        "gris = fenêtre conservée"
    )

    plt.tight_layout()

    if fichier:
        fig.savefig(
            fichier,
            dpi=150,
            facecolor="white",
            transparent=False
        )

        print(f"Figure enregistrée : {fichier}")

    plt.show()


# ============================================================
# EXÉCUTION
# ============================================================

if __name__ == "__main__":

    grille, essais_par_mouvement, moyennes, captures = traiter_tout()

    tracer_controle_impacts(
        captures,
        grille,
        fichier="controle_impacts.png"
    )

    tracer_moyennes(
        grille,
        essais_par_mouvement,
        moyennes,
        composante=0,
        fichier="moyenne_flexion_alignee_frappe.png"
    )


# ============================================================
# CALCUL DES ANGLES MAXIMAUX
# ============================================================

def afficher_angles_maximaux(grille, moyennes):

    print("\n==========================================")
    print("       ANGLES MAXIMAUX PAR MOUVEMENT")
    print("==========================================")

    for composante, nom_angle in enumerate(NOMS_ANGLES):

        print(f"\n{nom_angle}")

        for mouvement in CAPTURES:

            courbe = moyennes[mouvement][composante]

            # On ignore les valeurs manquantes
            valides = np.where(np.isfinite(courbe))[0]

            if len(valides) == 0:
                print(f"{mouvement} : aucune donnée disponible")
                continue

            # Recherche du maximum
            i_max = valides[np.argmax(courbe[valides])]

            angle_max = courbe[i_max]
            temps_max = grille[i_max]

            print(
                f"{mouvement:8s} : "
                f"maximum = {angle_max:.2f} deg "
                f"à t = {temps_max:.3f} s"
            )

    print("\n==========================================")



if __name__ == "__main__":

    grille, essais_par_mouvement, moyennes, captures = traiter_tout()

    tracer_controle_impacts(
        captures,
        grille,
        fichier="controle_impacts.png"
    )

    tracer_moyennes(
        grille,
        essais_par_mouvement,
        moyennes,
        composante=0,
        fichier="moyenne_flexion_alignee_frappe.png"
    )

    # Affichage des angles maximaux
    afficher_angles_maximaux(grille, moyennes)