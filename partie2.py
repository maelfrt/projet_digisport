# -*- coding: utf-8 -*-

# %%
# # Influence de l'inclinaison du tronc sur le reste du corps (frappe de ballon)
#
# Question : quand le tronc est plus ou moins penché au moment de la frappe
# (mouvement arrière / neutre / avant), quelles articulations ou parties du
# corps changent le plus ?
#
# Démarche :
# 1. Pour chaque capture : détection de l'impact (pied le plus proche du
#    ballon, avant que le ballon ne parte), comme dans le script précédent.
#    Les 4 marqueurs ballon ("ballon 1" ... "ballon 4") sont gérés, et un
#    marqueur présent en double dans un fichier ne fait plus planter.
# 2. Inclinaison du tronc = angle du vecteur centre des hanches -> centre des
#    épaules par rapport à la verticale, dans le plan "avant / haut".
#    + = tronc penché vers l'avant (dans la direction de la frappe).
# 3. Une quinzaine de variables corporelles (angles d'articulations,
#    inclinaisons de segments, torsion épaules/bassin, vitesse du pied...).
# 4. Deux analyses :
#    A. À l'IMPACT : une valeur par capture (6 points). Y a-t-il un lien entre
#       l'inclinaison du tronc et la posture de chaque articulation ?
#    B. DANS LE TEMPS (dernière seconde avant l'impact) : les variables
#       évoluent-elles ensemble pendant le geste ? (matrice de corrélation)

# %%
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyomeca import Markers
from scipy.stats import rankdata

plt.style.use("default")

# ---------------------------------------------------------------- Données

from pathlib import Path

# Dossier contenant le script
BASE_DIR = Path(__file__).resolve().parent

# Dossier contenant les fichiers C3D
DOSSIER = BASE_DIR / "data2"

# Fichier de référence
FICHIER_REF = DOSSIER / "ANAT_REF.c3d"

# (fichier corps, fichier ballon) pour chaque capture : le ballon est suivi
# dans les fichiers "*_ball_precis", le corps dans les fichiers "*_pas_ball".
CAPTURES = {
    "Arrière": [("arriere1_pas_ball.c3d", "arriere1_ball_precis.c3d"),
                ("arriere2_pas_ball.c3d", "arriere2_ball_precis.c3d")],
    "Neutre":  [("droit1_pas_ball.c3d", "droit1_ball_precis.c3d"),
                ("droit2_pas_ball.c3d", "droit2_ball_precis.c3d")],
    "Avant":   [("avant1_pas_ball.c3d", "avant1_ball_precis.c3d"),
                ("avant2_pas_ball.c3d", "avant2_ball_precis.c3d")],
}

COULEURS = {"Arrière": "tab:red", "Neutre": "tab:green", "Avant": "tab:blue"}

# Vérification des chemins
print("Dossier de travail :", BASE_DIR)
print("Fichier de référence :", FICHIER_REF)
print("Fichier existant :", FICHIER_REF.exists())

# ---------------------------------------------------------------- Réglages
MARQUEUR_PIED = "pied droit"   # pied qui frappe
SEUIL_BALLON_MM = 50           # le ballon est "parti" s'il s'éloigne de plus de 50 mm
AXE_VERTICAL = None            # None = détection automatique ; sinon 0, 1 ou 2 (x, y, z)
FENETRE_S = 1.0                # durée analysée avant l'impact (limitée par la capture la plus courte)
FENETRE_IMPACT_S = 0.05        # "valeur à l'impact" = moyenne sur les 0,05 s qui précèdent l'impact
TROU_MAX_S = 0.05              # un trou de données plus long que ça n'est pas comblé
R_CRITIQUE = 0.81              # |r| minimal pour être significatif (p < 0,05) avec 6 points


# %%
# ## Fonctions de base

_deja_signales = set()


def position(essai, nom):
    """(3, n_frames) en mm. Si un nom de marqueur existe plusieurs fois dans le
    fichier, on prend la moyenne des doublons (et on le signale)."""
    v = essai.sel(channel=nom).values
    if v.ndim == 3:
        if (id(essai), nom) not in _deja_signales:
            _deja_signales.add((id(essai), nom))
            print(f"  ATTENTION : le marqueur '{nom}' existe {v.shape[1]} fois dans ce "
                  f"fichier, moyenne des doublons utilisée (à vérifier dans Vicon/Nexus)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(v[0:3], axis=1)
    return v[0:3, :]


def milieu(a, b):
    return (a + b) / 2


def positions_ballon(essai):
    """Marqueurs du ballon : tableau (n_marqueurs, 3, n_frames)."""
    noms = list(dict.fromkeys(str(n) for n in essai.channel.values.tolist()))
    ballons = [n for n in noms if "ball" in n.lower() and "moyenne" not in n.lower()]
    ballons = ballons or [n for n in noms if "ball" in n.lower()]
    if not ballons and len(noms) == 1:
        ballons = noms
    if not ballons:
        raise KeyError(f"Aucun marqueur de ballon dans ce fichier. Marqueurs trouvés : {noms}")
    print(f"  marqueurs ballon utilisés : {ballons}")
    return np.stack([position(essai, n) for n in ballons])


# %%
# ## Détection de l'impact
#
# Le ballon vient du fichier "*_ball_precis" (même capture que le fichier corps).
# - "départ" du ballon = première image où il s'est éloigné de plus de
#   SEUIL_BALLON_MM de sa position de départ
# - impact = image où le pied (fichier corps) est le plus proche du ballon,
#   dans les 0,1 s qui précèdent ce départ
# Si le ballon ne bouge pas, on s'arrête avec un message : chercher un "minimum"
# sur toute la capture donnerait un faux impact (et des valeurs absurdes).

def depart_ballon(ballons):
    deplacements = []
    for b in ballons:
        valides = np.where(~np.isnan(b).any(axis=0))[0]
        if len(valides):
            depart = np.median(b[:, valides[:20]], axis=1)
            deplacements.append(np.linalg.norm(b - depart[:, None], axis=0))
    if not deplacements:
        raise ValueError("Marqueurs du ballon absents sur toute la capture")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        deplacement = np.nanmedian(np.stack(deplacements), axis=0)
    parti = np.where(deplacement > SEUIL_BALLON_MM)[0]
    if len(parti) == 0:
        raise ValueError(f"Le ballon ne bouge pas de plus de {SEUIL_BALLON_MM} mm (frappe non détectée)")
    return int(parti[0])


def detecter_impact(pied, ballons, i_parti, rate):
    n = min(pied.shape[1], ballons.shape[2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        distance = np.nanmin(np.linalg.norm(ballons[:, :, :n] - pied[None, :, :n], axis=1), axis=0)
    debut = max(0, i_parti - int(round(0.1 * rate)))
    fenetre = distance[debut: i_parti + 1]
    if np.all(np.isnan(fenetre)):
        raise ValueError("Distance pied-ballon inconnue autour de la frappe")
    return debut + int(np.nanargmin(fenetre))


def vitesse_ballon(ballons, i_parti, rate, duree_s=0.06):
    """Vitesse initiale du ballon : pente de la position sur duree_s après son départ.
    Retourne (vecteur en m/s, vitesse en m/s)."""
    nb = max(4, int(round(duree_s * rate)))
    t = np.arange(nb) / rate
    pentes = []
    for b in ballons:
        seg = b[:, i_parti: i_parti + nb]
        ok = ~np.isnan(seg).any(axis=0)
        if ok.sum() >= 3:
            tt = t[: seg.shape[1]][ok]
            pentes.append([np.polyfit(tt, seg[k, ok], 1)[0] for k in range(3)])
    if not pentes:
        return None, np.nan
    v = np.median(np.array(pentes), axis=0) / 1000
    return v, float(np.linalg.norm(v))


MARQUEURS_UTILES = ["hanche droite", "hanche gauche", "epaule droite", "epaule gauche",
                    "genou exterieur droit", "genou interieur droite",
                    "genou exterieur gauche", "genou interieur gauche",
                    "cheville droite", "cheville gauche", "pied gauche",
                    "lombaire gauche", "lombaire droite", "colonne centre", "front"]


def signaler_marqueurs_manquants(essai, i_imp, rate):
    """Affiche les marqueurs qui manquent juste avant l'impact (ils rendent NaN les variables
    qui en dépendent, et ces captures disparaissent alors des graphiques)."""
    nb = int(round(FENETRE_IMPACT_S * rate))
    manquants = []
    for nom in MARQUEURS_UTILES + [MARQUEUR_PIED]:
        seg = position(essai, nom)[:, max(0, i_imp - nb): i_imp + 1]
        part = np.isnan(seg).any(axis=0).mean()
        if part > 0:
            manquants.append(f"{nom} ({100 * part:.0f} % des images)")
    if manquants:
        print("  ATTENTION : marqueurs manquants juste avant l'impact : " + ", ".join(manquants))


# %%
# ## Repère : verticale et direction "avant"
#
# - verticale : l'axe du laboratoire (x, y ou z) le plus aligné avec le tronc
#   du sujet debout dans ANAT_REF (centre des hanches -> centre des épaules).
# - avant : direction horizontale du départ du ballon (= direction de la
#   frappe). Si le ballon n'est pas exploitable, on prend la direction
#   perpendiculaire à la ligne des hanches.

def axe_vertical(ref):
    hanches = milieu(position(ref, "hanche gauche"), position(ref, "hanche droite"))
    epaules = milieu(position(ref, "epaule gauche"), position(ref, "epaule droite"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tronc = np.nanmean(epaules - hanches, axis=1)
    k = int(np.argmax(np.abs(tronc))) if AXE_VERTICAL is None else AXE_VERTICAL
    up = np.zeros(3)
    up[k] = np.sign(tronc[k])
    return up


def direction_avant(up, depart, droite_bassin):
    if depart is not None:
        h = depart - (depart @ up) * up
        if np.linalg.norm(h) > 1e-6:
            return h / np.linalg.norm(h), "départ du ballon"
    d = droite_bassin - (droite_bassin @ up) * up          # ligne des hanches, à l'horizontale
    avant = np.cross(up, d)                                # axes : x avant, y haut, z droite
    return avant / np.linalg.norm(avant), "ligne des hanches (repli)"


# %%
# ## Variables corporelles
#
# Toutes calculées à chaque image à partir des marqueurs (pas de calibration
# ROM ici : on compare des postures entre captures, donc seules les
# différences comptent).

def angle_3pts(a, b, c):
    """Angle (deg) au point b entre les segments b->a et b->c."""
    u, v = a - b, c - b
    cos = np.sum(u * v, axis=0) / (np.linalg.norm(u, axis=0) * np.linalg.norm(v, axis=0))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def inclinaison(v, ref_dir, avant):
    """Angle (deg) du vecteur v par rapport à ref_dir, dans le plan (ref_dir, avant).
    + = le bout du segment est plus vers l'avant."""
    return np.degrees(np.arctan2(avant @ v, ref_dir @ v))


def torsion_epaules_bassin(epaule_d, epaule_g, hanche_d, hanche_g, up):
    """Angle (deg) entre ligne des épaules et ligne des hanches, vues de dessus.
    + = épaule droite plus en avant (par rapport au bassin)."""
    s, h = epaule_d - epaule_g, hanche_d - hanche_g
    s = s - np.outer(up, up @ s)
    h = h - np.outer(up, up @ h)
    return np.degrees(np.arctan2(up @ np.cross(h, s, axis=0), np.sum(h * s, axis=0)))


def vitesse(p, rate, fenetre=5):
    """Vitesse (m/s) d'un marqueur, lissée sur quelques images."""
    v = np.linalg.norm(np.gradient(p, axis=1), axis=0) * rate / 1000
    return pd.Series(v).rolling(fenetre, center=True, min_periods=1).mean().values


def calculer_variables(essai, up, avant, rate):
    P = lambda nom: position(essai, nom)
    hanche_d, hanche_g = P("hanche droite"), P("hanche gauche")
    epaule_d, epaule_g = P("epaule droite"), P("epaule gauche")
    c_hanches, c_epaules = milieu(hanche_d, hanche_g), milieu(epaule_d, epaule_g)
    genou_d = milieu(P("genou exterieur droit"), P("genou interieur droite"))
    genou_g = milieu(P("genou exterieur gauche"), P("genou interieur gauche"))
    cheville_d, cheville_g = P("cheville droite"), P("cheville gauche")
    pied_d, pied_g = P(MARQUEUR_PIED), P("pied gauche")
    lombaires = milieu(P("lombaire gauche"), P("lombaire droite"))
    droite = np.cross(avant, up)                           # axe "droite" du repère

    tronc = c_epaules - c_hanches
    inclinaison_tronc = inclinaison(tronc, up, avant)

    variables = {
        # articulations (0 = jambe tendue / alignée, plus grand = plus fléchi)
        "Genou D (flexion)":    180 - angle_3pts(hanche_d, genou_d, cheville_d),
        "Genou G (flexion)":    180 - angle_3pts(hanche_g, genou_g, cheville_g),
        "Hanche D (flexion)":   180 - angle_3pts(c_epaules, hanche_d, genou_d),
        "Hanche G (flexion)":   180 - angle_3pts(c_epaules, hanche_g, genou_g),
        "Cheville D":           180 - angle_3pts(genou_d, cheville_d, pied_d),
        "Cheville G":           180 - angle_3pts(genou_g, cheville_g, pied_g),
        "Rachis (flexion)":     180 - angle_3pts(lombaires, P("colonne centre"), c_epaules),
        # inclinaison des segments dans le plan avant/haut (+ = bout vers l'avant)
        "Cuisse D (inclin.)":   inclinaison(genou_d - hanche_d, -up, avant),
        "Cuisse G (inclin.)":   inclinaison(genou_g - hanche_g, -up, avant),
        "Jambe D (inclin.)":    inclinaison(cheville_d - genou_d, -up, avant),
        "Jambe G (inclin.)":    inclinaison(cheville_g - genou_g, -up, avant),
        "Tête (inclin.)":       inclinaison(P("front") - c_epaules, up, avant),
        # autres
        "Tronc (inclin. latérale)": np.degrees(np.arctan2(droite @ tronc, up @ tronc)),
        "Torsion épaules/bassin": torsion_epaules_bassin(epaule_d, epaule_g, hanche_d, hanche_g, up),
        "Vitesse pied D (m/s)": vitesse(pied_d, rate),
    }
    return inclinaison_tronc, variables


# %%
# ## Mise sur une grille de temps commune (0 = impact)

def sur_grille(t, y, grille, trou_max_s=TROU_MAX_S):
    valides = np.where(~np.isnan(y))[0]
    if len(valides) < 2:
        return np.full(len(grille), np.nan)
    tv = t[valides]
    res = np.interp(grille, tv, y[valides])
    pos = np.searchsorted(tv, grille)
    gauche = tv[np.clip(pos - 1, 0, len(tv) - 1)]
    droite = tv[np.clip(pos, 0, len(tv) - 1)]
    res[((droite - gauche) > trou_max_s) & ~np.isclose(grille, droite)] = np.nan
    res[(grille < tv[0] - 1e-9) | (grille > tv[-1] + 1e-9)] = np.nan
    return res


def moyenne(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a)


# %%
# ## Traitement des captures

def traiter_captures():
    if not FICHIER_REF.exists():
        raise FileNotFoundError(f"Fichier de référence introuvable : {FICHIER_REF}")

    up = axe_vertical(Markers.from_c3d(str(FICHIER_REF)))
    print("Axe vertical détecté (x, y, z) :", up)

    captures = []
    for mouvement, paires in CAPTURES.items():
        for f_corps, f_ballon in paires:
            print(f"{mouvement} - {f_corps}")
            chemin_corps, chemin_ballon = DOSSIER / f_corps, DOSSIER / f_ballon
            for chemin in (chemin_corps, chemin_ballon):
                if not chemin.exists():
                    raise FileNotFoundError(f"Capture introuvable : {chemin}")

            corps = Markers.from_c3d(str(chemin_corps))
            balle = Markers.from_c3d(str(chemin_ballon))
            rate, n = float(corps.attrs["rate"]), corps.shape[2]
            if abs(float(balle.attrs["rate"]) - rate) > 1e-6 or balle.shape[2] != n:
                print(f"  ATTENTION : fichiers corps/ballon différents ({n} images à {rate:g} Hz "
                      f"contre {balle.shape[2]} à {float(balle.attrs['rate']):g} Hz) : "
                      "synchronisation non garantie")

            ballons = positions_ballon(balle)
            i_parti = depart_ballon(ballons)
            i_imp = detecter_impact(position(corps, MARQUEUR_PIED), ballons, i_parti, rate)
            depart, v_ballon = vitesse_ballon(ballons, i_parti, rate)
            droite_bassin = position(corps, "hanche droite")[:, i_imp] - position(corps, "hanche gauche")[:, i_imp]
            avant, source = direction_avant(up, depart, droite_bassin)
            tronc, variables = calculer_variables(corps, up, avant, rate)
            signaler_marqueurs_manquants(corps, i_imp, rate)
            captures.append(dict(mouvement=mouvement, fichier=f_corps, rate=rate, n=n, i_imp=i_imp,
                                 v_ballon=v_ballon, tronc_brut=tronc, variables=variables))
            print(f"  impact à {i_imp / rate:.2f} s | direction avant : {source} | "
                  f"vitesse ballon : {v_ballon:.1f} m/s")

    duree = min(min(c["i_imp"] / c["rate"] for c in captures), FENETRE_S)
    rate = max(c["rate"] for c in captures)
    grille = np.arange(-int(np.floor(duree * rate)), 1) / rate
    print(f"\nFenêtre analysée : {grille[0]:.2f} s -> impact")

    masque = grille >= -FENETRE_IMPACT_S
    for c in captures:
        t_rel = (np.arange(c["n"]) - c["i_imp"]) / c["rate"]
        c["tronc"] = sur_grille(t_rel, c["tronc_brut"], grille)
        c["series"] = {nom: sur_grille(t_rel, y, grille) for nom, y in c["variables"].items()}
        c["tronc_impact"] = moyenne(c["tronc"][masque])
        c["impact"] = {nom: moyenne(s[masque]) for nom, s in c["series"].items()}
        c["impact"]["Vitesse ballon (m/s)"] = c["v_ballon"]

    print("\nInclinaison du tronc à l'impact (+ = vers l'avant) :")
    for c in captures:
        print(f"  {c['mouvement']:8s} {c['fichier']:28s} {c['tronc_impact']:+6.1f}°")
    moyennes = [np.mean([c["tronc_impact"] for c in captures if c["mouvement"] == m]) for m in CAPTURES]
    if not (moyennes[0] < moyennes[1] < moyennes[2]):
        print("  ATTENTION : l'inclinaison moyenne ne suit pas l'ordre "
              "arrière < neutre < avant. Vérifiez la direction 'avant' et la figure 1.")
    return grille, captures


# %%
# ## Analyse A : à l'impact (une valeur par capture)
#
# - rho de Spearman : le lien est-il monotone (robuste, bien adapté à 6 points) ?
# - pente : degrés de variation de la variable par degré d'inclinaison du tronc
# - effet : variation estimée sur toute la plage d'inclinaisons observée
# - score : |rho| x |effet| -> classement (fort lien ET variation importante)

def analyse_a_impact(captures):
    x = np.array([c["tronc_impact"] for c in captures])
    lignes = []
    for nom in captures[0]["impact"]:
        y = np.array([c["impact"][nom] for c in captures])
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() < 4 or np.std(y[ok]) == 0:
            continue
        xs, ys = x[ok], y[ok]
        pente = np.polyfit(xs, ys, 1)[0]
        lignes.append(dict(variable=nom,
                           rho_spearman=np.corrcoef(rankdata(xs), rankdata(ys))[0, 1],
                           r_pearson=np.corrcoef(xs, ys)[0, 1],
                           pente=pente,
                           effet_sur_plage=pente * (xs.max() - xs.min()),
                           n=int(ok.sum())))
    tableau = pd.DataFrame(lignes).set_index("variable")
    # score = lien (|rho|) x variation réelle : évite de classer en tête une variable
    # qui suit parfaitement le tronc mais ne bouge presque pas (ou l'inverse)
    tableau["score"] = tableau["rho_spearman"].abs() * tableau["effet_sur_plage"].abs()
    return tableau


# %%
# ## Analyse B : dans le temps (avant l'impact)
#
# Pour chaque capture : corrélation entre l'inclinaison du tronc et chaque
# variable sur toute la fenêtre. On donne la moyenne sur les captures et le
# nombre de captures où le signe est le même (cohérence).

def analyse_b_temps(captures):
    lignes = []
    for nom in captures[0]["series"]:
        rs = []
        for c in captures:
            x, y = c["tronc"], c["series"][nom]
            ok = ~np.isnan(x) & ~np.isnan(y)
            if ok.sum() >= 10 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
                rs.append(np.corrcoef(x[ok], y[ok])[0, 1])
        if rs:
            rs = np.array(rs)
            lignes.append(dict(variable=nom, r_moyen_temps=rs.mean(),
                               coherence_signe=f"{max((rs > 0).sum(), (rs < 0).sum())}/{len(rs)}"))
    return pd.DataFrame(lignes).set_index("variable")


def matrice_correlation(captures):
    """Matrice de corrélation (tronc + toutes les variables), moyennée sur les captures."""
    matrices = []
    for c in captures:
        d = {"Inclinaison tronc": c["tronc"], **c["series"]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            matrices.append(pd.DataFrame(d).corr().values)
    noms = ["Inclinaison tronc"] + list(captures[0]["series"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return pd.DataFrame(np.nanmean(np.stack(matrices), axis=0), index=noms, columns=noms)


# %%
# ## Figures

def _enregistrer(fig, fichier):
    plt.tight_layout()
    if fichier:
        fig.savefig(fichier, dpi=150, facecolor="white", transparent=False)
        print(f"Figure enregistrée : {fichier}")
    plt.show()


def tracer_inclinaison_tronc(grille, captures, fichier=None):
    """Figure 1 : l'inclinaison du tronc est-elle bien différente entre les 3 mouvements ?"""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    vus = set()
    for c in captures:
        style = "--" if c["mouvement"] in vus else "-"
        vus.add(c["mouvement"])
        ax.plot(grille, c["tronc"], style, color=COULEURS[c["mouvement"]], linewidth=2,
                label=f"{c['mouvement']} ({c['fichier'][:12]})")
    ax.axvline(0, color="black", linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.set_title("Inclinaison du tronc avant la frappe (+ = penché vers l'avant)")
    ax.set_xlabel("Temps par rapport à l'impact (s)")
    ax.set_ylabel("deg")
    ax.legend(fontsize=8)
    _enregistrer(fig, fichier)


def tracer_nuages(captures, tableau_a, fichier=None):
    """Figure 2 : chaque variable à l'impact en fonction de l'inclinaison du tronc."""
    noms = list(tableau_a.index)
    x = np.array([c["tronc_impact"] for c in captures])
    nb_col = 4
    nb_lig = int(np.ceil(len(noms) / nb_col))
    fig, axes = plt.subplots(nb_lig, nb_col, figsize=(3.6 * nb_col, 2.9 * nb_lig), facecolor="white")
    axes = np.atleast_1d(axes).ravel()
    marge = 0.05 * (np.nanmax(x) - np.nanmin(x))
    for ax, nom in zip(axes, noms):
        y = np.array([c["impact"][nom] for c in captures])
        for xi, yi, c in zip(x, y, captures):
            ax.scatter(xi, yi, color=COULEURS[c["mouvement"]], s=40)
        ax.set_xlim(np.nanmin(x) - marge, np.nanmax(x) + marge)      # même axe x pour tous
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() >= 3:
            a, b = np.polyfit(x[ok], y[ok], 1)
            xx = np.array([x[ok].min(), x[ok].max()])
            ax.plot(xx, a * xx + b, color="grey", linestyle="--", linewidth=1)
        n_ok = int(tableau_a.loc[nom, "n"])
        incomplet = n_ok < len(captures)
        titre = f"{nom}\nρ = {tableau_a.loc[nom, 'rho_spearman']:+.2f}"
        if incomplet:
            titre += f"  (seulement {n_ok}/{len(captures)} captures)"
        ax.set_title(titre, fontsize=9, color="tab:red" if incomplet else "black")
        ax.tick_params(labelsize=8)
    for ax in axes[len(noms):]:
        ax.axis("off")
    for m, couleur in COULEURS.items():
        axes[0].scatter([], [], color=couleur, label=m)
    axes[0].legend(fontsize=7)
    fig.supxlabel("Inclinaison du tronc à l'impact (deg, + = vers l'avant)")
    _enregistrer(fig, fichier)


def tracer_classement(tableau_a, tableau_b, fichier=None):
    """Figure 3 : classement des variables selon leur lien avec l'inclinaison du tronc."""
    t = tableau_a.join(tableau_b, how="left")
    t = t.sort_values("score")                                        # le plus fort en haut
    y = np.arange(len(t))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 0.45 * len(t) + 2), facecolor="white", sharey=True)

    ax1.barh(y, t["rho_spearman"], color=np.where(t["rho_spearman"] >= 0, "tab:blue", "tab:orange"))
    for lim in (-R_CRITIQUE, R_CRITIQUE):
        ax1.axvline(lim, color="grey", linestyle="--", linewidth=1)
    ax1.set_yticks(y)
    ax1.set_yticklabels(t.index, fontsize=9)
    for yi, (rho, eff) in enumerate(zip(t["rho_spearman"], t["effet_sur_plage"])):
        ax1.text(rho + np.sign(rho) * 0.02, yi, f"{eff:+.0f}", va="center", fontsize=8,
                 ha="left" if rho >= 0 else "right")
    ax1.set_xlim(-1.15, 1.15)
    ax1.set_title("A. À l'impact (6 captures), classé par score\nbarre = ρ de Spearman ; texte = variation sur la plage de tronc (deg)", fontsize=9)

    ax2.barh(y, t["r_moyen_temps"].fillna(0), color=np.where(t["r_moyen_temps"].fillna(0) >= 0, "tab:blue", "tab:orange"))
    for yi, (r, coh) in enumerate(zip(t["r_moyen_temps"], t["coherence_signe"])):
        if not np.isnan(r):
            ax2.text(np.sign(r) * 0.02 + r, yi, f" {coh}", va="center", fontsize=8,
                     ha="left" if r >= 0 else "right")
    ax2.set_xlim(-1.15, 1.15)
    ax2.set_title("B. Dans le temps (1 s avant l'impact)\nr moyen ; texte = captures de même signe", fontsize=9)
    _enregistrer(fig, fichier)


def tracer_matrice(matrice, fichier=None):
    """Figure 4 : matrice de corrélation complète (moyenne des captures)."""
    fig, ax = plt.subplots(figsize=(10, 8.5), facecolor="white")
    im = ax.imshow(matrice.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrice)))
    ax.set_yticks(range(len(matrice)))
    ax.set_xticklabels(matrice.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(matrice.index, fontsize=8)
    for i in range(len(matrice)):
        for j in range(len(matrice)):
            v = matrice.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Corrélations dans le temps (moyenne sur les captures)")
    _enregistrer(fig, fichier)


# %%
if __name__ == "__main__":
    grille, captures = traiter_captures()

    tableau_a = analyse_a_impact(captures)
    tableau_b = analyse_b_temps(captures)
    resultats = tableau_a.join(tableau_b, how="left").sort_values("score", ascending=False)
    print("\nVariables classées par lien avec l'inclinaison du tronc (analyse A) :")
    print(resultats.round(2).to_string())
    incompletes = resultats[resultats["n"] < len(captures)]
    if len(incompletes):
        print("\nATTENTION : variables calculées sur moins de captures (données manquantes) :")
        print(incompletes[["n"]].to_string())
    print("\nLes 3 variables les plus liées à l'inclinaison du tronc :")
    for nom, l in resultats.head(3).iterrows():
        print(f"  - {nom} : rho = {l['rho_spearman']:+.2f}, {l['pente']:+.2f} par degré de tronc "
              f"(soit {l['effet_sur_plage']:+.1f} sur la plage observée)")
    print(f"\nAvec seulement {len(captures)} captures, il faut |r| > {R_CRITIQUE} pour un lien "
          "statistiquement significatif ; en dessous, lisez les résultats comme des tendances.")
    resultats.round(3).to_csv("resultats_inclinaison_tronc.csv")

    tracer_inclinaison_tronc(grille, captures, "fig1_inclinaison_tronc.png")
    tracer_nuages(captures, tableau_a, "fig2_variables_vs_tronc.png")
    tracer_classement(tableau_a, tableau_b, "fig3_classement.png")
    tracer_matrice(matrice_correlation(captures), "fig4_matrice_correlation.png")