import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pyomeca import Markers
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

plt.style.use("default")

BASE_DIR = Path(__file__).resolve().parent
DOSSIER = BASE_DIR / "data2"
FICHIER_REF = DOSSIER / "ANAT_REF.c3d"
FICHIER_CAPTURE_2 = DOSSIER / "droit2_pas_ball.c3d"


def charger_c3d(chemin):
    return Markers.from_c3d(str(Path(chemin)))


CAPTURES = {
    "Arrière": [("arriere1_pas_ball.c3d", "arriere1_ball_precis.c3d"),
                ("arriere2_pas_ball.c3d", "arriere2_ball_precis.c3d")],
    "Droit":   [("droit1_pas_ball.c3d", "droit1_ball_precis.c3d"),
                ("droit2_pas_ball.c3d", "droit2_ball_precis.c3d")],
    "Avant":   [("avant1_pas_ball.c3d", "avant1_ball_precis.c3d"),
                ("avant2_pas_ball.c3d", "avant2_ball_precis.c3d")],
}
COULEURS = {"Arrière": "tab:red", "Droit": "tab:green", "Avant": "tab:blue"}

MARQUEUR_PIED = "pied droit"
SEUIL_BALLON_MM = 50
AXE_VERTICAL = None
DUREE_VITESSE_BALLON_S = 0.06
FENETRE_IMPACT_S = 0.05
TROU_MAX_S = 0.05
FENETRE_AVANT_S, FENETRE_APRES_S = 0.6, 0.1

NOMS_ANGLES = ["Flexion (+) / extension (-)",
               "Abduction (+) / adduction (-)",
               "Rotation interne (+) / externe (-)"]
COULEURS_ANGLES = ["tab:red", "tab:green", "tab:blue"]


def position(essai, nom):
    v = essai.sel(channel=nom).values
    if v.ndim == 3:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(v[0:3], axis=1)
    return v[0:3, :]


def milieu(a, b):
    return (a + b) / 2


def moyenne(a):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a)


def combler_petits_trous(y, nmax):
    y = y.copy()
    nan = np.isnan(y)
    i, n = 0, len(y)
    while i < n:
        if nan[i]:
            j = i
            while j < n and nan[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= nmax:
                y[i:j] = np.interp(np.arange(i, j), [i - 1, j], [y[i - 1], y[j]])
            i = j
        else:
            i += 1
    return y


def derivee(y, rate, fenetre_s=0.07, ordre=3, angle=False):
    fen = max(int(round(fenetre_s * rate)) | 1, 5)
    y = combler_petits_trous(y, int(round(TROU_MAX_S * rate)))
    res = np.full(len(y), np.nan)
    bornes = np.flatnonzero(np.diff(np.r_[0, (~np.isnan(y)).astype(int), 0]))
    for deb, fin in zip(bornes[::2], bornes[1::2]):
        if fin - deb >= fen:
            seg = y[deb:fin]
            if angle:
                seg = np.degrees(np.unwrap(np.radians(seg)))
            res[deb:fin] = savgol_filter(seg, fen, ordre, deriv=1, delta=1 / rate)
    return res


def vitesse_marqueur(p, rate):
    return np.sqrt(sum(derivee(p[k], rate) ** 2 for k in range(3))) / 1000


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


def positions_ballon(essai):
    noms = list(dict.fromkeys(str(n) for n in essai.channel.values.tolist()))
    ballons = [n for n in noms if "ball" in n.lower() and "moyenne" not in n.lower()]
    ballons = ballons or [n for n in noms if "ball" in n.lower()]
    if not ballons and len(noms) == 1:
        ballons = noms
    return np.stack([position(essai, n) for n in ballons])


def depart_ballon(ballons):
    deplacements = []
    for b in ballons:
        valides = np.where(~np.isnan(b).any(axis=0))[0]
        if len(valides):
            depart = np.median(b[:, valides[:20]], axis=1)
            deplacements.append(np.linalg.norm(b - depart[:, None], axis=0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        deplacement = np.nanmedian(np.stack(deplacements), axis=0)
    parti = np.where(deplacement > SEUIL_BALLON_MM)[0]
    return int(parti[0])


def detecter_impact(pied, ballons, i_parti, rate):
    n = min(pied.shape[1], ballons.shape[2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        distance = np.nanmin(np.linalg.norm(ballons[:, :, :n] - pied[None, :, :n], axis=1), axis=0)
    debut = max(0, i_parti - int(round(0.1 * rate)))
    fenetre = distance[debut: i_parti + 1]
    return debut + int(np.nanargmin(fenetre)), distance


def vitesse_ballon(ballons, i_parti, rate):
    nb = max(4, int(round(DUREE_VITESSE_BALLON_S * rate)))
    t = np.arange(nb) / rate
    pentes = []
    for b in ballons:
        seg = b[:, i_parti: i_parti + nb]
        ok = ~np.isnan(seg).any(axis=0)
        if ok.sum() >= 3:
            tt = t[: seg.shape[1]][ok]
            pentes.append([np.polyfit(tt, seg[k, ok], 1)[0] for k in range(3)])
    if not pentes:
        return None
    return np.median(np.array(pentes), axis=0) / 1000


def axe_vertical(ref):
    hanches = milieu(position(ref, "hanche gauche"), position(ref, "hanche droite"))
    epaules = milieu(position(ref, "epaule gauche"), position(ref, "epaule droite"))
    tronc = np.array([moyenne((epaules - hanches)[k]) for k in range(3)])
    k = int(np.argmax(np.abs(tronc))) if AXE_VERTICAL is None else AXE_VERTICAL
    up = np.zeros(3)
    up[k] = np.sign(tronc[k])
    return up


def direction_avant(up, v_ballon, droite_bassin):
    if v_ballon is not None:
        h = v_ballon - (v_ballon @ up) * up
        if np.linalg.norm(h) > 1e-6:
            return h / np.linalg.norm(h)
    d = droite_bassin - (droite_bassin @ up) * up
    avant = np.cross(up, d)
    return avant / np.linalg.norm(avant)


def inclinaison_tronc(essai, up, avant):
    hanches = milieu(position(essai, "hanche gauche"), position(essai, "hanche droite"))
    epaules = milieu(position(essai, "epaule gauche"), position(essai, "epaule droite"))
    return np.degrees(np.arctan2(avant @ (epaules - hanches), up @ (epaules - hanches)))


def normaliser(v):
    norme = np.linalg.norm(v)
    if norme < 1e-8:
        raise ValueError
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


NOMS_GENOU = ["cuisse droite", "genou exterieur droit", "genou interieur droite",
              "mollet droite", "cheville droite"]


def angles_genou_droit(essai):
    cuisse, gnext, gnint, mollet, cheville = [position(essai, nom) for nom in NOMS_GENOU]
    angles = np.full((3, cuisse.shape[1]), np.nan)
    for i in range(cuisse.shape[1]):
        pts = [m[:, i] for m in (cuisse, gnext, gnint, mollet, cheville)]
        if any(np.isnan(p).any() for p in pts):
            continue
        try:
            R_rel = repere_cuisse(pts[0], pts[1], pts[2]).T @ repere_jambe(pts[1], pts[2], pts[3], pts[4])
            angles[:, i] = Rotation.from_matrix(R_rel).as_euler("ZXY", degrees=True)
        except ValueError:
            continue
    return angles


def angle_reference(ref):
    angles = angles_genou_droit(ref)
    vitesse = np.convolve(np.linalg.norm(np.diff(position(ref, "cheville droite"), axis=1), axis=0),
                          np.ones(50) / 50, mode="same")
    return angles[:, np.nanargmin(vitesse)]


def traiter_captures(ref, angles_ref):
    up = axe_vertical(ref)

    captures = []
    for mouvement, paires in CAPTURES.items():
        for f_corps, f_ballon in paires:
            corps = charger_c3d(DOSSIER / f_corps)
            balle = charger_c3d(DOSSIER / f_ballon)
            rate, n = float(corps.attrs["rate"]), corps.shape[2]

            ballons = positions_ballon(balle)
            i_parti = depart_ballon(ballons)
            pied = position(corps, MARQUEUR_PIED)
            i_imp, distance = detecter_impact(pied, ballons, i_parti, rate)

            droite = position(corps, "hanche droite")[:, i_imp] - position(corps, "hanche gauche")[:, i_imp]
            avant = direction_avant(up, vitesse_ballon(ballons, i_parti, rate), droite)
            tronc = inclinaison_tronc(corps, up, avant)

            flexion = (angles_genou_droit(corps) - angles_ref[:, None])[0]
            nb_i = int(round(FENETRE_IMPACT_S * rate))
            captures.append(dict(mouvement=mouvement, fichier=f_corps, rate=rate, n=n,
                                 i_imp=i_imp, i_parti=i_parti, distance=distance,
                                 vit_pied=vitesse_marqueur(pied, rate),
                                 omega=derivee(flexion, rate, angle=True),
                                 tronc_impact=moyenne(tronc[max(0, i_imp - nb_i): i_imp + 1])))

    avant_s = min(FENETRE_AVANT_S, min(c["i_imp"] / c["rate"] for c in captures))
    apres_s = min(FENETRE_APRES_S, min((c["n"] - 1 - c["i_imp"]) / c["rate"] for c in captures))
    rate = max(c["rate"] for c in captures)
    grille = np.arange(-int(avant_s * rate), int(apres_s * rate) + 1) / rate
    for c in captures:
        t_rel = (np.arange(c["n"]) - c["i_imp"]) / c["rate"]
        c["g_pied"] = sur_grille(t_rel, c["vit_pied"], grille)
        c["g_omega"] = sur_grille(t_rel, c["omega"], grille)
    return grille, captures


def _enregistrer(fig, fichier):
    plt.tight_layout()
    fig.savefig(fichier, dpi=150, facecolor="white", transparent=False)
    plt.show()


def moyenne_courbes(courbes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(np.stack(courbes), axis=0)


def afficher_angles(temps, angles, fichier):
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, facecolor="white")
    for k, couleur in enumerate(COULEURS_ANGLES):
        axes[k].plot(temps, angles[k], color=couleur)
        axes[k].set_ylabel(f"{NOMS_ANGLES[k]}\n(deg)", fontsize=8)
        axes[k].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[-1].set_xlabel("Temps (s)")
    _enregistrer(fig, fichier)


def afficher_effet_calibration(temps, angles, angles_cal, angles_ref, fichier):
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, facecolor="white")
    for k, couleur in enumerate(COULEURS_ANGLES):
        axes[k].plot(temps, angles[k], color=couleur, alpha=0.4, label="brut")
        axes[k].plot(temps, angles_cal[k], color=couleur, label="calibré")
        axes[k].axhline(angles_ref[k], color=couleur, linewidth=0.8, linestyle=":", alpha=0.7,
                        label=f"référence = {angles_ref[k]:.1f}°")
        axes[k].set_ylabel(f"{NOMS_ANGLES[k]}\n(deg)", fontsize=8)
        axes[k].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Temps (s)")
    _enregistrer(fig, fichier)


def tracer_controle_impact(captures, grille, fichier):
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), facecolor="white", sharey=True)
    for ax, c in zip(axes.ravel(), captures):
        t = np.arange(len(c["distance"])) / c["rate"]
        t_imp = c["i_imp"] / c["rate"]
        ax.plot(t, c["distance"], color=COULEURS[c["mouvement"]],
                label=f"{c['mouvement']} - {c['fichier'][:11]}")
        ax.axvspan(t_imp + grille[0], t_imp + grille[-1], color="grey", alpha=0.2)
        ax.axvline(t_imp, color="black", linewidth=1.2)
        ax.axvline(c["i_parti"] / c["rate"], color="orange", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Distance pied-ballon (mm)")
        ax.legend(fontsize=8)
    _enregistrer(fig, fichier)


def tracer_profils(grille, captures, fichier):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, facecolor="white")
    for mouvement, couleur in COULEURS.items():
        essais = [c for c in captures if c["mouvement"] == mouvement]
        if not essais:
            continue
        tronc = np.mean([c["tronc_impact"] for c in essais])
        ax1.plot(grille, moyenne_courbes([c["g_pied"] for c in essais]), color=couleur, linewidth=2.5,
                 label=f"{mouvement} (moyenne de {len(essais)}, tronc {tronc:+.0f}°)")
        ax2.plot(grille, moyenne_courbes([c["g_omega"] for c in essais]), color=couleur, linewidth=2.5)
    for ax in (ax1, ax2):
        ax.axvline(0, color="black", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax1.set_ylabel("Vitesse du pied (m/s)")
    ax2.set_ylabel("Vitesse angulaire du genou (°/s)")
    ax2.set_xlabel("Temps par rapport à l'impact (s)")
    ax1.legend(fontsize=8)
    _enregistrer(fig, fichier)


if __name__ == "__main__":
    ref = charger_c3d(FICHIER_REF)
    angles_ref = angle_reference(ref)

    capture_2 = charger_c3d(FICHIER_CAPTURE_2)
    temps = np.arange(capture_2.shape[2]) / float(capture_2.attrs["rate"])
    angles = angles_genou_droit(capture_2)
    angles_cal = angles - angles_ref[:, None]
    afficher_angles(temps, angles, "angle_brut.png")
    afficher_angles(temps, angles_cal, "angle_calibre.png")
    afficher_effet_calibration(temps, angles, angles_cal, angles_ref, "effet_calibration.png")

    grille, captures = traiter_captures(ref, angles_ref)
    tracer_controle_impact(captures, grille, "controle_impact.png")
    tracer_profils(grille, captures, "fig1_profils_pied_genou.png")