"""
Crack-healing simulation in seawater
====================================
Compares three healing agents in ONE crack exposed to seawater:
  1. UREA     - ureolytic bacteria (e.g. Sporosarcina pasteurii)
  2. NONUREA  - non-ureolytic bacteria on calcium lactate (Jonkers-type Bacillus)
  3. ENZYME   - free / encapsulated urease (EICP), no living cells

Model idea (1D along crack depth, drawn as 2D in the animation):
  - The crack is split into N slices from the mouth (x = 0, seawater) to the tip (x = L).
  - In each slice we track concentrations in the crack water, plus the crack aperture a(x, t).
  - Species diffuse along the crack (Fick's law) and seawater fixes the values at the mouth.
  - The healing agent at the crack walls turns substrate into CO3^2-, which precipitates with Ca2+
    as CaCO3 on the walls, so the aperture shrinks.
  - If the mouth closes, diffusion from the sea is blocked. This effect comes out of the model itself.

Units: SI throughout -> m, s, mol/m^3 (= mmol/L = mM).

Run:  python crack_healing.py
Output: crack_healing.png (one figure, two panels) + a summary table printed in the terminal.
        Set MAKE_ANIMATION = True to also save crack_healing.gif.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

DAY = 86400.0
DAYS = 90               # simulated time after cracking
MAKE_ANIMATION = False  # True -> also save crack_healing.gif (takes ~30 s)
CAPSULE_MODE = True     # True: bacteria/enzyme convert the agent INSIDE the capsule at the crack wall and
                        #       only the products (CO3, Ca, NH4) enter the crack water (no washout of agent).
                        # False: agent is first released into the crack water and converted there.

# Parameters

P = {
    # geometry
    "L":        0.02,     # crack depth [m]
    "a0":       0.2e-3,   # initial crack width [m] - source: book
    "N":        40,       # number of slices along depth (numerical, not physical)
    "a_min":    2e-6,     # width counted as "sealed" [m] (modelling choice)

    # seawater boundary (mouth)
    "T":        8.0,     # temperature [degC]           Danish waters ~5-15 degC
    "O2_sea":   0.28,     # mol/m3
    "Ca_sea":   10.3,     # mol/m3                        Millero et al. 2008 (standard seawater)

    # diffusion coefficients at 25 degC [m2/s] 
    "D25": {"O2": 2.2e-9, "Ca": 0.79e-9, "S": 1.0e-9, "CO3": 0.92e-9}, 
    "Ea_D": 18e3,         # J/mol, temperature scaling of D (Stokes-Einstein approx.)

    # calcite
    "M_CaCO3":  0.1001,   # kg/mol
    "rho_CaCO3": 2710.0,  # kg/m3
    "k_prec":   0.25/3600,# 1/s   CaCO3 precipitation rate constant, Mitchell et al. 2019 (S. pasteurii)

    # temperature correction of biological rates
    "T_ref":    30.0,     # degC at which the rate constants below were measured
    "Q10":      5.0,      # Jain 2021 (Achal & Chin eds.), p. 88: rate x5-10 from 10 to 20 degC -> run 5 and 10

    # healing agent reservoir in the crack walls
    # Capsules cut by the crack (links to the poster's stoichiometry calculation):
    
    "d_cap":    8e-3,     # capsule diameter [m] (poster: 5 mm or 16 mm)
    "R0": 0.276,   # 3 x 0.276 = 3x Jonkers dose (45 kg/m3), 8 mm capsules
    "N_v":      None,     # capsules per m3 concrete - Source: from dosage, e.g. 15 kg/m3 (Jonkers 2011)
    
    # Capsules cut per m2 of crack plane = N_v * d_cap (stereology); split over the two walls:
    "k_rel":    np.log(2)/(3*DAY),  # 1/s  release from capsules/matrix (half-time 3 d)

    # CAPSULE_MODE only: conversion rate at the wall = rate per cell x cells per m2 of wall
    "spores_v": 5e10,               # spores per m3 concrete, Jonkers 2011 (5x10^7 per dm3)
    "cell_factor": 55.0,            # growth after germination: active cells = spores x this factor.
                                    # 1 = no growth. Lab tests use 1e7-1e8 cells/mL (Lauchnor 2015), vs ~6e5/mL
                                    # from Jonkers' dose -> growth of 10-100x is needed to reach lab densities.
    "V_cell":   6.4e-12 / 3600,     # mol urea per CFU per s at T_ref, Lauchnor et al. 2015 (Vmax per cell)
}

# Agent-specific parameters. Same structure for all three -> fair comparison.
AGENTS = {
    "UREA": dict(
        k_max   = 5.3e-6,     # Source: Jonkers + Lauchnor et al. 2015
        Ks      = 305.0,      # mol/m3  Lauchnor et al. 2015 (Km = 305 mM)
        needs_O2 = False,     # urease keeps working without O2
        CO3_yield = 1.0,      # mol CO3 per mol urea
        NH4_per_S = 2.0,      # CO(NH2)2 + 2H2O -> 2NH4+ + CO3^2-
        Ca_per_S  = 0.0,      # urea contains no Ca -> Ca must come from seawater (set 1 if Ca salt is co-encapsulated)
        O2_per_S  = 0.0,
        t_act   = 1*DAY,      # spore activation time      
        half_life = 30*DAY,   # loss of activity            
        color   = "#2a78d6",
    ),
    "NONUREA": dict(
        k_max   = 1e-3,       # s^-1 (Calibrated on Palin et al. 2017 bead data (~1 mm3 calcite / 0.112 g beads / 14 d, 8 degC))
        Ks      = 1.0,        # mol/m3 lactate half-saturation      
        K_O2    = 0.01,       # mol/m3 O2 half-saturation (~10 uM)  
        needs_O2 = True,
        CO3_yield = 1.0,      # 1 direct; up to 6 if all CO2 reacts with portlandite (Jonkers) -> sensitivity case
        Ca_per_S  = 1.0,      # calcium lactate brings its own Ca
        NH4_per_S = 0.0,
        O2_per_S  = 6.0,      # Ca(C3H5O3)2 + 6 O2 -> CaCO3 + 5 CO2 + 5 H2O  (balanced; some papers print 7)
        t_act   = 1*DAY,
        half_life = 30*DAY,
        color   = "#eb6834",
    ),
    "ENZYME": dict(
        k_max   = 5.3e-6,  # same intrinsic activity as UREA (assumption) -> isolates the effect of decay
        Ks      = 305.0,
        needs_O2 = False,
        CO3_yield = 1.0,
        Ca_per_S  = 0.0,       # same as UREA
        NH4_per_S = 2.0,
        O2_per_S  = 0.0,
        t_act   = 0.0,         # active immediately (no germination)
        half_life = 6*3600,    # enzyme half-life is only an assumption.
        color   = "#1baf7a",
    ),
}


# Helper functions
def pos(x, eps=1e-9):
    """Smooth version of max(x, 0): keeps the solver stable when a concentration hits zero."""
    return 0.5 * (x + np.sqrt(x * x + eps * eps))

def temp_factor_bio(T):
    """Q10 correction of biological rates from T_ref to T."""
    return P["Q10"] ** ((T - P["T_ref"]) / 10.0)

def temp_factor_D(T):
    """Arrhenius-type correction of diffusion coefficients from 25 degC."""
    R = 8.314
    return np.exp(-P["Ea_D"] / R * (1/(T+273.15) - 1/298.15))

def activity(t, ag):
    """Fraction of healing agent that is active at time t (0..1).
    Bacteria: ramp up (germination) then slow decay. Enzyme: starts at 1, decays."""
    kd = np.log(2) / ag["half_life"]
    ramp = 1.0 if ag["t_act"] == 0 else 1 - np.exp(-t / ag["t_act"])
    return ramp * np.exp(-kd * t)


# Model equations  (method of lines: space discretised, time solved by ODE solver)

# State vector y = [O2, Ca, S, CO3, R, a] each of length N
#   O2, Ca, S (substrate: urea or lactate), CO3 : concentrations in crack water [mol/m3]
#   R : remaining substrate in the wall reservoir [mol/m2]
#   a : local crack aperture [m]
SPECIES = ["O2", "Ca", "S", "CO3"]

def unpack(y, N):
    return {k: y[i*N:(i+1)*N] for i, k in enumerate(SPECIES + ["R", "a"])}

def diffusion(c, c_mouth, a, D, dx):
    """d c/dt from diffusion along the crack. Flux scales with aperture a
    (a closed slice blocks transport). Mouth = seawater value, tip = no flux."""
    c_ext = np.concatenate([[c_mouth], c, [c[-1]]])
    a_ext = np.concatenate([[a[0]], a, [a[-1]]])
    a_face = 0.5 * (a_ext[1:] + a_ext[:-1])              # aperture at slice faces
    flux = -D * a_face * np.diff(c_ext) / dx             # mol/(m*s) per m crack length
    return -np.diff(flux) / dx / a

def conversion_rate(t, s, ag, T):
    """Substrate conversion rate by the healing agent [mol/m3/s] in each slice.
    Michaelis-Menten x active fraction x temperature (x O2 limitation if needed)."""
    S = pos(s["S"])
    r = (ag["k_max"] or 0.0) * temp_factor_bio(T) * activity(t, ag) * S / (1 + S / ag["Ks"])
    if ag["needs_O2"]:
        O2 = pos(s["O2"])
        r = r * O2 / (ag["K_O2"] + O2)                   # O2 limitation (non-ureolytic route)
    return r

def capsule_rate(t, s, ag, T):
    """CAPSULE_MODE: conversion inside the capsules [mol per m2 of wall per s].
    Saturated (Vmax) rate per cell x cells per m2 wall, until the capsule runs empty."""
    cells_area = P["spores_v"] * P["cell_factor"] * P["d_cap"] / 2   # active cells per m2 of crack wall
    R0 = wall_reservoir()
    r = P["V_cell"] * cells_area * temp_factor_bio(T) * activity(t, ag) \
        * pos(s["R"]) / (pos(s["R"]) + 1e-3 * R0)
    if ag["needs_O2"]:
        O2 = pos(s["O2"])
        r = r * O2 / (ag["K_O2"] + O2)                          # O2 from the crack water at the wall
    return r

def rhs(t, y, ag, T):
    N, dx = P["N"], P["L"] / P["N"]
    s = unpack(y, N)
    a = np.maximum(s["a"], P["a_min"])
    fD, fB = temp_factor_D(T), temp_factor_bio(T)
    open_ = 1 - np.exp(-(s["a"] - P["a_min"]) / 5e-6)    # switches reactions off when sealed
    open_ = np.clip(open_, 0, 1)

    if CAPSULE_MODE:
        # conversion inside the capsule; only products enter the crack water (2 walls -> factor 2/a)
        J_rel = 0.0 * s["R"]
        src = 0.0 * s["R"]
        r_cap = capsule_rate(t, s, ag, T) * open_
        cap = 2 * r_cap / a * open_                      # mol/m3/s of converted substrate (none where sealed)
        r = 0.0 * s["R"]
    else:
        # release of substrate (+ Ca) from the walls, then conversion in the crack water
        J_rel = P["k_rel"] * s["R"]                      # mol/m2/s
        src = 2 * J_rel / a                              # mol/m3/s into crack water
        r_cap = 0.0 * s["R"]
        cap = 0.0 * s["R"]
        r = conversion_rate(t, s, ag, T)

    # precipitation Ca2+ + CO3^2- -> CaCO3 
    # TODO: simple first-order in CO3 when Ca is available; could be replaced by a saturation-index law
    r_p = P["k_prec"] * s["CO3"] * s["Ca"] / (s["Ca"] + 0.1) * open_

    # assemble 
    D = {k: P["D25"][k] * fD for k in P["D25"]}
    dO2 = diffusion(s["O2"], P["O2_sea"], a, D["O2"], dx) - ag["O2_per_S"] * (r + cap)
    dCa = diffusion(s["Ca"], P["Ca_sea"], a, D["Ca"], dx) + ag["Ca_per_S"] * (src + cap) - r_p
    dS  = diffusion(s["S"], 0.0, a, D["S"], dx) + src - r
    dCO3 = diffusion(s["CO3"], 0.0, a, D["CO3"], dx) + ag["CO3_yield"] * (r + cap) - r_p
    dR  = -J_rel - r_cap
    da  = -(P["M_CaCO3"] / P["rho_CaCO3"]) * r_p * a     # solid volume per crack-plane area
    return np.concatenate([dO2, dCa, dS, dCO3, dR, da])


# Run simulations

def wall_reservoir():
    """Substrate per m2 of crack wall [mol/m2]. Uses the capsule numbers if N_v is set."""
    if P["N_v"]:
        return P["n_cap"] * P["N_v"] * P["d_cap"] / 2
    return P["R0"]

def run(agent_name, T=None, days=90, n_out=200):
    T = P["T"] if T is None else T
    ag, N = AGENTS[agent_name], P["N"]
    y0 = np.concatenate([
        np.full(N, P["O2_sea"]),   # crack fills with seawater at cracking
        np.full(N, P["Ca_sea"]),
        np.zeros(N), np.zeros(N),
        np.full(N, wall_reservoir()),
        np.full(N, P["a0"]),
    ])
    t_eval = np.linspace(0, days * DAY, n_out)
    # which unknowns influence each other (same slice + neighbouring slices) -> stable, fast solver
    tri = np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)
    sparsity = np.kron(np.ones((6, 6)), tri)
    sol = solve_ivp(rhs, (0, days * DAY), y0, args=(ag, T), t_eval=t_eval,
                    method="BDF", rtol=1e-6, atol=1e-10, jac_sparsity=sparsity)
    if not sol.success:
        print(f"WARNING {agent_name} at {T} degC: solver stopped early ({sol.message})")
    return sol


# Output formatting and plotting

ORDER = ["UREA", "NONUREA", "ENZYME"]
LABEL = {"UREA": "Urea bacteria", "NONUREA": "Non-urea bacteria", "ENZYME": "Enzyme (urease)"}
STYLE = {"UREA": "-", "NONUREA": "--", "ENZYME": ":"}
SEAL = 0.95   # "sealed" = 95 % less water flow (same measure as Palin et al. 2017)

def results(name, sol, T):
    """Numbers we report for one agent."""
    N, dx = P["N"], P["L"] / P["N"]
    t = sol.t / DAY
    a = sol.y[5*N:6*N, :]                                # crack width [m] per slice and time
    narrow = a.min(axis=0)
    flow_red = 1 - (narrow / P["a0"]) ** 3               # water flow ~ width^3 (cubic law)
    i = np.where(flow_red >= SEAL)[0]
    caco3 = (P["a0"] - a[:, -1]).sum() * dx * P["rho_CaCO3"] * 1e3      # g per m of crack
    if CAPSULE_MODE:   # substrate used up in the capsules, both walls [mol per m of crack]
        converted = 2 * (wall_reservoir() - sol.y[4*N:5*N, -1]).sum() * dx
    else:
        r = [conversion_rate(tt, unpack(sol.y[:, k], N), AGENTS[name], T) * np.maximum(a[:, k], P["a_min"])
             for k, tt in enumerate(sol.t)]
        converted = np.sum([ri.sum() * dx for ri in r][1:] * np.diff(sol.t))  # mol per m of crack
    return dict(t=t, a=a, narrow=narrow, O2=sol.y[0:N, :],
                days_to_seal=(f"{t[i[0]]:.0f}" if i.size else "not sealed"),
                flow_red=100 * flow_red[-1], caco3=caco3,
                nh4=1e3 * AGENTS[name]["NH4_per_S"] * converted)

def make_figure(res, o2_day=7):
    x = (np.arange(P["N"]) + 0.5) * P["L"] / P["N"] * 1e3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for n in ORDER:
        r, c = res[n], AGENTS[n]["color"]
        ax1.plot(r["t"], r["narrow"] * 1e6, STYLE[n], color=c, lw=2, label=LABEL[n])
        j = np.argmin(abs(r["t"] - o2_day))
        ax2.plot(x, r["O2"][:, j] * 1e3, STYLE[n], color=c, lw=2, label=LABEL[n])
    ax1.axhline(P["a0"] * (1 - SEAL) ** (1/3) * 1e6, color="grey", lw=1)
    ax1.text(1, P["a0"] * (1 - SEAL) ** (1/3) * 1e6, " sealed (95 % less water flow)",
             va="bottom", fontsize=9, color="grey")
    ax1.set(xlabel="Days after cracking", ylabel="Narrowest crack width [µm]",
            ylim=(0, P["a0"] * 1e6 * 1.05), title="A. Crack sealing over time")
    ax2.set(xlabel="Depth from crack mouth [mm]", ylabel="O₂ in crack water [µM]",
            ylim=(0, None), title=f"B. Oxygen in the crack, day {o2_day}")
    for ax in (ax1, ax2):
        ax.title.set_ha("left"); ax.title.set_position((0, 1))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e3e6ea")
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(f"Simulated self-healing of a {P['a0']*1e3:.1f} mm crack in seawater at {P['T']:.0f} °C",
                 x=0.06, ha="left")
    fig.tight_layout()
    fig.savefig("crack_healing.png", dpi=200)

def make_animation(res, frames=60):
    from matplotlib.animation import FuncAnimation, PillowWriter
    depth = np.linspace(0, P["L"] * 1e3, P["N"] + 1)
    h0 = P["a0"] / 2 * 1e6
    xs = np.linspace(-h0, h0, 81); xc = 0.5 * (xs[1:] + xs[:-1])
    t = res[ORDER[0]]["t"]; idx = np.linspace(0, len(t) - 1, frames).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(10, 5.5), sharey=True)
    fig.subplots_adjust(top=0.85, bottom=0.2, wspace=0.15)
    fig.text(0.06, 0.03, "Grey = concrete · coloured band = calcite · purple = O\u2082 in crack water "
             "(white = none) · width not to scale with depth", fontsize=9, color="grey")
    def draw(k):
        i = idx[k]
        for ax, n in zip(axes, ORDER):
            ax.clear(); r = res[n]
            half = np.maximum(r["a"][:, i], 0) / 2 * 1e6
            ax.set_facecolor("#b9bec5")
            ax.fill_betweenx(depth, -h0, h0, color=AGENTS[n]["color"], lw=0)
            water = np.where(np.abs(xc)[None, :] < half[:, None], (r["O2"][:, i] * 1e3)[:, None], np.nan)
            ax.pcolormesh(xs, depth, water, cmap="Purples", vmin=0, vmax=P["O2_sea"] * 1.8e3)
            ax.set(xlim=(-1.6 * h0, 1.6 * h0), ylim=(P["L"] * 1e3, 0), title=LABEL[n], xlabel="Width [µm]")
        axes[0].set_ylabel("Depth from crack mouth [mm] (sea at top)")
        fig.suptitle(f"Crack healing at {P['T']:.0f} °C, day {t[i]:.0f}", x=0.06, ha="left")
    FuncAnimation(fig, draw, frames=frames).save("crack_healing.gif", writer=PillowWriter(fps=10))

if __name__ == "__main__":
    res = {}
    for n in ORDER:
        if AGENTS[n]["k_max"] is None:
            print(f"NOTE: k_max not set for {n} -> it does nothing until you add a value.")
        res[n] = results(n, run(n, days=DAYS), P["T"])

    make_figure(res)
    print(f"\nResults after {DAYS} days at {P['T']:.0f} °C (per metre of crack)")
    print(f"{'Agent':<20}{'Days to seal':>14}{'Flow reduced':>14}{'CaCO3 [g]':>12}{'NH4+ [mmol]':>13}")
    for n in ORDER:
        r = res[n]
        print(f"{LABEL[n]:<20}{r['days_to_seal']:>14}{r['flow_red']:>13.0f}%{r['caco3']:>12.3f}{r['nh4']:>13.2f}")
    if MAKE_ANIMATION:
        make_animation(res)
    print("\nSaved crack_healing.png" + (" and crack_healing.gif" if MAKE_ANIMATION else ""))
