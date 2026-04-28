import streamlit as st
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- UI CONTROLS ----------------
st.title("Seasonal Influenza Workplace Simulator (Policy Model)")

POPULATION = st.slider("Population size", 100, 2000, 400, 50)
VE = st.slider("Vaccine Effectiveness", 0.0, 0.9, 0.4, 0.05)
UPTAKE = st.slider("Vaccine Uptake", 0.0, 1.0, 0.6, 0.05)
CONTACTS = st.slider("Workplace size", 4, 20, 10, 1)

SIM_DAYS = 90
BASELINE_IMMUNITY = 0.13

SYMPTOMATIC_RATE = 0.5
ASYMPTOMATIC_FACTOR = 0.5

E_MEAN, E_SD = 1.9, 1.23
IS_MEAN, IS_SD = 4, 1.5
IA_MEAN, IA_SD = 4, 1.5

ABS_10 = int(0.10 * POPULATION)
ABS_25 = int(0.25 * POPULATION)


# ---------------- CALIBRATED TRANSMISSION ----------------
def calibrate_beta(pop):
    return 0.18 * (400 / pop) ** 0.15

BETA_WORKPLACE = calibrate_beta(POPULATION)


# ---------------- MODEL OBJECT ----------------
class Person:
    def __init__(self, vax, imm):
        self.vax = vax
        self.imm = imm
        self.state = "S"
        self.symp = False
        self.days = 0
        self.duration = 0
        self.inf_day = 0

    def susc(self):
        return (1 - VE) if self.vax else 1.0


# ---------------- NETWORK ----------------
def build_network():
    G = nx.Graph()
    G.add_nodes_from(range(POPULATION))

    nodes = list(G.nodes)
    random.shuffle(nodes)

    i = 0
    while i < POPULATION:
        size = max(2, int(np.random.poisson(CONTACTS)))
        group = nodes[i:i+size]

        for a in group:
            for b in group:
                if a != b:
                    G.add_edge(a, b)

        i += size

    for i in range(POPULATION):
        for j in range(i+1, POPULATION):
            if random.random() < 0.01:
                G.add_edge(i, j)

    return G


# ---------------- SIMULATION ENGINE ----------------
def run_sim():

    G = build_network()

    pop = []
    for _ in range(POPULATION):
        vax = random.random() < UPTAKE
        imm = random.random() < BASELINE_IMMUNITY
        pop.append(Person(vax, imm))

    # initial infections
    for i in random.sample(range(POPULATION), 2):
        p = pop[i]
        p.state = "E"
        p.duration = max(1, int(np.random.normal(E_MEAN, E_SD)))

    curve, rt, abs_curve = [], [], []

    for day in range(SIM_DAYS):

        # ---------------- SEASONAL FORCING ----------------
        season = 0.5 + 0.5 * np.sin(2 * np.pi * day / 60)
        beta_scale = 0.7 + 0.6 * season

        # ---------------- EXTERNAL SEEDING ----------------
        if day % 7 == 0:
            for _ in range(np.random.poisson(1.5)):
                idx = random.randint(0, POPULATION - 1)
                if pop[idx].state == "S":
                    pop[idx].state = "E"
                    pop[idx].duration = max(1, int(np.random.normal(E_MEAN, E_SD)))

        new_inf = 0
        inf = 0
        absent = 0

        # ---------------- TRANSMISSION ----------------
        for i, p in enumerate(pop):

            if p.state != "I":
                continue

            if p.symp and p.inf_day >= 1:
                absent += 1
                continue

            inf += 1

            for j in G.neighbors(i):
                q = pop[j]

                if q.state != "S" or q.imm:
                    continue

                beta = BETA_WORKPLACE * beta_scale * q.susc()

                if not p.symp:
                    beta *= ASYMPTOMATIC_FACTOR

                if random.random() < beta:
                    q.state = "E"
                    q.days = 0
                    q.duration = max(1, int(np.random.normal(E_MEAN, E_SD)))
                    new_inf += 1

            p.inf_day += 1

        # ---------------- STATE UPDATES ----------------
        for p in pop:
            if p.state in ["E", "I"]:
                p.days += 1

                if p.days >= p.duration:
                    if p.state == "E":
                        p.state = "I"
                        p.days = 0
                        p.inf_day = 0
                        p.symp = random.random() < SYMPTOMATIC_RATE

                        p.duration = max(1, int(np.random.normal(
                            IS_MEAN if p.symp else IA_MEAN,
                            IS_SD if p.symp else IA_SD
                        )))
                    else:
                        p.state = "R"

        curve.append(new_inf)
        abs_curve.append(absent)
        rt.append(new_inf / inf if inf > 0 else 0)

    return np.array(curve), np.array(rt), np.array(abs_curve)


# ---------------- RUN MODEL ----------------
curve, rt, abs_curve = run_sim()

days = np.arange(SIM_DAYS)

# ---------------- PLOTS ----------------
st.subheader("Epidemic Dynamics")

fig, ax = plt.subplots()
ax.plot(days, curve, label="Infections")
ax.plot(days, abs_curve, label="Absenteeism")
ax.legend()
st.pyplot(fig)

st.subheader("Rt Over Time")

fig2, ax2 = plt.subplots()
ax2.plot(days, rt)
ax2.axhline(1, linestyle="--")
st.pyplot(fig2)

st.subheader("Workforce Thresholds")

fig3, ax3 = plt.subplots()
ax3.plot(days, abs_curve)
ax3.axhline(ABS_10, color="orange", linestyle="--", label="10% threshold")
ax3.axhline(ABS_25, color="red", linestyle="--", label="25% threshold")
ax3.legend()
st.pyplot(fig3)

# ---------------- METRICS ----------------
st.subheader("Workforce Impact")

workdays_lost = np.sum(abs_curve)
peak_abs = int(np.max(abs_curve))
days_10 = int(np.sum(abs_curve >= ABS_10))
days_25 = int(np.sum(abs_curve >= ABS_25))

st.metric("Total workdays lost", int(workdays_lost))
st.metric("Peak absenteeism", peak_abs)
st.metric("Days >10% absent", days_10)
st.metric("Days >25% absent", days_25)

# ---------------- EXPORT ----------------
df = pd.DataFrame({
    "day": days,
    "infections": curve,
    "rt": rt,
    "absenteeism": abs_curve
})

summary = pd.DataFrame([{
    "population": POPULATION,
    "VE": VE,
    "uptake": UPTAKE,
    "contacts": CONTACTS,
    "workdays_lost": workdays_lost,
    "peak_absenteeism": peak_abs,
    "days_above_10%": days_10,
    "days_above_25%": days_25
}])

st.download_button(
    "Download time series CSV",
    df.to_csv(index=False),
    "flu_timeseries.csv"
)

st.download_button(
    "Download summary CSV",
    summary.to_csv(index=False),
    "flu_summary.csv"
)
