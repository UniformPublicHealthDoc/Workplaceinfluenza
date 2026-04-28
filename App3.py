import streamlit as st
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- STREAMLIT INPUTS ----------------
st.title("Influenza Workplace Outbreak Simulator (Policy Model)")

POPULATION = st.slider("Population size", 100, 2000, 400, 50)
VE = st.slider("Vaccine Effectiveness", 0.0, 0.9, 0.4, 0.05)
UPTAKE = st.slider("Vaccine Uptake", 0.0, 1.0, 0.6, 0.05)
CONTACTS = st.slider("Workplace size (avg)", 4, 20, 10, 1)

SIM_DAYS = 60
BASELINE_IMMUNITY = 0.13

SYMPTOMATIC_RATE = 0.5
ASYMPTOMATIC_FACTOR = 0.5

E_MEAN, E_SD = 1.9, 1.23
IS_MEAN, IS_SD = 4, 1.5
IA_MEAN, IA_SD = 4, 1.5

ABS_10 = int(0.10 * POPULATION)


# ---------------- R0 CALIBRATION ----------------
def calibrate_beta(pop):
    """
    Scales transmission so R0 stays ~1.5 regardless of population size
    """
    base = 0.18
    return base * (400 / pop) ** 0.15  # mild normalization


BETA_WORKPLACE = calibrate_beta(POPULATION)
BETA_COMMUNITY = 0.03


# ---------------- HELPERS ----------------
def sample_days(m, s):
    return max(1, int(np.random.normal(m, s)))


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

    # community edges
    for i in range(POPULATION):
        for j in range(i+1, POPULATION):
            if random.random() < 0.01:
                G.add_edge(i, j)

    return G


# ---------------- SIMULATION ----------------
def run_sim():

    G = build_network()

    pop = []
    for _ in range(POPULATION):
        vax = random.random() < UPTAKE
        imm = random.random() < BASELINE_IMMUNITY
        pop.append(Person(vax, imm))

    for i in random.sample(range(POPULATION), 2):
        p = pop[i]
        p.state = "E"
        p.duration = sample_days(E_MEAN, E_SD)

    curve, rt, abs_curve = [], [], []

    for _ in range(SIM_DAYS):

        new_inf = 0
        inf = 0
        absent = 0

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

                beta = BETA_WORKPLACE * q.susc()

                if not p.symp:
                    beta *= ASYMPTOMATIC_FACTOR

                if random.random() < beta:
                    q.state = "E"
                    q.days = 0
                    q.duration = sample_days(E_MEAN, E_SD)
                    new_inf += 1

            p.inf_day += 1

        for p in pop:
            if p.state in ["E", "I"]:
                p.days += 1

                if p.days >= p.duration:
                    if p.state == "E":
                        p.state = "I"
                        p.days = 0
                        p.inf_day = 0
                        p.symp = random.random() < SYMPTOMATIC_RATE
                        p.duration = sample_days(
                            IS_MEAN if p.symp else IA_MEAN,
                            IS_SD if p.symp else IA_SD
                        )
                    else:
                        p.state = "R"

        curve.append(new_inf)
        abs_curve.append(absent)
        rt.append(new_inf / inf if inf else 0)

    total_workdays = np.sum(abs_curve)

    return curve, rt, abs_curve, total_workdays


curve, rt, abs_curve, workdays = run_sim()

days = np.arange(SIM_DAYS)

# ---------------- PLOTS ----------------
st.subheader("Epidemic Curve + Absenteeism")
fig, ax = plt.subplots()
ax.plot(days, curve, label="Infections")
ax.plot(days, abs_curve, label="Absenteeism")
ax.legend()
st.pyplot(fig)

st.subheader("Rt")
fig2, ax2 = plt.subplots()
ax2.plot(days, rt)
ax2.axhline(1, linestyle="--")
st.pyplot(fig2)

st.subheader("Absenteeism Thresholds")
fig3, ax3 = plt.subplots()
ax3.plot(days, abs_curve)
ax3.axhline(ABS_10, color="orange", linestyle="--", label="10% threshold")
ax3.legend()
st.pyplot(fig3)

# ---------------- METRICS ----------------
st.subheader("Workforce Impact")

peak_abs = int(max(abs_curve))
days_10 = int(sum(x >= ABS_10 for x in abs_curve))

st.metric("Total workdays lost", int(workdays))
st.metric("Peak absenteeism", peak_abs)
st.metric("Days >10% absent", days_10)

# ---------------- EXPORT REPORT ----------------
report = pd.DataFrame({
    "day": days,
    "new_infections": curve,
    "rt": rt,
    "absenteeism": abs_curve
})

summary = pd.DataFrame([{
    "population": POPULATION,
    "VE": VE,
    "uptake": UPTAKE,
    "contacts": CONTACTS,
    "total_workdays_lost": workdays,
    "peak_absenteeism": peak_abs,
    "days_above_10%": days_10,
    "beta_workplace": BETA_WORKPLACE
}])

st.download_button(
    "Download full simulation report (CSV)",
    data=report.to_csv(index=False),
    file_name="flu_simulation_timeseries.csv"
)

st.download_button(
    "Download summary report (CSV)",
    data=summary.to_csv(index=False),
    file_name="flu_simulation_summary.csv"
)
