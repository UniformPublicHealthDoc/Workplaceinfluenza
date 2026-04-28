import streamlit as st
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- UI ----------------
st.title("Workplace Influenza Model (Two-Layer FOI)")

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


# ---------------- TRANSMISSION PARAMETERS ----------------
BETA_WORK = 0.18
BETA_COMM = 0.015  # small but persistent external pressure


# ---------------- PERSON ----------------
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

    return G


# ---------------- FORCE OF INFECTION ----------------
def run_sim():

    G = build_network()

    pop = []
    for _ in range(POPULATION):
        vax = random.random() < UPTAKE
        imm = random.random() < BASELINE_IMMUNITY
        pop.append(Person(vax, imm))

    # initial infection
    for i in random.sample(range(POPULATION), 2):
        pop[i].state = "E"
        pop[i].duration = max(1, int(np.random.normal(E_MEAN, E_SD)))

    curve, rt, abs_curve = [], [], []

    for day in range(SIM_DAYS):

        # ---------------- SEASONAL COMMUNITY FORCE ----------------
        season = 0.5 + 0.5 * np.sin(2 * np.pi * day / 60)
        lambda_comm_base = BETA_COMM * (0.5 + season)

        new_inf = 0
        infectious = 0
        absent = 0

        # ---------------- WORKPLACE FORCE ----------------
        for i, p in enumerate(pop):

            if p.state != "I":
                continue

            if p.symp and p.inf_day >= 1:
                absent += 1
                continue

            infectious += 1

            for j in G.neighbors(i):
                q = pop[j]

                if q.state != "S" or q.imm:
                    continue

                # workplace exposure
                lambda_work = BETA_WORK * q.susc()

                if not p.symp:
                    lambda_work *= ASYMPTOMATIC_FACTOR

                # total force of infection (work + community)
                lambda_total = lambda_work + lambda_comm_base

                if random.random() < 1 - np.exp(-lambda_total):
                    q.state = "E"
                    q.days = 0
                    q.duration = max(1, int(np.random.normal(E_MEAN, E_SD)))
                    new_inf += 1

            p.inf_day += 1

        # ---------------- STATE TRANSITIONS ----------------
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

        rt.append(new_inf / infectious if infectious > 0 else 0)

    return np.array(curve), np.array(rt), np.array(abs_curve)


# ---------------- RUN ----------------
curve, rt, abs_curve = run_sim()

days = np.arange(SIM_DAYS)

# ---------------- PLOTS ----------------
st.subheader("Epidemic Curve")
fig, ax = plt.subplots()
ax.plot(days, curve)
st.pyplot(fig)

st.subheader("Rt")
fig2, ax2 = plt.subplots()
ax2.plot(days, rt)
ax2.axhline(1, linestyle="--")
st.pyplot(fig2)

st.subheader("Absenteeism")
fig3, ax3 = plt.subplots()
ax3.plot(days, abs_curve)
ax3.axhline(ABS_10, color="orange", linestyle="--")
st.pyplot(fig3)

# ---------------- METRICS ----------------
st.subheader("Workforce Impact")

workdays = np.sum(abs_curve)

st.metric("Total workdays lost", int(workdays))
st.metric("Peak absenteeism", int(np.max(abs_curve)))
st.metric("Days >10% absent", int(np.sum(abs_curve >= ABS_10)))

# ---------------- EXPORT ----------------
df = pd.DataFrame({
    "day": days,
    "infections": curve,
    "rt": rt,
    "absenteeism": abs_curve
})

st.download_button(
    "Download CSV",
    df.to_csv(index=False),
    "flu_two_layer_model.csv"
)
