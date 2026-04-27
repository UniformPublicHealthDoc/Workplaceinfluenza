import streamlit as st
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

# ---------------- BASIC SETTINGS ----------------
POPULATION = 400
SIM_DAYS = 60

BASELINE_IMMUNITY = 0.13

SYMPTOMATIC_RATE = 0.5
ASYMPTOMATIC_FACTOR = 0.5

E_MEAN, E_SD = 1.9, 1.23
IS_MEAN, IS_SD = 4, 1.5
IA_MEAN, IA_SD = 4, 1.5

BETA_COMMUNITY = 0.03  # fixed after calibration

ABS_THRESHOLD = int(0.10 * POPULATION)


# ---------------- HELPERS ----------------
def sample_days(m, s):
    return max(1, int(np.random.normal(m, s)))


class Person:
    def __init__(self, vaccinated, immune):
        self.vaccinated = vaccinated
        self.immune = immune
        self.state = "S"
        self.symptomatic = False
        self.days = 0
        self.duration = 0
        self.inf_day = 0

    def susceptibility(self, VE):
        return (1 - VE) if self.vaccinated else 1.0


# ---------------- NETWORK ----------------
def build_network(workplace_size):
    G = nx.Graph()
    G.add_nodes_from(range(POPULATION))

    nodes = list(G.nodes)
    random.shuffle(nodes)

    i = 0
    while i < POPULATION:
        size = max(2, int(np.random.poisson(workplace_size)))
        group = nodes[i:i+size]

        for a in group:
            for b in group:
                if a != b:
                    G.add_edge(a, b)

        i += size

    # light community mixing
    for i in range(POPULATION):
        for j in range(i+1, POPULATION):
            if random.random() < 0.01:
                G.add_edge(i, j)

    return G


# ---------------- SIMULATION ----------------
def run_sim(VE, uptake, contacts):

    BETA_WORKPLACE = 0.18  # calibrated for R0 ~1.5

    G = build_network(contacts)

    pop = []
    for _ in range(POPULATION):
        vaccinated = random.random() < uptake
        immune = random.random() < BASELINE_IMMUNITY
        pop.append(Person(vaccinated, immune))

    for i in random.sample(range(POPULATION), 2):
        p = pop[i]
        p.state = "E"
        p.duration = sample_days(E_MEAN, E_SD)

    curve, rt, abs_curve = [], [], []

    for _ in range(SIM_DAYS):

        new_inf = 0
        inf_count = 0
        absent = 0

        for i, p in enumerate(pop):
            if p.state != "I":
                continue

            if p.symptomatic and p.inf_day >= 1:
                absent += 1
                continue

            inf_count += 1

            for j in G.neighbors(i):
                q = pop[j]
                if q.state != "S" or q.immune:
                    continue

                beta = BETA_WORKPLACE * q.susceptibility(VE)

                if not p.symptomatic:
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
                        p.symptomatic = random.random() < SYMPTOMATIC_RATE
                        p.duration = sample_days(
                            IS_MEAN if p.symptomatic else IA_MEAN,
                            IS_SD if p.symptomatic else IA_SD
                        )
                    else:
                        p.state = "R"

        curve.append(new_inf)
        abs_curve.append(absent)
        rt.append(new_inf / inf_count if inf_count else 0)

    return curve, rt, abs_curve


# ---------------- STREAMLIT UI ----------------
st.title("Influenza Workplace Outbreak Simulator")

VE = st.slider("Vaccine Effectiveness", 0.0, 0.9, 0.4, 0.05)
uptake = st.slider("Vaccine Uptake", 0.0, 1.0, 0.6, 0.05)
contacts = st.slider("Workplace Contacts (avg group size)", 4, 20, 10, 1)

curve, rt, abs_curve = run_sim(VE, uptake, contacts)

days = np.arange(len(curve))

# ---------------- PLOTS ----------------
fig, ax = plt.subplots()
ax.plot(days, curve, label="New infections")
ax.plot(days, abs_curve, label="Absenteeism")
ax.legend()
ax.set_title("Epidemic + Absenteeism")
st.pyplot(fig)

fig2, ax2 = plt.subplots()
ax2.plot(days, rt, color="purple")
ax2.axhline(1, linestyle="--")
ax2.set_title("Rt over time")
st.pyplot(fig2)

fig3, ax3 = plt.subplots()
ax3.plot(days, abs_curve)
ax3.axhline(ABS_THRESHOLD, color="red", linestyle="--", label="10% threshold")
ax3.set_title("Absenteeism Threshold")
ax3.legend()
st.pyplot(fig3)

st.write(f"Peak absenteeism: {max(abs_curve):.0f}")
st.write(f"Days above 10% threshold: {sum(x > ABS_THRESHOLD for x in abs_curve)}")
