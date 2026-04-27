import streamlit as st
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

# ---------------- PARAMETERS ----------------
POPULATION = st.slider("Population size", 100, 2000, 400, 50)
SIM_DAYS = 60

BASELINE_IMMUNITY = 0.13

SYMPTOMATIC_RATE = 0.5
ASYMPTOMATIC_FACTOR = 0.5

E_MEAN, E_SD = 1.9, 1.23
IS_MEAN, IS_SD = 4, 1.5
IA_MEAN, IA_SD = 4, 1.5

BETA_WORKPLACE = 0.18   # calibrated ~ influenza R0 range
BETA_COMMUNITY = 0.03

ABS_THRESHOLD = int(0.10 * POPULATION)


# ---------------- HELPERS ----------------
def sample_days(mean, sd):
    return max(1, int(np.random.normal(mean, sd)))


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

    G = build_network(contacts)

    pop = []
    for _ in range(POPULATION):
        vaccinated = random.random() < uptake
        immune = random.random() < BASELINE_IMMUNITY
        pop.append(Person(vaccinated, immune))

    # seed infection
    for i in random.sample(range(POPULATION), 2):
        p = pop[i]
        p.state = "E"
        p.duration = sample_days(E_MEAN, E_SD)

    curve = []
    rt_curve = []
    abs_curve = []

    for _ in range(SIM_DAYS):

        new_inf = 0
        infectious = 0
        absent = 0

        for i, p in enumerate(pop):
            if p.state != "I":
                continue

            # absenteeism rule
            if p.symptomatic and p.inf_day >= 1:
                absent += 1
                continue

            infectious += 1

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

        # state transitions
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
        rt_curve.append(new_inf / infectious if infectious > 0 else 0)

    total_workdays_lost = np.sum(abs_curve)

    return curve, rt_curve, abs_curve, total_workdays_lost


# ---------------- STREAMLIT UI ----------------
st.title("Influenza Workplace Outbreak Simulator")

VE = st.slider("Vaccine Effectiveness", 0.0, 0.9, 0.4, 0.05)
uptake = st.slider("Vaccine Uptake", 0.0, 1.0, 0.6, 0.05)
contacts = st.slider("Workplace Contacts (avg group size)", 4, 20, 10, 1)

curve, rt, abs_curve, workdays_lost = run_sim(VE, uptake, contacts)

days = np.arange(SIM_DAYS)

# ---------------- PLOTS ----------------
st.subheader("Epidemic + Absenteeism")

fig, ax = plt.subplots()
ax.plot(days, curve, label="New infections")
ax.plot(days, abs_curve, label="Absenteeism")
ax.legend()
st.pyplot(fig)

st.subheader("Rt over time")

fig2, ax2 = plt.subplots()
ax2.plot(days, rt, color="purple")
ax2.axhline(1, linestyle="--")
st.pyplot(fig2)

st.subheader("Absenteeism threshold (10%)")

fig3, ax3 = plt.subplots()
ax3.plot(days, abs_curve)
ax3.axhline(ABS_THRESHOLD, color="red", linestyle="--", label="10% threshold")
ax3.legend()
st.pyplot(fig3)

# ---------------- METRICS ----------------
st.subheader("Workforce Impact")

st.metric("Total workdays lost", int(workdays_lost))
st.metric("Peak absenteeism", int(max(abs_curve)))
st.metric("Days above 10% threshold", int(sum(x >= ABS_THRESHOLD for x in abs_curve)))
