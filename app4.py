import streamlit as st
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- UI ----------------
st.title("Metapopulation Workplace Influenza Model")

POPULATION = st.slider("Total population", 200, 3000, 800, 100)
WORKPLACES = st.slider("Number of workplaces", 5, 50, 15, 1)
VE = st.slider("Vaccine effectiveness", 0.0, 0.9, 0.4, 0.05)
UPTAKE = st.slider("Vaccine uptake", 0.0, 1.0, 0.6, 0.05)

SIM_DAYS = 90

BASELINE_IMMUNITY = 0.13
SYMPTOMATIC_RATE = 0.5

# ---------------- PARAMETERS ----------------
p_work = 0.08       # within workplace transmission
p_comm = 0.0012     # community infection risk
p_between = 0.002   # cross-workplace mixing

E_MEAN, E_SD = 1.9, 1.2
I_MEAN, I_SD = 4, 1.5


# ---------------- PERSON ----------------
class Person:
    def __init__(self, workplace, vax, imm):
        self.workplace = workplace
        self.vax = vax
        self.imm = imm

        self.state = "S"
        self.symp = False

        self.days = 0
        self.duration = 0
        self.inf_day = 0

    def susc(self):
        return (1 - VE) if self.vax else 1.0


# ---------------- POPULATION STRUCTURE ----------------
def build_population():

    pop = []

    workplaces = [[] for _ in range(WORKPLACES)]

    for i in range(POPULATION):
        wp = i % WORKPLACES

        vax = random.random() < UPTAKE
        imm = random.random() < BASELINE_IMMUNITY

        p = Person(wp, vax, imm)
        pop.append(p)
        workplaces[wp].append(i)

    return pop, workplaces


# ---------------- SIMULATION ----------------
def run_sim():

    pop, workplaces = build_population()

    # initial infections
    for _ in range(3):
        i = random.randint(0, POPULATION - 1)
        pop[i].state = "E"
        pop[i].duration = max(1, int(np.random.normal(E_MEAN, E_SD)))

    curve, rt, abs_curve = [], [], []

    for day in range(SIM_DAYS):

        season = 0.5 + 0.5 * np.sin(2 * np.pi * day / 60)
        comm_risk = p_comm * (0.5 + season)

        new_inf = 0
        infectious = 0
        absent = 0

        # ---------------- TRANSMISSION ----------------
        for i, p in enumerate(pop):

            if p.state != "I":
                continue

            if p.symp and p.inf_day >= 1:
                absent += 1
                continue

            infectious += 1

            # --- within workplace ---
            wp = workplaces[p.workplace]

            for j in wp:
                if j == i:
                    continue

                q = pop[j]

                if q.state != "S" or q.imm:
                    continue

                if random.random() < p_work * q.susc():
                    q.state = "E"
                    q.duration = max(1, int(np.random.normal(E_MEAN, E_SD)))
                    new_inf += 1

            # --- between workplaces (light mixing) ---
            if random.random() < p_between:
                j = random.randint(0, POPULATION - 1)
                q = pop[j]

                if q.state == "S" and not q.imm:
                    if random.random() < p_work * 0.5:
                        q.state = "E"
                        q.duration = max(1, int(np.random.normal(E_MEAN, E_SD)))
                        new_inf += 1

            # --- community infection ---
            if random.random() < comm_risk:
                j = random.randint(0, POPULATION - 1)
                q = pop[j]

                if q.state == "S":
                    q.state = "E"
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
                        p.duration = max(1, int(np.random.normal(I_MEAN, I_SD)))
                    else:
                        p.state = "R"

        curve.append(new_inf)
        abs_curve.append(absent)
        rt.append(new_inf / max(infectious, 1))

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
st.pyplot(fig3)

st.metric("Total workdays lost", int(np.sum(abs_curve)))
st.metric("Peak absenteeism", int(np.max(abs_curve)))
