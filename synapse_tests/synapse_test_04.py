# ============================================================
# synapse_test_03.py
# Random distribution of synapses across the dendrites.
# Goal described:
#   - one function that randomly places EXCITATORY synapses
#   - one function that randomly places INHIBITORY synapses
#   - run and see how the cell responds.
#
# START SMALL: 20 excitatory + 10 inhibitory to verify it works.
# Scale up to baseline later (see the CONFIG block).
# ============================================================

from neuron import h, gui
import matplotlib.pyplot as plt
import random

random.seed(1)   # fixed seed -> same random placement every run (reproducible)

# ------------------------------------------------------------
# CONFIG - change these numbers to scale up later
# ------------------------------------------------------------
N_EXC = 180   # baseline: 10% of CA3 count (~1800 full) from your table
N_INH = 18    # baseline: 10% of PV basket count (~183 full) from your table

EXC_RATE = 10   # excitatory NetStim rate (Hz) - CA3 from your table
INH_RATE = 22   # inhibitory NetStim rate (Hz) - PV basket from your table

EXC_WEIGHT = 0.00256   # excitatory conductance (uS) - CA3 proximal from your table
INH_WEIGHT = 0.00095   # inhibitory conductance (uS) - PV basket from your table

# ------------------------------------------------------------
# STEP 1 - Load the cell
# ------------------------------------------------------------
h.load_file("cellM1.hoc")
cell = h.CA1_PC_cAC_sig("morphology", "mpg141017_a1-2_idC.asc")

# Collect the dendrite sections into Python lists so we can pick randomly.
apical_secs = list(cell.apical)   # apical dendrites
basal_secs  = list(cell.basal)    # basal dendrites
all_dends   = apical_secs + basal_secs

# We keep references to every object we create.
# (If we don't, Python's garbage collector deletes them and the synapses vanish.)
exc_syns, exc_stims, exc_ncs = [], [], []
inh_syns, inh_stims, inh_ncs = [], [], []

# ------------------------------------------------------------
# FUNCTION 1 - randomly place EXCITATORY synapses on the dendrites
# ------------------------------------------------------------
def add_excitatory(n):
    for i in range(n):
        sec = random.choice(all_dends)        # pick a random dendrite section
        loc = random.random()                 # pick a random position 0-1 along it

        syn = h.Exp2Syn(sec(loc))             # create the synapse there
        syn.tau1 = 0.5
        syn.tau2 = 3.0
        syn.e    = 0                          # excitatory

        stim = h.NetStim()
        stim.number   = 1000                  # many spikes (steady input)
        stim.start    = 100
        stim.interval = 1000.0 / EXC_RATE     # convert Hz to interval (ms)
        stim.noise    = 1                     # random (Poisson-like) firing

        nc = h.NetCon(stim, syn)
        nc.weight[0] = EXC_WEIGHT

        exc_syns.append(syn); exc_stims.append(stim); exc_ncs.append(nc)

# ------------------------------------------------------------
# FUNCTION 2 - randomly place INHIBITORY synapses on the dendrites
# ------------------------------------------------------------
def add_inhibitory(n):
    for i in range(n):
        sec = random.choice(all_dends)        # pick a random dendrite section
        loc = random.random()

        syn = h.Exp2Syn(sec(loc))
        syn.tau1 = 2.0                        # bistratified-like dendritic inhibition
        syn.tau2 = 16.1
        syn.e    = -75                        # inhibitory

        stim = h.NetStim()
        stim.number   = 1000
        stim.start    = 100
        stim.interval = 1000.0 / INH_RATE
        stim.noise    = 1

        nc = h.NetCon(stim, syn)
        nc.weight[0] = INH_WEIGHT

        inh_syns.append(syn); inh_stims.append(stim); inh_ncs.append(nc)

# ------------------------------------------------------------
# STEP 2 - place the synapses
# ------------------------------------------------------------
add_excitatory(N_EXC)
add_inhibitory(N_INH)
print(f"Placed {len(exc_syns)} excitatory and {len(inh_syns)} inhibitory synapses.")

# ------------------------------------------------------------
# STEP 3 - record
# ------------------------------------------------------------
t_vec  = h.Vector().record(h._ref_t)
v_soma = h.Vector().record(cell.soma[0](0.5)._ref_v)

# ------------------------------------------------------------
# STEP 4 - run
# ------------------------------------------------------------
h.celsius = 34
h.v_init  = -70
h.tstop   = 1000
h.finitialize(h.v_init)
h.continuerun(h.tstop)

# ------------------------------------------------------------
# STEP 5 - plot
# ------------------------------------------------------------
plt.figure(figsize=(11, 4))
plt.plot(t_vec, v_soma, color="black", linewidth=0.8)
plt.axhline(y=-70, color="gray", linestyle="--", linewidth=0.8, label="rest (-70 mV)")
plt.xlabel("Time (ms)")
plt.ylabel("Soma membrane potential (mV)")
plt.title(f"CA1 model: BASELINE 10% - {N_EXC} excitatory + {N_INH} inhibitory synapses")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

print("Done. If the cell fires action potentials you'll see tall spikes (~ +30 mV).")
print("If not, the membrane will just wobble below threshold.")
print("Change N_EXC, N_INH, or the WEIGHT values to explore (as Dr. Cavarretta suggested).")
