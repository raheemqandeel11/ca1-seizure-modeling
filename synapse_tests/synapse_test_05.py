# ============================================================
# synapse_test_04.py
# Conductance test 
# Keep the synapse COUNT fixed, vary the excitatory CONDUCTANCE,
# and compare how the firing changes.
#
# Runs the SAME baseline (180 exc + 18 inh) three times at three
# different excitatory weights, then plots all three stacked.
# ============================================================

from neuron import h, gui
import matplotlib.pyplot as plt
import random

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
N_EXC = 180
N_INH = 18
EXC_RATE = 10
INH_RATE = 22
INH_WEIGHT = 0.00095   # inhibition fixed

# The three excitatory conductance levels we compare:
#  - low  = half the table value
#  - base = the CA3 proximal value from your table
#  - high = double the table value
EXC_WEIGHTS = {
    "Low (0.5x = 0.00128)":  0.00128,
    "Baseline (0.00256)":    0.00256,
    "High (2x = 0.00512)":   0.00512,
}

# ------------------------------------------------------------
# Load the cell ONCE
# ------------------------------------------------------------
h.load_file("cellM1.hoc")
cell = h.CA1_PC_cAC_sig("morphology", "mpg141017_a1-2_idC.asc")

apical_secs = list(cell.apical)
basal_secs  = list(cell.basal)
all_dends   = apical_secs + basal_secs

# ------------------------------------------------------------
# Function that builds synapses for a given excitatory weight,
# runs the simulation, and returns the recorded trace.
# ------------------------------------------------------------
def run_with_exc_weight(exc_weight):
    random.seed(1)   # SAME random placement every time -> fair comparison

    # keep references so objects are not deleted
    syns, stims, ncs = [], [], []

    # excitatory
    for i in range(N_EXC):
        sec = random.choice(all_dends)
        loc = random.random()
        syn = h.Exp2Syn(sec(loc))
        syn.tau1 = 0.5; syn.tau2 = 3.0; syn.e = 0
        stim = h.NetStim()
        stim.number = 1000; stim.start = 100
        stim.interval = 1000.0 / EXC_RATE; stim.noise = 1
        nc = h.NetCon(stim, syn); nc.weight[0] = exc_weight
        syns.append(syn); stims.append(stim); ncs.append(nc)

    # inhibitory (fixed)
    for i in range(N_INH):
        sec = random.choice(all_dends)
        loc = random.random()
        syn = h.Exp2Syn(sec(loc))
        syn.tau1 = 2.0; syn.tau2 = 16.1; syn.e = -75
        stim = h.NetStim()
        stim.number = 1000; stim.start = 100
        stim.interval = 1000.0 / INH_RATE; stim.noise = 1
        nc = h.NetCon(stim, syn); nc.weight[0] = INH_WEIGHT
        syns.append(syn); stims.append(stim); ncs.append(nc)

    # record
    t_vec  = h.Vector().record(h._ref_t)
    v_soma = h.Vector().record(cell.soma[0](0.5)._ref_v)

    # run
    h.celsius = 34; h.v_init = -70; h.tstop = 1000
    h.finitialize(h.v_init)
    h.continuerun(h.tstop)

    # count spikes (count upward threshold crossings at 0 mV)
    v = list(v_soma)
    spikes = sum(1 for j in range(1, len(v)) if v[j-1] < 0 and v[j] >= 0)

    # return copies so they are not overwritten on the next run
    return list(t_vec), list(v_soma), spikes

# ------------------------------------------------------------
# Run all three and collect results
# ------------------------------------------------------------
results = {}
for label, w in EXC_WEIGHTS.items():
    t, v, n_spikes = run_with_exc_weight(w)
    results[label] = (t, v, n_spikes)
    print(f"{label}: {n_spikes} action potentials")

# ------------------------------------------------------------
# Plot all three stacked for comparison
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
for ax, (label, (t, v, n_spikes)) in zip(axes, results.items()):
    ax.plot(t, v, color="black", linewidth=0.7)
    ax.axhline(y=-70, color="gray", linestyle="--", linewidth=0.7)
    ax.set_ylabel("Vm (mV)")
    ax.set_title(f"{label}  ->  {n_spikes} spikes")
axes[-1].set_xlabel("Time (ms)")
fig.suptitle("Effect of excitatory conductance on firing (synapse count fixed at 180/18)")
plt.tight_layout()
plt.show()

print("\nDone. More conductance -> more firing. Same synapse count, same placement.")
print("This isolates the effect of synaptic STRENGTH from synapse NUMBER.")
