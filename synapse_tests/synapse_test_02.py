# ============================================================
# synapse_test_02.py
# Second synapse test: ONE excitatory + ONE inhibitory synapse
# Goal: see an EPSP (depolarizing bump) and an IPSP (hyperpolarizing dip)
#       at the soma, using the same Exp2Syn mechanism with different e.
# ============================================================

from neuron import h, gui
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# STEP 1 - Load the cell model
# ------------------------------------------------------------
h.load_file("cellM1.hoc")
cell = h.CA1_PC_cAC_sig("morphology", "mpg141017_a1-2_idC.asc")

# ------------------------------------------------------------
# STEP 2a - EXCITATORY synapse (CA3 proximal, from your table)
# ------------------------------------------------------------
exc_syn = h.Exp2Syn(cell.apic[0](0.5))
exc_syn.tau1 = 0.5    # rise time (ms)
exc_syn.tau2 = 3.0    # decay time (ms)
exc_syn.e    = 0      # 0 mV -> excitatory (depolarizing)

# ------------------------------------------------------------
# STEP 2b - INHIBITORY synapse (PV basket, from your table)
# ------------------------------------------------------------
# PV basket cells target the soma, so we place this one ON the soma.
inh_syn = h.Exp2Syn(cell.soma[0](0.5))
inh_syn.tau1 = 4.6    # rise time (ms)  - PV basket from your table
inh_syn.tau2 = 32.4   # decay time (ms) - PV basket from your table
inh_syn.e    = -75    # -75 mV -> inhibitory (hyperpolarizing)

# ------------------------------------------------------------
# STEP 3 - Presynaptic inputs (one NetStim each)
# ------------------------------------------------------------
# Excitatory NetStim - fires first
exc_stim = h.NetStim()
exc_stim.number   = 3
exc_stim.start    = 100   # excitatory spikes start at 100 ms
exc_stim.interval = 100   # 10 Hz
exc_stim.noise    = 0

# Inhibitory NetStim - fires later so you can see it separately
inh_stim = h.NetStim()
inh_stim.number   = 3
inh_stim.start    = 450   # inhibitory spikes start at 450 ms
inh_stim.interval = 100   # 10 Hz
inh_stim.noise    = 0

# ------------------------------------------------------------
# STEP 4 - Connect each NetStim to its synapse (NetCon)
# ------------------------------------------------------------
exc_nc = h.NetCon(exc_stim, exc_syn)
exc_nc.weight[0] = 0.00256   # CA3 proximal conductance (uS) from your table

inh_nc = h.NetCon(inh_stim, inh_syn)
inh_nc.weight[0] = 0.00095   # PV basket conductance (uS) from your table

# ------------------------------------------------------------
# STEP 5 - Recording
# ------------------------------------------------------------
t_vec  = h.Vector().record(h._ref_t)
v_soma = h.Vector().record(cell.soma[0](0.5)._ref_v)

# ------------------------------------------------------------
# STEP 6 - Run
# ------------------------------------------------------------
h.celsius = 34
h.v_init  = -70
h.tstop   = 800     # long enough to show both exc (early) and inh (late)
h.finitialize(h.v_init)
h.continuerun(h.tstop)

# ------------------------------------------------------------
# STEP 7 - Plot
# ------------------------------------------------------------
plt.figure(figsize=(11, 4))
plt.plot(t_vec, v_soma, color="black")
plt.axhline(y=-70, color="gray", linestyle="--", linewidth=0.8, label="rest (-70 mV)")
plt.xlabel("Time (ms)")
plt.ylabel("Soma membrane potential (mV)")
plt.title("CA1 model: excitatory EPSPs (100-300 ms) vs inhibitory IPSPs (450-650 ms)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

print("Done. Early bumps = excitatory EPSPs (go UP).")
print("Later dips = inhibitory IPSPs (go DOWN below rest).")
