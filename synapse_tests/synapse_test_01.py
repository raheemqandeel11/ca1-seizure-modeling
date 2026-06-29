# ============================================================
# synapse_test_01.py
# First synapse test for the CA1 pyramidal neuron model
# Goal: place ONE excitatory synapse, drive it with a NetStim,
#       and plot the membrane potential response.
# ============================================================

from neuron import h, gui   # h = NEURON, gui = loads the GUI + standard run tools
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# STEP 1 - Load the cell model
# ------------------------------------------------------------
# This loads cellM1.hoc which builds the CA1 pyramidal neuron.
# After this line the cell object exists with all its sections
# (soma, apic[], dend[], axon).
h.load_file("cellM1.hoc")

# Create one instance of the cell.
# The two arguments are the morphology folder and the .asc file.
cell = h.CA1_PC_cAC_sig("morphology", "mpg141017_a1-2_idC.asc")

# ------------------------------------------------------------
# STEP 2 - Create ONE excitatory synapse
# ------------------------------------------------------------
# We place it on the first apical dendrite, at the midpoint (0.5).
# Exp2Syn is the built-in two-exponential synapse mechanism.
syn = h.Exp2Syn(cell.apic[0](0.5))

# Set the synapse parameters from your parameter table (CA3 proximal):
syn.tau1 = 0.5    # rise time  (ms)
syn.tau2 = 3.0    # decay time (ms)
syn.e    = 0      # reversal potential (mV) -> 0 = excitatory (AMPA-like)

# ------------------------------------------------------------
# STEP 3 - Create the presynaptic input (NetStim)
# ------------------------------------------------------------
# A NetStim fires artificial spikes to drive the synapse.
stim = h.NetStim()
stim.number   = 5      # fire 5 spikes total (test value - just to see EPSPs)
stim.start    = 100    # first spike at 100 ms (test value - lets baseline show first)
stim.interval = 100    # 100 ms between spikes (= 10 Hz) -> CA3 rate from your table
stim.noise    = 0      # 0 = perfectly regular, 1 = fully random

# ------------------------------------------------------------
# STEP 4 - Connect the NetStim to the synapse (NetCon)
# ------------------------------------------------------------
# A NetCon is the "wire" that carries spikes from the NetStim
# to the synapse. The weight sets the synaptic strength (gmax).
nc = h.NetCon(stim, syn)
nc.weight[0] = 0.00256   # gmax in microsiemens (CA3 proximal from your table)

# ------------------------------------------------------------
# STEP 5 - Set up recording
# ------------------------------------------------------------
# Record time and the somatic membrane potential.
t_vec = h.Vector().record(h._ref_t)              # time
v_soma = h.Vector().record(cell.soma[0](0.5)._ref_v)  # soma voltage

# ------------------------------------------------------------
# STEP 6 - Run the simulation
# ------------------------------------------------------------
h.celsius = 34       # temperature (must match optimization)
h.v_init  = -70      # starting membrane potential
h.tstop   = 700      # run for 700 ms (so all 5 spikes at 10 Hz are visible)
h.finitialize(h.v_init)
h.continuerun(h.tstop)

# ------------------------------------------------------------
# STEP 7 - Plot the result
# ------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(t_vec, v_soma, color="black")
plt.xlabel("Time (ms)")
plt.ylabel("Soma membrane potential (mV)")
plt.title("CA1 model: response to one excitatory synapse (5 spikes at 20 Hz)")
plt.tight_layout()
plt.show()

print("Done. You should see 5 EPSPs (small bumps) at the soma starting at 100 ms.")
