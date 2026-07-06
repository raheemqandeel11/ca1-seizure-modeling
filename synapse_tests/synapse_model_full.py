# ============================================================
# synapse_model_full.py
# Full baseline CA1 model with ALL NINE afferent sources.
# Uses mod files:
#   - MeMo_AmpaNmda  (excitatory)  -> strength via gampa_max, gnmda_max
#   - MeMo_GABAA     (inhibitory)  -> strength via g_max
#
# NMDA is turned OFF (gnmda_max = 0) for the baseline, matching Table 3.
# Synapses are placed by DISTANCE from soma so each source lands in its
# correct layer (Table 1B). All values from the project parameter table.
# ============================================================

from neuron import h, gui
import matplotlib.pyplot as plt
import random

random.seed(1)

# ------------------------------------------------------------
# STEP 1 - Load the cell
# ------------------------------------------------------------
h.load_file("cellM1.hoc")
cell = h.CA1_PC_cAC_sig("morphology", "mpg141017_a1-2_idC.asc")

# keep references so objects are not garbage collected
keep = []

# ------------------------------------------------------------
# Helper: get apical sections whose midpoint distance from soma
# falls in a given range. This implements the Table 1B layers.
# ------------------------------------------------------------
h.distance(sec=cell.soma[0])   # set origin at soma

def apical_in_range(dmin, dmax):
    secs = []
    for sec in cell.apical:
        d = h.distance(sec(0.5))
        if dmin <= d < dmax:
            secs.append(sec)
    return secs

def basal_in_range(dmin, dmax):
    secs = []
    for sec in cell.basal:
        d = h.distance(sec(0.5))
        if dmin <= d < dmax:
            secs.append(sec)
    return secs

# Precompute the layer section lists
SR_prox = apical_in_range(0, 150)      # CA3 proximal
SR_dist = apical_in_range(150, 300)    # CA3 distal, bistratified
SLM     = apical_in_range(300, 1e9)    # entorhinal, reuniens, O-LM, neurogliaform
basal_all = basal_in_range(0, 300)     # stratum oriens
apic_50_300 = apical_in_range(50, 300) # bistratified apical part
apic_0_300  = apical_in_range(0, 300)  # ivy apical part

print(f"Layer section counts: SR_prox={len(SR_prox)}, SR_dist={len(SR_dist)}, "
      f"SLM={len(SLM)}, basal={len(basal_all)}")

# ------------------------------------------------------------
# EXCITATORY placement function (MeMo_AmpaNmda)
# ------------------------------------------------------------
def add_excitatory(sections, n, gampa_max, rate_hz, name):
    if len(sections) == 0:
        print(f"  WARNING: no sections for {name}, skipping")
        return
    interval = 1000.0 / rate_hz
    for i in range(n):
        sec = random.choice(sections)
        loc = random.random()
        syn = h.MeMo_AmpaNmda(sec(loc))
        syn.gampa_max = gampa_max
        syn.gnmda_max = 0.0          # NMDA OFF for baseline
        syn.ampatau   = 3.0          # AMPA decay (matches table tau2)
        stim = h.NetStim()
        stim.number = 1000; stim.start = 100
        stim.interval = interval; stim.noise = 1
        nc = h.NetCon(stim, syn)
        nc.weight[0] = 1             # weight is a trigger; strength is gampa_max
        keep.extend([syn, stim, nc])

# ------------------------------------------------------------
# INHIBITORY placement function (MeMo_GABAA)
# ------------------------------------------------------------
def add_inhibitory(sections, n, g_max, rate_hz, name, on_soma=False):
    interval = 1000.0 / rate_hz
    for i in range(n):
        if on_soma:
            seg = cell.soma[0](0.5)
        else:
            if len(sections) == 0:
                print(f"  WARNING: no sections for {name}, skipping")
                return
            sec = random.choice(sections)
            seg = sec(random.random())
        syn = h.MeMo_GABAA(seg)
        syn.g_max = g_max
        # e = -75 already set in the mod file
        stim = h.NetStim()
        stim.number = 1000; stim.start = 100
        stim.interval = interval; stim.noise = 1
        nc = h.NetCon(stim, syn)
        nc.weight[0] = 1
        keep.extend([syn, stim, nc])

# ------------------------------------------------------------
# STEP 2 - Place all NINE sources (Table 2 values, 10% baseline counts)
# ------------------------------------------------------------
# --- Excitatory (4) ---
add_excitatory(SR_prox, 180, 0.00256, 10, "CA3 proximal")
add_excitatory(SR_dist, 180, 0.00796, 10, "CA3 distal")
add_excitatory(SLM,      20, 0.00150,  1, "Entorhinal")
add_excitatory(SLM,       5, 0.00134,  5, "Nucleus Reuniens")

# --- Inhibitory (5) ---
add_inhibitory(None,          18, 0.00095, 22.0, "PV basket", on_soma=True)
add_inhibitory(SR_dist,       10, 0.00099,  5.9, "Bistratified")
add_inhibitory(SLM,            8, 0.00037, 13.0, "O-LM")
add_inhibitory(apic_0_300,    42, 0.000115, 0.7, "Ivy")
add_inhibitory(SLM,           14, 0.000115, 6.7, "Neurogliaform")

print(f"Total synaptic objects placed: {len(keep)//3}")

# ------------------------------------------------------------
# STEP 3 - Record
# ------------------------------------------------------------
t_vec  = h.Vector().record(h._ref_t)
v_soma = h.Vector().record(cell.soma[0](0.5)._ref_v)

# ------------------------------------------------------------
# STEP 4 - Run
# ------------------------------------------------------------
h.celsius = 34
h.v_init  = -70
h.tstop   = 1000
h.finitialize(h.v_init)
h.continuerun(h.tstop)

# ------------------------------------------------------------
# STEP 5 - Count spikes and report firing rate
# ------------------------------------------------------------
v = list(v_soma)
spikes = sum(1 for j in range(1, len(v)) if v[j-1] < 0 and v[j] >= 0)
rate = spikes / (h.tstop / 1000.0)
print(f"\nSpikes: {spikes}   Firing rate: {rate:.1f} Hz over {h.tstop:.0f} ms")

# ------------------------------------------------------------
# STEP 6 - Plot
# ------------------------------------------------------------
plt.figure(figsize=(11, 4))
plt.plot(t_vec, v_soma, color="black", linewidth=0.7)
plt.axhline(y=-70, color="gray", linestyle="--", linewidth=0.7, label="rest (-70 mV)")
plt.xlabel("Time (ms)")
plt.ylabel("Soma membrane potential (mV)")
plt.title(f"CA1 baseline: all 9 afferent sources  ->  {spikes} spikes ({rate:.1f} Hz)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
