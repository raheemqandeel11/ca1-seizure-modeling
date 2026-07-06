# ============================================================
# synapse_model_tuned.py
# FINAL TUNED BASELINE - CA1 model, all 9 afferent sources.
# Uses Dr. Cavarretta's mod files (MeMo_AmpaNmda, MeMo_GABAA).
#
# Tuned to a physiological firing rate. Per Mizuseki & Buzsaki
# (2013), CA1 pyramidal rates are lognormal 0.001-10 Hz, with
# ~70% of cells below 1 Hz during active behavior. This baseline
# targets the typical low-rate range (~1-3 Hz), appropriate as a
# healthy resting state for seizure work.
#
# Excitatory active fraction found by sweep = 0.05 (~20 active
# excitatory synapses -> ~1 Hz). Inhibition at full table baseline.
# All conductances and firing rates from the project parameter table.
# NMDA off (baseline). Run length 3 s for a stable rate estimate.
# ============================================================

from neuron import h, gui
import matplotlib.pyplot as plt
import random

# ------------------------------------------------------------
# TUNED PARAMETERS
# ------------------------------------------------------------
EXC_FRACTION = 0.05     # from the sweep: lands the cell at ~1 Hz
RUN_MS       = 3000     # 3 seconds for a stable firing-rate estimate

random.seed(1)

# ------------------------------------------------------------
# Load the cell
# ------------------------------------------------------------
h.load_file("cellM1.hoc")
cell = h.CA1_PC_cAC_sig("morphology", "mpg141017_a1-2_idC.asc")
h.distance(sec=cell.soma[0])

def apical_in_range(dmin, dmax):
    return [s for s in cell.apical if dmin <= h.distance(s(0.5)) < dmax]
def basal_in_range(dmin, dmax):
    return [s for s in cell.basal if dmin <= h.distance(s(0.5)) < dmax]

SR_prox    = apical_in_range(0, 150)
SR_dist    = apical_in_range(150, 300)
SLM        = apical_in_range(300, 1e9)
apic_0_300 = apical_in_range(0, 300)

keep = []
def scaled(n, frac):
    return max(1, int(round(n * frac)))

# ------------------------------------------------------------
# Excitatory (MeMo_AmpaNmda) - scaled by EXC_FRACTION
# ------------------------------------------------------------
def add_exc(sections, n, gampa_max, rate_hz, name):
    if not sections:
        print(f"  WARNING: no sections for {name}")
        return
    interval = 1000.0 / rate_hz
    for _ in range(n):
        sec = random.choice(sections)
        syn = h.MeMo_AmpaNmda(sec(random.random()))
        syn.gampa_max = gampa_max
        syn.gnmda_max = 0.0
        syn.ampatau   = 3.0
        stim = h.NetStim(); stim.number = 100000; stim.start = 100
        stim.interval = interval; stim.noise = 1
        nc = h.NetCon(stim, syn); nc.weight[0] = 1
        keep.extend([syn, stim, nc])

# ------------------------------------------------------------
# Inhibitory (MeMo_GABAA) - full table baseline
# ------------------------------------------------------------
def add_inh(sections, n, g_max, rate_hz, name, on_soma=False):
    interval = 1000.0 / rate_hz
    for _ in range(n):
        seg = cell.soma[0](0.5) if on_soma else random.choice(sections)(random.random())
        syn = h.MeMo_GABAA(seg); syn.g_max = g_max
        stim = h.NetStim(); stim.number = 100000; stim.start = 100
        stim.interval = interval; stim.noise = 1
        nc = h.NetCon(stim, syn); nc.weight[0] = 1
        keep.extend([syn, stim, nc])

# EXCITATORY (4) - scaled to ~5% active
add_exc(SR_prox, scaled(180, EXC_FRACTION), 0.00256, 10, "CA3 proximal")
add_exc(SR_dist, scaled(180, EXC_FRACTION), 0.00796, 10, "CA3 distal")
add_exc(SLM,     scaled(20,  EXC_FRACTION), 0.00150,  1, "Entorhinal")
add_exc(SLM,     scaled(5,   EXC_FRACTION), 0.00134,  5, "Nucleus Reuniens")

# INHIBITORY (5) - full table baseline
add_inh(None,       18, 0.00095, 22.0, "PV basket", on_soma=True)
add_inh(SR_dist,    10, 0.00099,  5.9, "Bistratified")
add_inh(SLM,         8, 0.00037, 13.0, "O-LM")
add_inh(apic_0_300, 42, 0.000115, 0.7, "Ivy")
add_inh(SLM,        14, 0.000115, 6.7, "Neurogliaform")

n_exc = scaled(180,EXC_FRACTION)*2 + scaled(20,EXC_FRACTION) + scaled(5,EXC_FRACTION)
print(f"Active excitatory synapses: ~{n_exc}  |  Inhibitory: 92 (full baseline)")

# ------------------------------------------------------------
# Record and run
# ------------------------------------------------------------
t_vec  = h.Vector().record(h._ref_t)
v_soma = h.Vector().record(cell.soma[0](0.5)._ref_v)

h.celsius = 34; h.v_init = -70; h.tstop = RUN_MS
h.finitialize(h.v_init); h.continuerun(h.tstop)

v = list(v_soma)
spikes = sum(1 for j in range(1, len(v)) if v[j-1] < 0 and v[j] >= 0)
rate = spikes / (RUN_MS / 1000.0)
print(f"\nSpikes: {spikes} over {RUN_MS/1000:.0f} s   Firing rate: {rate:.2f} Hz")

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(t_vec, v_soma, color="black", linewidth=0.6)
plt.axhline(y=-70, color="gray", linestyle="--", linewidth=0.7, label="rest (-70 mV)")
plt.xlabel("Time (ms)")
plt.ylabel("Soma membrane potential (mV)")
plt.title(f"CA1 tuned baseline: 9 sources, ~{n_exc} active exc  ->  {rate:.2f} Hz "
          f"(physiological, Mizuseki & Buzsaki 2013)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
