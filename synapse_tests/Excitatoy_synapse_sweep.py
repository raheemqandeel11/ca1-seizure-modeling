# ============================================================
# synapse_sweep.py
# Excitatory active-fraction SWEEP.
# Goal: find the excitatory active fraction that lands the CA1
# cell in the physiological 1-3 Hz range (Mizuseki & Buzsaki 2013:
# CA1 pyramidal rates 0.001-10 Hz lognormal, 70% below 1 Hz).
#
# METHOD: sweep the parameter across a range and report firing rate for each.
#
# Inhibition stays at FULL table baseline. Only the EXCITATORY
# active fraction changes. All conductances and rates come from
# the project parameter table. NMDA off (baseline).
# ============================================================

from neuron import h, gui
import matplotlib.pyplot as plt
import random

# fractions of the excitatory 10% baseline to test
EXC_FRACTIONS = [1.0, 0.5, 0.25, 0.1, 0.05]

# ------------------------------------------------------------
# Load the cell ONCE
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

# ------------------------------------------------------------
# Build + run the model for a given excitatory fraction.
# Inhibition always at full baseline (inh_frac = 1.0).
# Returns (spikes, rate, t, v).
# ------------------------------------------------------------
def build_and_run(exc_frac):
    random.seed(1)
    keep = []

    def scaled(n, frac):
        return max(1, int(round(n * frac)))

    def add_exc(sections, n, gampa_max, rate_hz):
        if not sections: return
        interval = 1000.0 / rate_hz
        for _ in range(n):
            sec = random.choice(sections)
            syn = h.MeMo_AmpaNmda(sec(random.random()))
            syn.gampa_max = gampa_max
            syn.gnmda_max = 0.0
            syn.ampatau = 3.0
            stim = h.NetStim(); stim.number = 1000; stim.start = 100
            stim.interval = interval; stim.noise = 1
            nc = h.NetCon(stim, syn); nc.weight[0] = 1
            keep.extend([syn, stim, nc])

    def add_inh(sections, n, g_max, rate_hz, on_soma=False):
        interval = 1000.0 / rate_hz
        for _ in range(n):
            seg = cell.soma[0](0.5) if on_soma else random.choice(sections)(random.random())
            syn = h.MeMo_GABAA(seg); syn.g_max = g_max
            stim = h.NetStim(); stim.number = 1000; stim.start = 100
            stim.interval = interval; stim.noise = 1
            nc = h.NetCon(stim, syn); nc.weight[0] = 1
            keep.extend([syn, stim, nc])

    # EXCITATORY - scaled by exc_frac
    add_exc(SR_prox, scaled(180, exc_frac), 0.00256, 10)
    add_exc(SR_dist, scaled(180, exc_frac), 0.00796, 10)
    add_exc(SLM,     scaled(20,  exc_frac), 0.00150,  1)
    add_exc(SLM,     scaled(5,   exc_frac), 0.00134,  5)

    # INHIBITORY - always full baseline
    add_inh(None,       18, 0.00095, 22.0, on_soma=True)
    add_inh(SR_dist,    10, 0.00099,  5.9)
    add_inh(SLM,         8, 0.00037, 13.0)
    add_inh(apic_0_300, 42, 0.000115, 0.7)
    add_inh(SLM,        14, 0.000115, 6.7)

    t_vec  = h.Vector().record(h._ref_t)
    v_soma = h.Vector().record(cell.soma[0](0.5)._ref_v)

    h.celsius = 34; h.v_init = -70; h.tstop = 1000
    h.finitialize(h.v_init); h.continuerun(h.tstop)

    v = list(v_soma)
    spikes = sum(1 for j in range(1, len(v)) if v[j-1] < 0 and v[j] >= 0)
    rate = spikes / (h.tstop / 1000.0)
    return spikes, rate, list(t_vec), v

# ------------------------------------------------------------
# Run the sweep
# ------------------------------------------------------------
print("Excitatory active-fraction sweep (inhibition at full baseline):")
print("Target: 1-3 Hz (Mizuseki & Buzsaki 2013)\n")

results = []
for frac in EXC_FRACTIONS:
    spikes, rate, t, v = build_and_run(frac)
    n_exc = max(1,int(round(180*frac)))*2 + max(1,int(round(20*frac))) + max(1,int(round(5*frac)))
    results.append((frac, n_exc, spikes, rate, t, v))
    print(f"  exc_frac={frac:<5} (~{n_exc} exc synapses)  ->  {spikes} spikes  =  {rate:.1f} Hz")

# ------------------------------------------------------------
# Plot all traces stacked
# ------------------------------------------------------------
fig, axes = plt.subplots(len(results), 1, figsize=(11, 2.2*len(results)), sharex=True)
for ax, (frac, n_exc, spikes, rate, t, v) in zip(axes, results):
    ax.plot(t, v, color="black", linewidth=0.6)
    ax.axhline(y=-70, color="gray", linestyle="--", linewidth=0.6)
    in_range = "  <-- in 1-3 Hz range" if 1.0 <= rate <= 3.0 else ""
    ax.set_title(f"exc_frac={frac} (~{n_exc} exc)  ->  {rate:.1f} Hz{in_range}", fontsize=10)
    ax.set_ylabel("Vm (mV)")
axes[-1].set_xlabel("Time (ms)")
fig.suptitle("Excitatory active-fraction sweep — finding realistic CA1 firing (1-3 Hz target)")
plt.tight_layout()
plt.show()

print("\nPick the fraction whose rate falls in 1-3 Hz. That is the tuned baseline.")
