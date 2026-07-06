# ============================================================
# synapse_sweep_both.py
# Active-fraction sweep reducing BOTH excitatory AND inhibitory
# together by the same fraction (keeps E/I ratio constant).
#
# Compare against synapse_sweep.py (excitatory-only) to see which
# tuning approach gives a more realistic trace at the target rate.
#
# All conductances and rates from the project parameter table.
# NMDA off. Target: physiological 1-3 Hz (Mizuseki & Buzsaki 2013).
# ============================================================

from neuron import h, gui
import matplotlib.pyplot as plt
import random

# fractions applied to BOTH excitatory and inhibitory
FRACTIONS = [1.0, 0.5, 0.25, 0.1, 0.05]

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

def build_and_run(frac):
    random.seed(1)
    keep = []
    def scaled(n):
        return max(1, int(round(n * frac)))

    def add_exc(sections, n, gampa_max, rate_hz):
        if not sections: return
        interval = 1000.0 / rate_hz
        for _ in range(n):
            syn = h.MeMo_AmpaNmda(random.choice(sections)(random.random()))
            syn.gampa_max = gampa_max; syn.gnmda_max = 0.0; syn.ampatau = 3.0
            stim = h.NetStim(); stim.number = 100000; stim.start = 100
            stim.interval = interval; stim.noise = 1
            nc = h.NetCon(stim, syn); nc.weight[0] = 1
            keep.extend([syn, stim, nc])

    def add_inh(sections, n, g_max, rate_hz, on_soma=False):
        interval = 1000.0 / rate_hz
        for _ in range(n):
            seg = cell.soma[0](0.5) if on_soma else random.choice(sections)(random.random())
            syn = h.MeMo_GABAA(seg); syn.g_max = g_max
            stim = h.NetStim(); stim.number = 100000; stim.start = 100
            stim.interval = interval; stim.noise = 1
            nc = h.NetCon(stim, syn); nc.weight[0] = 1
            keep.extend([syn, stim, nc])

    # BOTH scaled by the same fraction
    add_exc(SR_prox, scaled(180), 0.00256, 10)
    add_exc(SR_dist, scaled(180), 0.00796, 10)
    add_exc(SLM,     scaled(20),  0.00150,  1)
    add_exc(SLM,     scaled(5),   0.00134,  5)

    add_inh(None,       scaled(18), 0.00095, 22.0, on_soma=True)
    add_inh(SR_dist,    scaled(10), 0.00099,  5.9)
    add_inh(SLM,        scaled(8),  0.00037, 13.0)
    add_inh(apic_0_300, scaled(42), 0.000115, 0.7)
    add_inh(SLM,        scaled(14), 0.000115, 6.7)

    t_vec  = h.Vector().record(h._ref_t)
    v_soma = h.Vector().record(cell.soma[0](0.5)._ref_v)
    h.celsius = 34; h.v_init = -70; h.tstop = 1000
    h.finitialize(h.v_init); h.continuerun(h.tstop)

    v = list(v_soma)
    spikes = sum(1 for j in range(1, len(v)) if v[j-1] < 0 and v[j] >= 0)
    n_exc = scaled(180)*2 + scaled(20) + scaled(5)
    n_inh = scaled(18)+scaled(10)+scaled(8)+scaled(42)+scaled(14)
    return spikes, spikes, n_exc, n_inh, list(t_vec), v

print("Sweep reducing BOTH excitatory and inhibitory (E/I ratio constant):")
print("Target: 1-3 Hz (Mizuseki & Buzsaki 2013)\n")

results = []
for frac in FRACTIONS:
    spikes, _, n_exc, n_inh, t, v = build_and_run(frac)
    rate = spikes / 1.0
    results.append((frac, n_exc, n_inh, spikes, rate, t, v))
    print(f"  frac={frac:<5} (~{n_exc} exc, ~{n_inh} inh)  ->  {spikes} spikes = {rate:.1f} Hz")

fig, axes = plt.subplots(len(results), 1, figsize=(11, 2.2*len(results)), sharex=True)
for ax, (frac, n_exc, n_inh, spikes, rate, t, v) in zip(axes, results):
    ax.plot(t, v, color="black", linewidth=0.6)
    ax.axhline(y=-70, color="gray", linestyle="--", linewidth=0.6)
    flag = "  <-- in 1-3 Hz" if 1.0 <= rate <= 3.0 else ""
    ax.set_title(f"frac={frac} (~{n_exc} exc, ~{n_inh} inh) -> {rate:.1f} Hz{flag}", fontsize=10)
    ax.set_ylabel("Vm (mV)")
axes[-1].set_xlabel("Time (ms)")
fig.suptitle("Sweep reducing BOTH E and I together (constant E/I ratio)")
plt.tight_layout()
plt.show()

print("\nCompare to the excitatory-only sweep to decide which tuning is more realistic.")
