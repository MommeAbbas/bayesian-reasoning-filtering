import numpy as np
import matplotlib.pyplot as plt

# Load results
results = np.load("particle_sweep_results.npy", allow_pickle=True).item()


def summarize_metric(method, metric):
    xs = []
    means = []
    stds = []
    for Np in sorted(results[method].keys()):
        vals = [m[metric] for m in results[method][Np]["metrics"]]
        xs.append(Np)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return np.array(xs), np.array(means), np.array(stds)


def summarize_runtime(method):
    xs = []
    means = []
    stds = []
    for Np in sorted(results[method].keys()):
        vals = results[method][Np]["runtime_per_traj"]
        xs.append(Np)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return np.array(xs), np.array(means), np.array(stds)



plt.figure(figsize=(6, 4))
for method in ["RBPF", "PF"]:
    xs, ys, errs = summarize_metric(method, "auc_early")
    plt.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=method)

plt.xscale("log")
plt.xlabel("Number of particles")
plt.ylabel("Early AUC (k = 5)")
plt.title("Early prediction performance vs particle count")
plt.legend()
plt.tight_layout()
plt.savefig("fig_auc_vs_particles.png", dpi=300)
plt.show()



# Plot 2: NLL vs particles
plt.figure(figsize=(6, 4))
for method in ["RBPF", "PF"]:
    xs, ys, errs = summarize_metric(method, "nll")
    plt.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=method)

plt.xscale("log")
plt.xlabel("Number of particles")
plt.ylabel("Negative Log-Likelihood")
plt.title("Probabilistic quality vs particle count")
plt.legend()
plt.tight_layout()
plt.savefig("fig_nll_vs_particles.png", dpi=300)
plt.show()



# Plot 3: Runtime vs particles
plt.figure(figsize=(6, 4))
for method in ["RBPF", "PF"]:
    xs, ys, errs = summarize_runtime(method)
    plt.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=method)

plt.xscale("log")
plt.xlabel("Number of particles")
plt.ylabel("Runtime per trajectory (seconds)")
plt.title("Runtime vs particle count")
plt.legend()
plt.tight_layout()
plt.savefig("fig_runtime_vs_particles.png", dpi=300)
plt.show()

