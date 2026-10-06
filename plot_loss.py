import re, sys
import matplotlib.pyplot as plt

log = open(sys.argv[1]).read()
out = sys.argv[2] if len(sys.argv) > 2 else "loss.png"

it = [(int(a), float(b)) for a, b in re.findall(r"iter (\d+): loss ([\d.]+)", log)]
ev = [(int(a), float(b), float(c)) for a, b, c in
      re.findall(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)", log)]

plt.figure(figsize=(8, 5))
plt.plot([e[0] for e in ev], [e[1] for e in ev], "o-", label="train loss (eval)")
plt.plot([e[0] for e in ev], [e[2] for e in ev], "o-", label="val loss (eval)")
plt.xlabel("iteration"); plt.ylabel("loss"); plt.legend(); plt.grid(alpha=0.3)
plt.title("Baseline (train_shakespeare_char)")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("best val loss:", min(e[2] for e in ev))