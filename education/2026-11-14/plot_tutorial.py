"""Original static teaching graphic generated only from saved arithmetic."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "results.json").read_text(encoding="utf-8"))
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "svg.hashsalt": "score-semantics-2026-11-14"})
fig = plt.figure(figsize=(10.8, 13.8), dpi=100, facecolor="#f6f4ed")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
NAVY, TEAL, MUTED = "#173748", "#087d7a", "#45616c"


def text(x, y, value, size=20, color=NAVY, weight="normal", **kwargs):
    ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
            ha="left", va="top", transform=ax.transAxes, **kwargs)


def card(y, height):
    ax.add_patch(FancyBboxPatch((.045, y), .91, height,
                               boxstyle="round,pad=0.012,rounding_size=0.012",
                               linewidth=1, edgecolor="#d6e1df", facecolor="white"))


text(.055, .966, "Score scale can change.\nThe ranking can stay.", 30, weight="bold", linespacing=1.15)
text(.055, .881, "Authored synthetic arithmetic • no model fit or clicks", 17, MUTED)

card(.48, .35)
text(.065, .815, "1  Same user, same three candidates", 22, TEAL, "bold")
text(.065, .773, "User vector [1, 2]     Score s = user · item", 20)
cols = [.075, .49, .66, .835]
for x, label in zip(cols, ["Item · vector", "s", "10s + 7", "Rank"]):
    text(x, .726, label, 18, MUTED, "bold")
for index, row in enumerate(data["rows"]):
    y = .686 - index * .055
    ax.add_patch(FancyBboxPatch((.063, y-.043), .87, .045,
                                boxstyle="round,pad=0.003", linewidth=0,
                                facecolor="#e7f3ef" if index == 0 else "#f5f7f6"))
    vector = ", ".join(f"{v:g}" for v in row["item_vector"])
    values = [f'{row["item"]}  [{vector}]', f'{row["score"]:.1f}',
              f'{row["transformed_score"]:.0f}', str(index+1)]
    for x, value in zip(cols, values):
        text(x, y-.003, value, 21, weight="bold" if index == 0 else "normal")
text(.075, .515, "A > B > C in both score columns", 23, TEAL, "bold")

card(.197, .247)
text(.065, .429, "2  BPR models a pairwise preference", 22, TEAL, "bold")
text(.065, .389, "Pair form:  σ(s_A − s_B),  where σ is sigmoid", 20)
pair = data["pair_A_over_B"]
text(.075, .343, "Original scores", 18, MUTED, "bold")
text(.535, .343, "Rescaled scores", 18, MUTED, "bold")
text(.075, .308, f'0.8 − 0.3 = {pair["margin"]:.1f}', 23)
text(.535, .308, f'15 − 10 = {pair["transformed_margin"]:.0f}', 23)
text(.075, .269, f'σ(0.5) = {pair["sigmoid_margin"]:.4f}', 24, weight="bold")
text(.535, .269, f'σ(5) = {pair["transformed_sigmoid_margin"]:.4f}', 24, weight="bold")
text(.075, .225, "Order stays fixed; margins and training loss change.", 17, MUTED)

text(.06, .164, "A pair-preference model does not establish P(click).", 20, weight="bold")
text(.06, .129, "Before displaying a percentage", 22, TEAL, "bold")
text(.06, .091, "Define the event, time window and exposure.\nFit/check calibration on separate outcome data.",
     20, linespacing=1.4)

fig.savefig(ROOT / "score-vs-probability.png", dpi=100, facecolor=fig.get_facecolor())
fig.savefig(ROOT / "score-vs-probability.svg", facecolor=fig.get_facecolor(),
            metadata={"Date": None, "Creator": "Matplotlib"})
plt.close(fig)
# Remove generated SVG path-attribute trailing whitespace while keeping tokens unchanged.
svg = ROOT / "score-vs-probability.svg"
svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n",
               encoding="utf-8", newline="\n")
print("Generated original 1080x1380 PNG and editable SVG from results.json")
