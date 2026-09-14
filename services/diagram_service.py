# -------------------------------------------------------------- #
#   Diagram & Formula Service                                      #
#   Recovered from agent/tools.py (v2 original)                    #
#   Provides: diagram generation, diagram detection, formula search#
# -------------------------------------------------------------- #

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)


# -------------------------------------------------------------- #
#   Diagram Detection (deterministic keyword matching)             #
# -------------------------------------------------------------- #

DIAGRAM_KEYWORDS = {
    "velocity_time": ["বেগ-সময়", "v-t graph", "velocity time", "বেগ ও সময়"],
    "displacement_time": ["দূরত্ব-সময়", "s-t graph", "displacement time", "সরণ-সময়"],
    "wave": ["তরঙ্গ", "wave", "শব্দতরঙ্গ", "আলোর তরঙ্গ"],
    "force_diagram": ["বল চিত্র", "force diagram", "বল প্রয়োগ", "নিউটন"],
    "projectile": ["প্রাস", "projectile", "কৌণিক বেগ", "আনত নিক্ষেপ"],
    "ray_diagram": [
        "আঁকো", "draw", "figure draw", "ray diagram", "কিরণ চিত্র",
        "রশ্মি চিত্র", "আয়না চিত্র", "লেন্স চিত্র", "figure gula",
        "চিত্র আঁক", "diagram আঁক", "mirror diagram", "আয়নার চিত্র",
        "রে ডায়াগ্রাম", "figure আঁক",
    ],
}


def detect_diagram_type(query: str) -> Optional[str]:
    """Query থেকে কোন ধরনের diagram দরকার সেটা detect করে।"""
    query_lower = query.lower()
    for diagram_type, keywords in DIAGRAM_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return diagram_type
    return None


# -------------------------------------------------------------- #
#   Diagram Generation (matplotlib — dark theme)                   #
# -------------------------------------------------------------- #

def generate_physics_diagram(diagram_type: str, params: dict) -> Optional[bytes]:
    """Physics concept এর জন্য matplotlib diagram generate করে।
    Returns PNG bytes or None on failure.

    diagram_type options:
    - "velocity_time": v-t graph
    - "displacement_time": s-t graph
    - "force_diagram": simple force vector
    - "wave": wave pattern
    - "projectile": projectile motion
    - "ray_diagram": convex mirror ray diagram
    """
    try:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#4a4a8a")

        t = np.linspace(0, params.get("t_max", 10), 300)

        if diagram_type == "velocity_time":
            u = params.get("initial_velocity", 0)
            a = params.get("acceleration", 2)
            v = u + a * t
            ax.plot(t, v, color="#00d4ff", linewidth=2.5, label=f"v = {u} + {a}t")
            ax.set_xlabel("সময় (s)")
            ax.set_ylabel("বেগ (m/s)")
            ax.set_title("বেগ-সময় লেখচিত্র (v-t graph)")
            ax.axhline(y=0, color="gray", linewidth=0.5)
            ax.legend(facecolor="#1a1a2e", labelcolor="white")

        elif diagram_type == "displacement_time":
            u = params.get("initial_velocity", 0)
            a = params.get("acceleration", 1)
            s = u * t + 0.5 * a * t**2
            ax.plot(t, s, color="#ff6b9d", linewidth=2.5, label=f"s = {u}t + ½({a})t²")
            ax.set_xlabel("সময় (s)")
            ax.set_ylabel("দূরত্ব (m)")
            ax.set_title("দূরত্ব-সময় লেখচিত্র (s-t graph)")
            ax.legend(facecolor="#1a1a2e", labelcolor="white")

        elif diagram_type == "wave":
            amplitude = params.get("amplitude", 1)
            freq = params.get("frequency", 1)
            x = np.linspace(0, 4 * np.pi / freq, 500)
            y = amplitude * np.sin(freq * x)
            ax.plot(x, y, color="#c3f584", linewidth=2.5)
            ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_xlabel("দূরত্ব (m)")
            ax.set_ylabel("সরণ (m)")
            ax.set_title(f"তরঙ্গ (বিস্তার={amplitude}, কম্পাঙ্ক={freq} Hz)")
            ax.annotate("তরঙ্গদৈর্ঘ্য (λ)", xy=(2*np.pi/freq, 0),
                        xytext=(np.pi/freq, amplitude*0.5),
                        arrowprops=dict(arrowstyle="->", color="yellow"),
                        color="yellow", fontsize=9)

        elif diagram_type == "force_diagram":
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect("equal")

            circle = plt.Circle((0, 0), 0.3, color="#00d4ff", zorder=5)
            ax.add_patch(circle)
            ax.text(0, 0, "বস্তু", ha="center", va="center",
                    color="black", fontsize=8, fontweight="bold", zorder=6)

            forces = params.get("forces", [
                {"name": "ওজন (W)", "dx": 0, "dy": -1.5, "color": "#ff6b9d"},
                {"name": "স্বাভাবিক বল (N)", "dx": 0, "dy": 1.5, "color": "#c3f584"},
            ])
            for force in forces:
                ax.annotate("",
                    xy=(force["dx"], force["dy"]),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=force["color"],
                                   lw=2, mutation_scale=20))
                ax.text(force["dx"] * 1.1 + 0.1, force["dy"] * 1.1,
                        force["name"], color=force["color"], fontsize=9)

            ax.set_title("বল চিত্র (Force Diagram)")
            ax.set_xticks([])
            ax.set_yticks([])

        elif diagram_type == "projectile":
            v0 = params.get("initial_velocity", 20)
            angle_deg = params.get("angle", 45)
            angle = np.radians(angle_deg)
            g = 9.8
            t_flight = 2 * v0 * np.sin(angle) / g
            t_proj = np.linspace(0, t_flight, 300)
            x = v0 * np.cos(angle) * t_proj
            y = v0 * np.sin(angle) * t_proj - 0.5 * g * t_proj**2
            ax.plot(x, y, color="#ffd700", linewidth=2.5)
            ax.fill_between(x, y, alpha=0.1, color="#ffd700")
            ax.axhline(y=0, color="gray", linewidth=1)
            t_max = v0 * np.sin(angle) / g
            h_max = v0**2 * np.sin(angle)**2 / (2 * g)
            ax.plot(v0 * np.cos(angle) * t_max, h_max, "ro", markersize=8)
            ax.annotate(f"সর্বোচ্চ উচ্চতা\n{h_max:.1f} m",
                        xy=(v0 * np.cos(angle) * t_max, h_max),
                        xytext=(v0 * np.cos(angle) * t_max * 0.6, h_max * 0.8),
                        arrowprops=dict(arrowstyle="->", color="red"),
                        color="red", fontsize=9)
            ax.set_xlabel("অনুভূমিক দূরত্ব (m)")
            ax.set_ylabel("উচ্চতা (m)")
            ax.set_title(f"প্রাসের গতি (v₀={v0} m/s, θ={angle_deg}°)")

        elif diagram_type == "ray_diagram":
            ax.set_xlim(-4, 4)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect("equal")
            ax.set_facecolor("#0d1117")
            fig.patch.set_facecolor("#0d1117")

            theta = np.linspace(np.pi * 0.6, np.pi * 1.4, 80)
            r = 3.0
            mx = r * np.cos(theta)
            my = r * np.sin(theta)
            ax.plot(mx, my, color="#4fc3f7", linewidth=3, label="উত্তল আয়না")

            ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")

            f = 1.5
            ax.plot(-f, 0, "o", color="#ffd54f", markersize=7, zorder=5)
            ax.text(-f, -0.25, "F", color="#ffd54f", ha="center", fontsize=10)
            ax.plot(-2*f, 0, "s", color="#ef9a9a", markersize=6, zorder=5)
            ax.text(-2*f, -0.25, "C", color="#ef9a9a", ha="center", fontsize=10)

            ax.annotate("", xy=(2.5, 1.0), xytext=(2.5, 0),
                        arrowprops=dict(arrowstyle="->", color="#a5d6a7", lw=2))
            ax.text(2.5, 1.1, "বস্তু", color="#a5d6a7", ha="center", fontsize=9)

            ax.annotate("", xy=(0, 1.0), xytext=(2.5, 1.0),
                        arrowprops=dict(arrowstyle="->", color="#ff8a65", lw=1.5))
            ax.annotate("", xy=(-f, 0), xytext=(0, 1.0),
                        arrowprops=dict(arrowstyle="->", color="#ff8a65", lw=1.5,
                                        linestyle="dashed"))

            ax.annotate("", xy=(0, 0.5), xytext=(2.5, 1.0),
                        arrowprops=dict(arrowstyle="->", color="#ce93d8", lw=1.5))
            ax.annotate("", xy=(2.5, 0.5), xytext=(0, 0.5),
                        arrowprops=dict(arrowstyle="->", color="#ce93d8", lw=1.5,
                                        linestyle="dashed"))

            ax.plot([-0.7, -0.7], [0, 0.6], color="#80deea",
                    linewidth=1.5, linestyle=":", alpha=0.8)
            ax.text(-0.7, 0.7, "প্রতিবিম্ব\n(অাভাসী)", color="#80deea",
                    ha="center", fontsize=8)

            ax.set_title("উত্তল আয়নায় রশ্মি চিত্র (Convex Mirror Ray Diagram)",
                        color="white", fontsize=11, pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color="#ff8a65", lw=2, label="আপতিত রশ্মি ১"),
                Line2D([0], [0], color="#ce93d8", lw=2, label="আপতিত রশ্মি ২"),
                Line2D([0], [0], color="#80deea", lw=1.5,
                       linestyle=":", label="প্রতিফলিত রশ্মি (বর্ধিত)"),
            ]
            ax.legend(handles=legend_elements, facecolor="#1a1a2e",
                     labelcolor="white", fontsize=8, loc="lower right")

        else:
            plt.close(fig)
            return None

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception as e:
        logger.error(f"Diagram generation error: {e}")
        plt.close("all")
        return None
