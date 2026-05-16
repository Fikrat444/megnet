"""
predict_bandgap_mf2020.py
-------------------------
Band gap prediction using MEGNet Multi-Fidelity 2020 models.
  - Cs2NaInCl6  -- Materials Project mp-989571
  - Cs2InAgCl6  -- Materials Project (a=10.68 A)

Multi-fidelity model trains on 4 levels of data quality:
  0 = PBE        (cheap DFT)
  1 = GLLB-SC    (improved DFT functional)
  2 = HSE        (hybrid DFT, more accurate)
  3 = Experiment (measured values)

6 models are loaded (different random data splits).
Final prediction = mean of 6 models.
Uncertainty      = standard deviation of 6 models.

Usage:
    python predict_bandgap_mf2020.py
"""

import os
import warnings
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
from megnet.models import MEGNetModel
from megnet.layers import _CUSTOM_OBJECTS
from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GaussianDistance
from megnet.utils.preprocessing import DummyScaler
from pymatgen.core import Structure, Lattice

# ── Model loading ─────────────────────────────────────────────────────────────

MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "mvl_models", "mf_2020", "pbe_gllb_hse_exp"
)

# MF-2020 models: cutoff = 5 Angstrom (larger than MP-2018 which used 4 A)
graph_converter = CrystalGraph(
    cutoff=5,
    bond_converter=GaussianDistance(np.linspace(0, 6, 100), 0.5)
)

def load_model(path):
    """Load a MF-2020 MEGNet model without recompiling."""
    model = tf.keras.models.load_model(
        path,
        custom_objects=_CUSTOM_OBJECTS,
        compile=False
    )
    wrapper = MEGNetModel.__new__(MEGNetModel)
    wrapper.model = model
    wrapper.graph_converter = graph_converter
    wrapper.target_scaler = DummyScaler()
    wrapper.dropout_on_predict = False
    return wrapper


print("=" * 65)
print("  MEGNet Multi-Fidelity 2020 Band Gap Prediction")
print("=" * 65)
print("\nLoading 6 MF models (pbe_gllb_hse_exp)...")

all_models = [
    load_model(os.path.join(MODEL_DIR, str(i), "best_model.hdf5"))
    for i in range(6)
]
print(f"✓ {len(all_models)} models loaded")

# ── Fidelity levels ───────────────────────────────────────────────────────────

FIDELITY_LEVELS = {
    0: "PBE",
    1: "GLLB-SC",
    2: "HSE",
    3: "Experiment",
}

# ── Structure builder ─────────────────────────────────────────────────────────

def build_fm3m_double_perovskite(a, B_site, Bprime_site, cl_x, fidelity=3):
    """
    Build a Fm-3m (#225) double perovskite: Cs2 B B' Cl6

    Wyckoff positions:
        4a  -> B        (0,     0,   0)
        4b  -> B'       (1/2,   0,   0)
        8c  -> Cs       (1/4, 1/4, 3/4)
        24e -> Cl       (cl_x,  0,   0)

    fidelity: 0=PBE, 1=GLLB-SC, 2=HSE, 3=Experiment
    """
    lattice = Lattice.cubic(a)
    species = (
        [B_site]      * 4 +
        [Bprime_site] * 4 +
        ["Cs"]        * 8 +
        ["Cl"]        * 24
    )
    frac_coords = [
        # B -- 4a
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
        # B' -- 4b
        [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.5, 0.5, 0.5],
        # Cs -- 8c
        [0.25, 0.25, 0.75], [0.75, 0.75, 0.75],
        [0.75, 0.25, 0.25], [0.25, 0.75, 0.25],
        [0.75, 0.75, 0.25], [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75], [0.75, 0.25, 0.75],
        # Cl -- 24e (cl_x, 0, 0) and symmetry equivalents
        [cl_x,   0.0,     0.0    ], [1-cl_x, 0.0,     0.0    ],
        [0.0,    cl_x,    0.0    ], [0.0,    1-cl_x,  0.0    ],
        [0.0,    0.0,     cl_x   ], [0.0,    0.0,     1-cl_x ],
        [0.5,    cl_x,    0.5    ], [0.5,    1-cl_x,  0.5    ],
        [cl_x,   0.5,     0.5    ], [1-cl_x, 0.5,     0.5    ],
        [0.5,    0.5,     cl_x   ], [0.5,    0.5,     1-cl_x ],
        [cl_x,   0.5,     0.0    ], [1-cl_x, 0.5,     0.0    ],
        [0.5,    cl_x,    0.0    ], [0.5,    1-cl_x,  0.0    ],
        [0.0,    0.5,     cl_x   ], [0.0,    0.5,     1-cl_x ],
        [cl_x,   0.0,     0.5    ], [1-cl_x, 0.0,     0.5    ],
        [0.0,    cl_x,    0.5    ], [0.0,    1-cl_x,  0.5    ],
        [0.5,    0.0,     cl_x   ], [0.5,    0.0,     1-cl_x ],
    ]
    struct = Structure(lattice, species, frac_coords)
    struct.state = [fidelity]
    return struct

# ── Define compounds ──────────────────────────────────────────────────────────

COMPOUNDS = {
    "Cs2NaInCl6": {
        "a": 10.76, "B_site": "In", "Bprime_site": "Na", "cl_x": 0.237725,
        "mp_id": "mp-989571", "lit_eg": 3.7,
    },
    "Cs2InAgCl6": {
        "a": 10.68, "B_site": "In", "Bprime_site": "Ag", "cl_x": 0.240636,
        "mp_id": "MP screenshot", "lit_eg": 3.3,
    },
}

# ── Helper: predict with all 6 models ────────────────────────────────────────

def predict(struct):
    """Run structure through all 6 models, return mean and std."""
    preds = [float(m.predict_structure(struct)[0]) for m in all_models]
    return float(np.mean(preds)), float(np.std(preds)), preds

# ── Section 1: Experimental fidelity results ──────────────────────────────────

print("\n\n>>> SECTION 1: Experimental Fidelity (fidelity=3) Results")
print("=" * 65)
print(f"{'Compound':<16} {'Mean Eg':>10} {'Std Dev':>10} {'Category':>12}")
print("=" * 65)

exp_results = {}
for name, info in COMPOUNDS.items():
    struct = build_fm3m_double_perovskite(
        info["a"], info["B_site"], info["Bprime_site"], info["cl_x"], fidelity=3
    )
    mean, std, preds = predict(struct)
    cat = "Insulator" if mean > 3.0 else ("Semiconductor" if mean > 0.5 else "Metal")
    exp_results[name] = {"mean": mean, "std": std, "preds": preds}
    print(f"{name:<16} {mean:>9.4f} eV {std:>9.4f} eV {cat:>12}")

print("=" * 65)

# ── Section 2: All fidelity levels comparison ─────────────────────────────────

print("\n\n>>> SECTION 2: All Fidelity Levels Comparison")

for name, info in COMPOUNDS.items():
    print(f"\n  {name}  ({info['mp_id']})  |  Literature: ~{info['lit_eg']} eV")
    print(f"  {'Fidelity':<14} {'Mean Eg':>10} {'Std Dev':>10}")
    print(f"  {'-'*36}")
    for fid, fid_name in FIDELITY_LEVELS.items():
        struct = build_fm3m_double_perovskite(
            info["a"], info["B_site"], info["Bprime_site"], info["cl_x"],
            fidelity=fid
        )
        mean, std, _ = predict(struct)
        marker = "  <-- closest to lit." if abs(mean - info["lit_eg"]) < 0.5 else ""
        print(f"  {fid_name:<14} {mean:>9.4f} eV {std:>9.4f} eV{marker}")

print("Notes:")
print("  - MF-2020 trained on PBE + GLLB-SC + HSE + Experimental data")
print("  - fidelity=3 asks the model to predict at experimental level")
print("  - High std dev = higher uncertainty across the 6 model ensemble")
print("  - High std dev = higher uncertainty across the 6 model ensemble")