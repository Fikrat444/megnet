"""
predict_bandgap.py
------------------
Band gap prediction using MEGNet pre-trained models.
  - Cs2NaInCl6  -- Materials Project mp-989571
  - Cs2InAgCl6  -- Materials Project (a=10.68 A)

Usage:
    python predict_bandgap.py
"""

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from megnet.models import MEGNetModel
from megnet.layers import _CUSTOM_OBJECTS
from megnet.data.crystal import CrystalGraph
from megnet.data.graph import GaussianDistance
from megnet.utils.preprocessing import DummyScaler
from pymatgen.core import Structure, Lattice

# ── Model loading ─────────────────────────────────────────────────────────────

# Path to pre-trained models inside the megnet repo
MODEL_BASE = os.path.join(os.path.dirname(__file__), "mvl_models", "mp-2018.6.1")

# MP-2018 models: 100 Gaussian bond features, cutoff = 4 Angstrom
graph_converter = CrystalGraph(
    cutoff=4,
    bond_converter=GaussianDistance(np.linspace(0, 5, 100), 0.5)
)

def load_model(filename):
    """Load a saved MEGNet model without recompiling (avoids Keras version issues)."""
    model = tf.keras.models.load_model(
        filename,
        custom_objects=_CUSTOM_OBJECTS,
        compile=False
    )
    wrapper = MEGNetModel.__new__(MEGNetModel)
    wrapper.model = model
    wrapper.graph_converter = graph_converter
    wrapper.target_scaler = DummyScaler()
    wrapper.dropout_on_predict = False
    return wrapper


print("=" * 60)
print("  MEGNet Band Gap Prediction")
print("=" * 60)
print("\nLoading models...")

reg = load_model(os.path.join(MODEL_BASE, "band_gap_regression.hdf5"))
cls = load_model(os.path.join(MODEL_BASE, "band_classification.hdf5"))

print("✓ band_gap_regression.hdf5")
print("✓ band_classification.hdf5")


# ── Structure builder ─────────────────────────────────────────────────────────

def build_fm3m_double_perovskite(a, B_site, Bprime_site, cl_x):
    """
    Build a Fm-3m (#225) double perovskite: Cs2 B B' Cl6

    Wyckoff positions:
        4a  -> B        (0,     0,   0)
        4b  -> B'       (1/2,   0,   0)
        8c  -> Cs       (1/4, 1/4, 3/4)
        24e -> Cl       (cl_x,  0,   0)
    """
    lattice = Lattice.cubic(a)

    species = (
        [B_site]      * 4 +   # 4a
        [Bprime_site] * 4 +   # 4b
        ["Cs"]        * 8 +   # 8c
        ["Cl"]        * 24    # 24e
    )

    frac_coords = [
        # B -- 4a
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        # B' -- 4b
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.5],
        # Cs -- 8c
        [0.25, 0.25, 0.75],
        [0.75, 0.75, 0.75],
        [0.75, 0.25, 0.25],
        [0.25, 0.75, 0.25],
        [0.75, 0.75, 0.25],
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        # Cl -- 24e (cl_x, 0, 0) and symmetry equivalents
        [cl_x,   0.0,     0.0    ],
        [1-cl_x, 0.0,     0.0    ],
        [0.0,    cl_x,    0.0    ],
        [0.0,    1-cl_x,  0.0    ],
        [0.0,    0.0,     cl_x   ],
        [0.0,    0.0,     1-cl_x ],
        [0.5,    cl_x,    0.5    ],
        [0.5,    1-cl_x,  0.5    ],
        [cl_x,   0.5,     0.5    ],
        [1-cl_x, 0.5,     0.5    ],
        [0.5,    0.5,     cl_x   ],
        [0.5,    0.5,     1-cl_x ],
        [cl_x,   0.5,     0.0    ],
        [1-cl_x, 0.5,     0.0    ],
        [0.5,    cl_x,    0.0    ],
        [0.5,    1-cl_x,  0.0    ],
        [0.0,    0.5,     cl_x   ],
        [0.0,    0.5,     1-cl_x ],
        [cl_x,   0.0,     0.5    ],
        [1-cl_x, 0.0,     0.5    ],
        [0.0,    cl_x,    0.5    ],
        [0.0,    1-cl_x,  0.5    ],
        [0.5,    0.0,     cl_x   ],
        [0.5,    0.0,     1-cl_x ],
    ]

    return Structure(lattice, species, frac_coords)


# ── Define compounds ──────────────────────────────────────────────────────────

compounds = {
    "Cs2NaInCl6": {
        "structure": build_fm3m_double_perovskite(
            a=10.76, B_site="In", Bprime_site="Na", cl_x=0.237725
        ),
        "mp_id":  "mp-989571",
        "lit_eg": 3.7,   # experimental / literature band gap (eV)
    },
    "Cs2InAgCl6": {
        "structure": build_fm3m_double_perovskite(
            a=10.68, B_site="In", Bprime_site="Ag", cl_x=0.240636
        ),
        "mp_id":  "MP screenshot",
        "lit_eg": 3.3,   # experimental / literature band gap (eV)
    },
}

# ── Predict and print results ─────────────────────────────────────────────────

print("\nRunning predictions...\n")
print("=" * 60)
print(f"{'Compound':<16} {'Eg MEGNet':>10} {'Classifier':>12} {'Category':>13}")
print("=" * 60)

results = {}
for name, data in compounds.items():
    struct  = data["structure"]
    eg      = float(reg.predict_structure(struct)[0])
    c       = float(cls.predict_structure(struct)[0])
    cat     = "Insulator" if eg > 3.0 else ("Semiconductor" if eg > 0.5 else "Metal")
    cls_str = "Non-metal" if c > 0.5 else "Metal"
    results[name] = eg
    print(f"{name:<16} {eg:>10.4f} eV {cls_str:>12}  {cat:>13}")

print("=" * 60)

# ── Comparison with literature ────────────────────────────────────────────────

print("\nComparison with literature:")
print("-" * 60)
for name, data in compounds.items():
    eg  = results[name]
    lit = data["lit_eg"]
    print(f"  {name}  ({data['mp_id']})")
    print(f"    MEGNet          : {eg:.4f} eV")
    print(f"    Corrected est.  : {eg*1.35:.2f} - {eg*1.50:.2f} eV  (x1.35-1.50)")
    print(f"    Literature      : ~{lit} eV")
    print()

# Model trained on GGA-PBE data -- systematically underestimates band gaps by ~30-40%
print("Note: GGA-PBE training data -> Eg underestimated by ~30-40%.")