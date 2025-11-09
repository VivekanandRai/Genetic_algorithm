"""
fitness_functions.py
Defines the biological response models:
- effectiveness(dose_A, dose_B, dose_C)
- side_effects(dose_A, dose_B, dose_C)
"""

import numpy as np

def effectiveness(doses):
    """
    Compute total effectiveness for doses [A, B, C].
    Returns scalar or array depending on input shape.
    """
    doses = np.asarray(doses)
    A, B, C = doses[..., 0], doses[..., 1], doses[..., 2]

    eff_A = 80 * np.exp(-((A - 50)**2) / 400.0)
    eff_B = 70 * np.exp(-((B - 40)**2) / 500.0)
    eff_C = 60 * np.exp(-((C - 30)**2) / 300.0)

    return eff_A + eff_B + eff_C


def side_effects(doses):
    """
    Compute side effect cost for doses [A, B, C].
    Returns scalar or array depending on input shape.
    """
    doses = np.asarray(doses)
    return 0.05 * np.sum(doses**2, axis=-1)
