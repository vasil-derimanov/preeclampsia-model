"""Feature definitions, constants, and column names for the PE prediction model."""

# --- Feature column names by category ---

MATERNAL_CHARACTERISTICS = [
    "maternal_age",
    "height",
    "pre_pregnancy_weight",
    "nulliparous",
    "conception_natural",
    "conception_ovulation_induction",
    "conception_ivf_et",
    "family_history_pe",
    "smoking",
]

MEDICAL_HISTORY = [
    "history_pe",
    "history_chronic_hypertension",
    "history_chronic_kidney_disease",
    "history_diabetes",
    "history_sle_aps",
]

BIOPHYSICAL_MARKERS = [
    "map_mmhg",
    "uta_pi",
]

BIOCHEMICAL_MARKERS = [
    "papp_a",
    "plgf",
]

ALL_FEATURES = MATERNAL_CHARACTERISTICS + MEDICAL_HISTORY + BIOPHYSICAL_MARKERS + BIOCHEMICAL_MARKERS

# Column used before one-hot encoding for conception method
CONCEPTION_METHOD_RAW = "conception_method"

# Columns that are one-hot encoded from conception_method
CONCEPTION_METHOD_COLUMNS = [
    "conception_natural",
    "conception_ovulation_induction",
    "conception_ivf_et",
]

# --- Target columns ---

TARGETS = {
    "pe_all": "pe_all",
    "preterm_pe": "preterm_pe",
    "term_pe": "term_pe",
}

# --- Feature set combinations (incremental, matching paper) ---

FEATURE_SETS = {
    "maternal_only": MATERNAL_CHARACTERISTICS + MEDICAL_HISTORY,
    "maternal_map": MATERNAL_CHARACTERISTICS + MEDICAL_HISTORY + ["map_mmhg"],
    "maternal_map_pappa": MATERNAL_CHARACTERISTICS + MEDICAL_HISTORY + ["map_mmhg", "papp_a"],
    "maternal_map_pappa_utapi": MATERNAL_CHARACTERISTICS + MEDICAL_HISTORY + ["map_mmhg", "papp_a", "uta_pi"],
    "full": ALL_FEATURES,
}

# --- Population statistics from Table 1 (for synthetic data generation) ---

# Non-PE population (N=4434) — continuous variables: (mean, std)
NON_PE_STATS = {
    "maternal_age": (29.63, 3.26),
    "height": (162.01, 4.78),
    "pre_pregnancy_weight": (56.97, 8.50),
    "map_mmhg": (82.62, 7.29),
    "uta_pi": (1.80, 0.47),
    "plgf": (33.81, 46.25),
    "papp_a": (4.56, 2.63),
}

# PE population (N=210) — continuous variables: (mean, std)
PE_STATS = {
    "maternal_age": (30.18, 3.74),
    "height": (162.00, 4.74),
    "pre_pregnancy_weight": (62.36, 11.11),
    "map_mmhg": (92.89, 11.25),
    "uta_pi": (1.81, 0.59),
    "plgf": (25.75, 12.07),
    "papp_a": (3.61, 2.37),
}

# Preterm PE population (N=49) — continuous variables: (mean, std)
PRETERM_PE_STATS = {
    "maternal_age": (31.02, 4.42),
    "height": (161.52, 4.23),
    "pre_pregnancy_weight": (62.81, 10.66),
    "map_mmhg": (95.71, 13.68),
    "uta_pi": (1.95, 0.61),
    "plgf": (24.87, 12.27),
    "papp_a": (3.07, 2.64),
}

# Binary feature prevalences — (non_pe_rate, pe_rate, preterm_pe_rate)
BINARY_PREVALENCES = {
    "nulliparous": (0.7756, 0.8143, 0.6327),
    "history_pe": (0.0063, 0.0905, 0.2449),
    "history_diabetes": (0.0061, 0.0286, 0.0408),
    "history_chronic_hypertension": (0.0056, 0.1429, 0.2449),
    "history_chronic_kidney_disease": (0.0, 0.0, 0.0),  # not reported, assume rare
    "family_history_pe": (0.0068, 0.0095, 0.0408),
    "history_sle_aps": (0.0101, 0.0048, 0.0),
    "smoking": (0.0023, 0.0048, 0.0),
}

# Assisted reproduction rate (ovulation induction + IVF-ET combined)
ASSISTED_REPRODUCTION_RATE = {
    "non_pe": 0.1024,
    "pe": 0.2048,
    "preterm_pe": 0.1224,
}

# --- Study parameters ---

N_TOTAL = 4644
N_PE = 210
N_PRETERM_PE = 49
N_TERM_PE = 161
PE_INCIDENCE = N_PE / N_TOTAL  # ~4.5%
