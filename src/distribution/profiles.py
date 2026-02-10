# src/distribution/profiles.py

ACTIVITY_PROFILES = {
    "casual": {
        "skip_prob": 0.15,
        "jitter_range": (-1, 3),
        "idle_gap_prob": 0.3,
        "idle_gap_range": (8, 20)
    },

    "steady": {
        "skip_prob": 0.05,
        "jitter_range": (0, 1),
        "idle_gap_prob": 0.05,
        "idle_gap_range": (3, 6)
    },

    "bursty": {
        "skip_prob": 0.2,
        "jitter_range": (-2, 2),
        "idle_gap_prob": 0.4,
        "idle_gap_range": (15, 40)
    },

    "night_owl": {
        "skip_prob": 0.1,
        "jitter_range": (-3, 4),
        "idle_gap_prob": 0.5,
        "idle_gap_range": (20, 60)
    },

    "debug": {
        "skip_prob":  0.0,
        "jitter_range": (0, 0),
        "idle_gap_prob": 0.0,
        "idle_gap_range": (0, 0)
}

}
