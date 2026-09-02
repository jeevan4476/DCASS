from src.engine.dssc_state_space import family_for_cluster, DEFAULT_SEMANTIC_FAMILIES


def test_family_for_cluster_nature():
    fam = family_for_cluster(0)
    assert fam.name == "nature_outdoor"


def test_family_for_cluster_urban():
    fam = family_for_cluster(42)
    assert fam.name == "urban_architecture"


def test_family_for_cluster_sound():
    fam = family_for_cluster(255)
    assert fam.name == "sound_atmosphere"


def test_family_for_cluster_boundary_last_nature():
    fam = family_for_cluster(41)
    assert fam.name == "nature_outdoor"


def test_family_for_cluster_fallback():
    # cluster_id out of any range → first family (safe fallback)
    fam = family_for_cluster(9999)
    assert fam == DEFAULT_SEMANTIC_FAMILIES[0]
