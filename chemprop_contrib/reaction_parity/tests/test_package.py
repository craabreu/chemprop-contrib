def test_package_is_importable_from_namespace():
    from chemprop_contrib import reaction_parity

    assert {
        "ParityCGRFeaturizer",
        "reverse_bmg",
        "ParityEquivariantMPNN",
        "build_parity_mask",
        "ODD",
        "EVEN",
        "FREE",
    } <= set(reaction_parity.__all__)
    for name in reaction_parity.__all__:
        assert hasattr(reaction_parity, name)


def test_package_is_listed_in_contrib_namespace():
    import chemprop_contrib

    assert "reaction_parity" in chemprop_contrib.__all__
