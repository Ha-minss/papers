from careerrec.figures import generate_figures


def test_generate_figures_is_importable() -> None:
    assert callable(generate_figures)
