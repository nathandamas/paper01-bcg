from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def test_reference_figures_meet_template_envelope_and_resolution():
    for number in range(1, 8):
        path = ROOT / "reference" / "figures" / f"Figure_{number}.png"
        with Image.open(path) as image:
            width, height = image.size
            dpi = image.info.get("dpi", (0, 0))
        assert width <= 2138
        assert height <= 2551
        assert min(dpi) >= 299

