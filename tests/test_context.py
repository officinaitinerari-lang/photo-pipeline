from pathlib import Path

import pytest

from src.image_loader import build_context, detect_analog, orientation_of

ROOT = Path("/data/input")
KW = ["analog", "film", "35mm", "pellicola", "kodak", "ilford"]


def test_folder_and_filename_context():
    p = ROOT / "Giappone_Nagano_Inverno" / "tempio_neve.jpg"
    assert build_context(p, ROOT) == "Giappone Nagano Inverno - tempio neve"


def test_nested_folders_are_joined():
    p = ROOT / "Italia" / "Dolomiti-Estate" / "lago.di.braies.tif"
    assert build_context(p, ROOT) == "Italia / Dolomiti Estate - lago di braies"


@pytest.mark.parametrize("name", ["DSC_0423.NEF", "IMG_1234.jpg", "_DSF8812.dng", "PXL_20240101_123456.jpg"])
def test_camera_default_names_are_ignored(name):
    assert build_context(ROOT / "Islanda_Vik" / name, ROOT) == "Islanda Vik"


def test_file_in_root_without_folder():
    assert build_context(ROOT / "porto_di_genova.png", ROOT) == "porto di genova"
    assert build_context(ROOT / "DSC_0001.jpg", ROOT) == ""


@pytest.mark.parametrize(
    "rel,expected",
    [
        ("Scansioni_35mm/roma.tif", True),
        ("Viaggi/KodakPortra400_Lisbona/tram.jpg", True),
        ("pellicola/x.jpg", True),
        ("Ilford HP5/porto.tif", True),
        ("Viaggi/Lisbona/tram.jpg", False),
        ("Digitale/analogue_look.jpg", True),
    ],
)
def test_analog_detection(rel, expected):
    assert detect_analog(ROOT / rel, ROOT, KW) is expected


def test_orientation():
    assert orientation_of(6000, 4000) == "landscape"
    assert orientation_of(4000, 6000) == "portrait"
    assert orientation_of(5000, 5000) == "landscape"
