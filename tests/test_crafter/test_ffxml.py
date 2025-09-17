from imolcraft.crafter.ffxml import check_vsite
import os
import pytest


@pytest.mark.parametrize(
    "xmlfile, n_vsite",
    [
        (os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "vsite_average2.xml"), 2),
        (os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "vsite_average3.xml"), 1)
    ]
)
def test_check_vsite(xmlfile, n_vsite):
    assert check_vsite(xmlfile) == n_vsite
