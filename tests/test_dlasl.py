import pytest
from pyasl.dlasl import concept_testing


def test_concept_testing():
    assert concept_testing() == 0
    assert concept_testing() == 0
    with pytest.raises(TypeError):
        concept_testing("not an integer")
