import pytest
import aemHMF as amf
import numpy

def test_test():
    assert hasattr(amf, "emu")
    assert hasattr(amf, "residual_gp")
    assert hasattr(amf, "tinkerMF")
