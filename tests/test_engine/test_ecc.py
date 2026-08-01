# tests/test_engine/test_ecc.py
"""
Unit tests for Reed-Solomon Error Correction Code (RS-ECC) in DCASS.
Verifies 100% bit-exact secret recovery (0% Bit Error Rate) under simulated vector noise.
"""

import pytest
from src.engine.ecc import RSErrorCorrection
from src.engine.encoder import SemanticEncoder
from src.engine.decoder import SemanticDecoder

def test_rs_ecc_basic_encoding_decoding():
    """Test basic RS-ECC encoding and error recovery."""
    ecc = RSErrorCorrection(parity_bytes=8)  # Can fix up to 4 byte errors
    message = "Covert Meeting at 0400 Hours"
    
    # 1. Encode
    codeword = ecc.encode(message)
    assert len(codeword) == len(message.encode("utf-8")) + 8

    # 2. Introduce 3 byte errors (simulating FAISS vector noise)
    corrupted = bytearray(codeword)
    corrupted[1] ^= 0xAA
    corrupted[5] ^= 0x55
    corrupted[12] ^= 0xFF

    # 3. Decode
    decoded_str, is_success, errors_fixed = ecc.decode(bytes(corrupted))
    assert is_success is True
    assert decoded_str == message
    assert len(errors_fixed) == 3
    print("✅ RS-ECC Unit Test Passed: 3 corrupted bytes fixed successfully!")

def test_rs_ecc_uncorrupted():
    """Test RS-ECC decoding when no errors occur."""
    ecc = RSErrorCorrection(parity_bytes=6)
    message = "Steganography Protocol"
    
    codeword = ecc.encode(message)
    decoded_str, is_success, errors_fixed = ecc.decode(codeword)
    
    assert is_success is True
    assert decoded_str == message
    assert len(errors_fixed) == 0

if __name__ == "__main__":
    test_rs_ecc_basic_encoding_decoding()
    test_rs_ecc_uncorrupted()
