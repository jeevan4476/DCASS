# tests/test_engine/test_ecc.py
"""
Unit tests for Reed-Solomon Error Correction Code (RS-ECC) in DCASS.
Verifies 100% bit-exact secret recovery (0% Bit Error Rate) under simulated vector noise.
"""

from src.engine.ecc import RSErrorCorrection

def test_rs_ecc_basic_encoding_decoding():
    """Test basic RS-ECC encoding and error recovery."""
    ecc = RSErrorCorrection(parity_bytes=8)  # Can fix up to 4 byte errors
    message = "Reliable transmission test block"

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


def test_rs_ecc_known_erasures_up_to_parity():
    """Known erasure positions allow full recovery up to R dropped bytes (and reject R+1)."""
    ecc = RSErrorCorrection(parity_bytes=8)
    assert ecc.max_correctable_erasures == 8
    message = "Reed-Solomon erasure decoding recovers dropped packets."

    codeword = ecc.encode(message)
    erase_positions = [0, 3, 7, 11, 15, 19, 23, 27]
    erased = bytearray(codeword)
    for pos in erase_positions:
        erased[pos] = 0

    decoded_str, ok, fixed = ecc.decode(bytes(erased), erase_pos=erase_positions)
    assert ok is True
    assert decoded_str == message
    assert set(erase_positions).issubset(set(fixed))

    # 9 erasures (> R=8) must fail cleanly
    over_positions = erase_positions + [31]
    erased_over = bytearray(codeword)
    for pos in over_positions:
        erased_over[pos] = 0
    _, ok_over, _ = ecc.decode(bytes(erased_over), erase_pos=over_positions)
    assert ok_over is False


def test_rs_ecc_mixed_errors_and_erasures():
    """Mixed unknown errors (e) and known erasures (v) succeed when 2*e + v <= R."""
    ecc = RSErrorCorrection(parity_bytes=8)
    message = "Combined error and erasure correction test payload."
    codeword = bytearray(ecc.encode(message))

    erase_pos = [2, 8, 14, 20]  # v = 4
    error_pos = [5, 25]         # e = 2 -> 2*2 + 4 = 8 <= 8
    for p in erase_pos:
        codeword[p] = 0
    for p in error_pos:
        codeword[p] ^= 0xA5

    decoded_str, ok, fixed = ecc.decode(bytes(codeword), erase_pos=erase_pos)
    assert ok is True
    assert decoded_str == message
    assert set(erase_pos + error_pos).issubset(set(fixed))

if __name__ == "__main__":
    test_rs_ecc_basic_encoding_decoding()
    test_rs_ecc_uncorrupted()
