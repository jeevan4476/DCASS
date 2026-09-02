# tests/test_engine/test_context.py
"""
Tests for Dynamic Context Keying (H-12).

Verifies that:
1. Permutations are deterministic, bijective, and secret-dependent.
2. A message encoded with a context key decodes exactly with the same key.
3. Decoding with a DIFFERENT epoch (or secret) does not silently succeed.
4. The epoch hint short-circuits candidate search.
"""

import time

import numpy as np
import pytest

from src.engine.context import ContextKeyManager


class TestPermutationDerivation:
    def test_permutation_is_bijective(self):
        mgr = ContextKeyManager(bucket_seconds=3600)
        perm = mgr.derive_permutation(mgr.current_epoch())
        assert sorted(perm.tolist()) == list(range(256))

    def test_permutation_deterministic_across_instances(self):
        ts = 1_750_000_000.0
        a = ContextKeyManager(bucket_seconds=3600).derive_permutation(
            ContextKeyManager(bucket_seconds=3600).epoch_at(ts)
        )
        b = ContextKeyManager(bucket_seconds=3600).derive_permutation(
            ContextKeyManager(bucket_seconds=3600).epoch_at(ts)
        )
        np.testing.assert_array_equal(a, b)

    def test_different_buckets_give_different_permutations(self):
        mgr = ContextKeyManager(bucket_seconds=3600)
        p1 = mgr.derive_permutation(mgr.epoch_at(1_750_000_000.0))
        p2 = mgr.derive_permutation(mgr.epoch_at(1_750_000_000.0 + 3600))
        assert not np.array_equal(p1, p2)

    def test_secret_changes_permutation(self):
        ts = 1_750_000_000.0
        p_plain = ContextKeyManager(bucket_seconds=3600).derive_permutation(
            ContextKeyManager(bucket_seconds=3600).epoch_at(ts)
        )
        p_keyed = ContextKeyManager(bucket_seconds=3600, secret=b"hunter2").derive_permutation(
            ContextKeyManager(bucket_seconds=3600, secret=b"hunter2").epoch_at(ts)
        )
        assert not np.array_equal(p_plain, p_keyed)

    def test_inverse_restores_identity(self):
        mgr = ContextKeyManager()
        epoch = mgr.current_epoch()
        inv = mgr.derive_inverse_permutation(epoch)
        perm = mgr.derive_permutation(epoch)
        np.testing.assert_array_equal(inv[perm], np.arange(256))
        np.testing.assert_array_equal(perm[inv], np.arange(256))

    def test_candidate_epochs_cover_boundary_skew(self):
        mgr = ContextKeyManager(bucket_seconds=3600)
        boundary = int(time.time() // 3600) * 3600
        epochs = mgr.candidate_epochs(boundary - 10)  # just before rollover
        ids = [e.epoch_id for e in epochs]
        # Must include both the pre-boundary and post-boundary buckets.
        before = mgr.epoch_at(boundary - 3600).epoch_id
        current = mgr.epoch_at(boundary).epoch_id
        assert current in ids and before in ids

    def test_invalid_bucket_rejected(self):
        with pytest.raises(ValueError):
            ContextKeyManager(bucket_seconds=0)


class TestContextualCodecRoundtrip:
    """End-to-end through the real encoder/decoder using a fake corpus."""

    @pytest.fixture()
    def codec(self):
        from tests.test_engine.test_exact_vcp_recovery import FakeIndex
        from src.engine.encoder import SemanticEncoder
        from src.engine.decoder import SemanticDecoder

        index = FakeIndex()
        encoder = SemanticEncoder(index=index)
        encoder._loaded = True
        decoder = SemanticDecoder(index=index)
        decoder._loaded = True
        return encoder, decoder

    def _mgr(self, secret=None):
        return ContextKeyManager(bucket_seconds=3600, secret=secret)

    def test_roundtrip_with_context_key(self, codec):
        encoder, decoder = codec
        mgr = self._mgr()
        msg = "Keyed channel works"

        result = encoder.encode(msg, use_ecc=True, context_manager=mgr)
        assert result.context_info.get("epoch_id")

        decoded = decoder.decode(
            result.media_ids,
            use_ecc=True,
            context_manager=mgr,
            context_epoch_hint=result.context_info["epoch_id"],
        )
        assert decoded.ecc_success
        assert decoded.context_epoch_id == result.context_info["epoch_id"]
        assert decoded.reconstructed_meaning == msg

    def test_wrong_secret_fails_closed(self, codec):
        """A receiver without the right secret must NOT get the message."""
        encoder, decoder = codec
        sender_mgr = self._mgr(secret=b"right-secret")
        msg = "Classified rendezvous"

        result = encoder.encode(
            msg, use_ecc=True, context_manager=sender_mgr
        )

        wrong_mgr = self._mgr(secret=b"wrong-secret")
        decoded = decoder.decode(
            result.media_ids,
            use_ecc=True,
            context_manager=wrong_mgr,
        )
        # Either ECC fails or the text is garbage; never the original.
        assert not (decoded.ecc_success and decoded.reconstructed_meaning == msg), (
            "Decoded original message without the correct context secret"
        )

    def test_no_context_manager_fails_on_keyed_traffic(self, codec):
        """Plain decode of keyed traffic must not silently recover it."""
        encoder, decoder = codec
        mgr = self._mgr(secret=b"k")
        result = encoder.encode(
            "Hidden in plain sight",
            use_ecc=True,
            context_manager=mgr,
        )
        decoded = decoder.decode(result.media_ids, use_ecc=True)
        assert not (
            decoded.ecc_success and decoded.reconstructed_meaning == "Hidden in plain sight"
        )

    def test_epoch_hint_bypasses_candidate_search(self, codec):
        encoder, decoder = codec
        mgr = self._mgr()
        result = encoder.encode(
            "hint path", use_ecc=True, context_manager=mgr
        )
        decoded = decoder.decode(
            result.media_ids,
            use_ecc=True,
            context_manager=mgr,
            context_epoch_hint=result.context_info["epoch_id"],
        )
        assert decoded.ecc_success
        assert decoded.reconstructed_meaning == "hint path"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
