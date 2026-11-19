# tests/test_engine/test_context.py
"""
Tests for Dynamic Context Keying (H-12).

Verifies that:
1. Permutations are deterministic, bijective, and secret-dependent.
2. A message encoded with a context key decodes exactly with the same key.
3. Decoding with a DIFFERENT epoch (or secret) does not silently succeed.
4. The epoch hint short-circuits candidate search.
5. Keying options that cannot apply are rejected loudly, never silently dropped.
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

    def test_roundtrip_without_ecc_keeps_frame_out_of_message(self, codec):
        """Keyed decode without RS must strip framing, not expose frame bytes."""
        encoder, decoder = codec
        mgr = self._mgr(secret=b"no-ecc-secret")
        msg = "framed without RS"
        result = encoder.encode(msg, use_ecc=False, context_manager=mgr)

        decoded = decoder.decode(
            result.media_ids,
            use_ecc=False,
            context_manager=mgr,
            context_epoch_hint=result.context_info["epoch_id"],
        )
        assert decoded.ecc_success
        assert decoded.reconstructed_meaning == msg

    def test_roundtrip_without_ecc_or_hint(self, codec):
        """Framing integrity alone must validate the epoch during candidate search."""
        encoder, decoder = codec
        mgr = self._mgr(secret=b"no-ecc-secret")
        msg = "candidate epoch search"
        result = encoder.encode(msg, use_ecc=False, context_manager=mgr)

        decoded = decoder.decode(result.media_ids, use_ecc=False, context_manager=mgr)
        assert decoded.ecc_success
        assert decoded.reconstructed_meaning == msg

    def test_no_ecc_wrong_secret_fails_closed(self, codec):
        encoder, decoder = codec
        msg = "must not leak without RS"
        result = encoder.encode(
            msg, use_ecc=False, context_manager=self._mgr(secret=b"right-secret")
        )
        decoded = decoder.decode(
            result.media_ids,
            use_ecc=False,
            context_manager=self._mgr(secret=b"wrong-secret"),
        )
        assert not (decoded.ecc_success and decoded.reconstructed_meaning == msg), (
            "Recovered the message with the wrong secret and no RS protection"
        )

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

    def test_epoch_hint_without_context_manager_raises(self, codec):
        """A context_epoch_hint with dynamic context disabled must not be ignored."""
        _, decoder = codec
        with pytest.raises(ValueError, match="context_epoch_hint"):
            decoder.decode(
                ["any_media_id"],
                use_ecc=True,
                context_epoch_hint="123|3600|time",
            )

    def test_keyed_encode_reports_original_payload_byte(self, codec):
        """EncodedChunk.payload_byte is the message byte, not the permuted cluster.

        The decoder has always reported the pre-permutation payload byte and
        the transmitted cluster separately; the encoder must use the same
        convention so the API's two fields mean the same thing on both sides.
        """
        encoder, decoder = codec
        mgr = self._mgr(secret=b"byte-semantics")
        msg = "payload byte semantics"

        result = encoder.encode(msg, use_ecc=True, context_manager=mgr)
        assert result.payload_symbols

        assert [c.payload_byte for c in result.encoded] == list(result.payload_symbols)
        assert any(
            c.cluster_id != c.payload_byte for c in result.encoded
        ), "a real key should not leave cluster_id equal to payload_byte everywhere"

        decoded = decoder.decode(
            result.media_ids,
            use_ecc=True,
            context_manager=mgr,
            context_epoch_hint=result.context_info["epoch_id"],
        )
        assert decoded.ecc_success
        assert [d.payload_byte for d in decoded.decoded] == list(result.payload_symbols)
        assert [d.cluster_id for d in decoded.decoded] == [
            c.cluster_id for c in result.encoded
        ]


class TestExternalSourceDeterminism:
    """Epoch hints carrying external (non-"time") source material must resolve
    deterministically WITHOUT re-fetching the external source."""

    def _mgr(self, fetcher):
        return ContextKeyManager(
            bucket_seconds=3600,
            sources=("time", "coingecko"),
            _source_fetchers={"coingecko": fetcher},
        )

    def test_epoch_id_carries_inline_materials_and_resolves_without_refetch(self):
        calls = []

        def fetcher(timeout):
            calls.append(timeout)
            # Value changes on EVERY call (crosses $100 buckets): a re-fetch
            # during resolve would return a different bucket than encode-time.
            return f"btc_usd_{100 * len(calls)}"

        mgr = self._mgr(fetcher)
        epoch = mgr.epoch_at(1_750_000_000.0)
        assert len(calls) == 1
        assert epoch.source_materials["coingecko"] == "btc_usd_100"
        # A non-"time" source contributed material -> 4th "|" field present.
        assert epoch.epoch_id.count("|") == 3
        assert "coingecko=btc_usd_100" in epoch.epoch_id

        resolved = mgr.resolve_epoch_hint(epoch.epoch_id)
        assert len(calls) == 1, "resolve_epoch_hint must NOT re-fetch external sources"
        assert resolved.source_materials["coingecko"] == "btc_usd_100"
        assert resolved.epoch_id == epoch.epoch_id
        np.testing.assert_array_equal(
            mgr.derive_permutation(resolved), mgr.derive_permutation(epoch)
        )

    def test_legacy_three_part_hint_still_refetches_external_source(self):
        calls = []

        def fetcher(timeout):
            calls.append(timeout)
            return "btc_usd_650"

        mgr = self._mgr(fetcher)
        resolved = mgr.resolve_epoch_hint("1750000000|3600|time,coingecko")
        assert len(calls) == 1, "legacy 3-part hints re-fetch the named source"
        assert resolved.source_materials["coingecko"] == "btc_usd_650"
        # And the resolved epoch re-serializes with inline materials.
        again = mgr.resolve_epoch_hint(resolved.epoch_id)
        assert len(calls) == 1
        assert again.source_materials["coingecko"] == "btc_usd_650"

    def test_time_only_epoch_id_unchanged_three_parts(self):
        mgr = ContextKeyManager(bucket_seconds=3600)
        epoch = mgr.epoch_at(1_750_000_000.0)
        base = int(1_750_000_000 // 3600) * 3600
        assert epoch.epoch_id == f"{base}|3600|time"

    def test_candidate_epochs_fetches_external_source_once(self):
        calls = []

        def fetcher(timeout):
            calls.append(timeout)
            return "btc_usd_650"

        mgr = self._mgr(fetcher)
        base = int(1_750_000_000 // 3600) * 3600
        epochs = mgr.candidate_epochs(1_750_000_000.0, tolerance_buckets=1)
        assert len(calls) == 1, "fetcher must run once per call, not once per offset epoch"
        assert len(epochs) == 3
        assert {e.bucket_start for e in epochs} == {base - 3600, base, base + 3600}
        # Every offset epoch reuses the SAME fetched material.
        assert len({e.source_materials.get("coingecko") for e in epochs}) == 1

    def test_resolve_hint_rejects_inline_material_for_unlisted_source(self):
        mgr = self._mgr(lambda timeout: "btc_usd_650")
        with pytest.raises(ValueError):
            mgr.resolve_epoch_hint("1750000000|3600|time|coingecko=btc_usd_650")


class TestDomainSeparatedKeys:
    """Fix: permutation and frame-HMAC keys must be domain-separated subkeys."""

    def test_derived_subkeys_are_distinct_and_not_raw_secret(self):
        import hashlib
        import hmac

        mgr = ContextKeyManager(bucket_seconds=3600, secret=b"hunter2")
        perm_key = mgr.derive_permutation_key()
        frame_secret = mgr.derive_frame_secret()
        assert isinstance(perm_key, bytes) and isinstance(frame_secret, bytes)
        assert perm_key == hmac.new(
            b"hunter2", b"dcass:v1:context-permutation", hashlib.sha256
        ).digest()
        assert frame_secret == hmac.new(
            b"hunter2", b"dcass:v1:frame-hmac", hashlib.sha256
        ).digest()
        assert perm_key != frame_secret
        assert perm_key != mgr.secret
        assert frame_secret != mgr.secret
        # Raw secret is untouched (tests read .secret).
        assert mgr.secret == b"hunter2"
        # Deterministic across instances.
        mgr2 = ContextKeyManager(bucket_seconds=3600, secret=b"hunter2")
        assert mgr2.derive_permutation_key() == perm_key
        assert mgr2.derive_frame_secret() == frame_secret

    def test_no_secret_returns_none(self):
        mgr = ContextKeyManager(bucket_seconds=3600)
        assert mgr.derive_permutation_key() is None
        assert mgr.derive_frame_secret() is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
