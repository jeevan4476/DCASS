# tests/test_analysis/test_steganalysis_detectors.py
"""
Smoke tests for the steganalysis detector zoo (src/analysis/steganalysis_detectors.py).

The Component 1 benchmark tests exercise the benchmark machinery only — no
passing run anywhere instantiated the detectors or their building blocks.
The C1 record is a chance-level negative, so nothing here claims detection
strength: each test pins construction on CPU, a small valid input shape
through eval-mode forward, and the exact elementary algebra where it is
cheap to know it (ABS, TLU, quantize, pooling, SE gating, residual wiring).
"""

from __future__ import annotations

import pytest
import torch

from src.analysis.steganalysis_detectors import (
    ABS,
    TLU,
    ConvBn,
    CooccurrenceFeatures,
    GlobalAvgPool2d,
    MBConvBlock,
    QuantizeTruncate,
    SEBlock,
    SRMFilterBank,
    SRMFLDA,
    SpatialPyramidPooling,
    Type1,
    Type2,
    Type3,
    Type4,
    DETECTORS,
    get_detector,
)

H = W = 16


def _image(batch: int = 2, channels: int = 1) -> torch.Tensor:
    """Deterministic small grayscale-ish input valid for every detector."""
    torch.manual_seed(7)
    return torch.rand(batch, channels, H, W)


class TestElementaryBlocks:
    def test_abs_is_exactly_tensor_abs(self):
        x = torch.tensor([-2.0, -0.5, 0.0, 3.0])
        assert torch.equal(ABS()(x), x.abs())

    def test_tlu_clamps_to_threshold(self):
        tlu = TLU(threshold=3.0)
        x = torch.tensor([-5.0, -3.0, -0.5, 0.5, 3.0, 5.0])
        expected = torch.tensor([-3.0, -3.0, -0.5, 0.5, 3.0, 3.0])
        assert torch.equal(tlu(x), expected)

    def test_global_avg_pool2d_spatial_to_channels(self):
        x = torch.rand(2, 3, 4, 5)
        out = GlobalAvgPool2d()(x)
        assert out.shape == (2, 3)
        assert torch.allclose(out, x.mean(dim=(2, 3)))

    def test_quantize_truncate_rounds_and_clips(self):
        qt = QuantizeTruncate(T=2)
        x = torch.tensor([0.4, 2.6, -3.2])
        assert torch.equal(qt(x), torch.tensor([0.0, 2.0, -2.0]))

    def test_quantize_truncate_step_keeps_values_in_range(self):
        qt = QuantizeTruncate(T=5)
        x = torch.tensor([0.4, 2.6, -3.2])
        assert torch.equal(qt(x), torch.tensor([0.0, 3.0, -3.0]))

    def test_srm_filter_bank_shape_and_finiteness(self):
        srm = SRMFilterBank()
        assert tuple(srm.filters.shape) == (30, 1, 5, 5)
        out = srm(_image())
        assert out.shape == (2, 30, H, W)
        assert torch.isfinite(out).all()

    def test_spatial_pyramid_pooling_21c(self):
        spp = SpatialPyramidPooling()  # pool sizes [1, 2, 4]
        x = torch.ones(2, 5, 8, 8)
        out = spp(x)
        assert out.shape == (2, 5 * 21)
        assert torch.equal(out, torch.ones(2, 5 * 21))

    def test_se_block_zero_weights_halves_input(self):
        torch.manual_seed(0)
        x = torch.rand(2, 6, 4, 4)
        block = SEBlock(channels=6, reduction=3).eval()
        with torch.no_grad():
            block.fc1.weight.zero_()
            block.fc2.weight.zero_()
        # sigmoid(0) = 0.5 gate, elementwise.
        assert torch.equal(block(x), 0.5 * x)


class TestResidualBlocks:
    def test_convbn_and_type1_type3_type4_shapes(self):
        x = torch.rand(2, 8, H, W)
        assert ConvBn(8, 16)(x).shape == (2, 16, H, W)
        assert Type1(8, 16)(x).shape == (2, 16, H, W)
        assert Type3(8, 16)(x).shape == (2, 16, 8, 8)
        assert Type4(8, 16)(x).shape == (2, 16, 1, 1)

    def test_type2_is_residual_identity_with_zero_convs(self):
        torch.manual_seed(0)
        x = torch.rand(2, 8, H, W)
        block = Type2(8, 8).eval()
        with torch.no_grad():
            block.type1.convbn.conv.weight.zero_()
            block.convbn.conv.weight.zero_()
        assert torch.equal(block(x), x)

    def test_mbconv_stride2_downsamples(self):
        torch.manual_seed(0)
        x = torch.rand(2, 8, H, W)
        out = MBConvBlock(
            in_channels=8, out_channels=16, expand_ratio=6, stride=2, kernel_size=5
        )(x)
        assert out.shape == (2, 16, 8, 8)
        assert torch.isfinite(out).all()

    def test_mbconv_residual_identity_with_zero_projection(self):
        torch.manual_seed(0)
        x = torch.rand(2, 8, H, W)
        block = MBConvBlock(
            in_channels=8, out_channels=8, expand_ratio=6, stride=1, kernel_size=3
        ).eval()
        assert block.use_residual is True  # stride 1 + in == out flips it on
        with torch.no_grad():
            block.project_conv.weight.zero_()
        assert torch.equal(block(x), x)


class TestCooccurrenceFeatures:
    def test_row_sums_equal_the_pair_count(self):
        torch.manual_seed(2)
        x = torch.randint(-2, 3, (2, 30, 4, 4)).float()
        features = CooccurrenceFeatures(T=2)(x)
        # 30 filters x [(0,1): 12, (1,0): 12, (1,1): 9, (1,-1): 9] pairs.
        assert features.shape == (2, 3000)
        assert torch.allclose(
            features.sum(dim=1), torch.full((2,), 1260.0)
        )

    def test_constant_input_piles_into_one_bin_per_direction(self):
        x = torch.zeros(1, 30, 4, 4)
        row = CooccurrenceFeatures(T=2)(x)[0]
        # One occupied bin per (filter, direction) pair: 30 * 4 of them.
        assert int((row != 0).sum()) == 30 * 4
        values = {float(v) for v in row.unique().tolist()}
        assert values <= {0.0, 9.0, 12.0}
        assert {9.0, 12.0} <= values


class TestSrmFlda:
    def test_unfitted_forward_returns_features(self):
        model = SRMFLDA().eval()
        out = model(_image())
        assert out.shape == (2, 3000)
        assert torch.isfinite(out).all()
        assert model.is_fitted is False

    def test_fit_flda_then_forward_gives_symmetric_logits(self):
        torch.manual_seed(1)
        n = 40
        feats0 = torch.randn(n, 3000)
        feats0[:, 0] += 5.0
        feats1 = torch.randn(n, 3000)
        feats1[:, 0] -= 5.0
        features = torch.cat([feats0, feats1])
        labels = torch.cat([torch.zeros(n), torch.ones(n)]).long()

        model = SRMFLDA()
        model.fit_flda(features, labels)
        assert model.is_fitted is True

        out = model(_image(batch=8))
        # forward concatenates [-decision, decision]: antisymmetric logits.
        assert out.shape == (8, 2)
        assert torch.equal(out[:, 1], -out[:, 0])


class TestDetectorRegistry:
    def test_registry_contains_all_seven(self):
        assert set(DETECTORS) == {
            "srm_flda",
            "xu_net",
            "ye_net",
            "zhu_net",
            "sr_net",
            "msr_net",
            "efficientnet_stega",
        }

    @pytest.mark.parametrize(
        "name",
        sorted(name for name in DETECTORS if name != "srm_flda"),
        ids=sorted(name for name in DETECTORS if name != "srm_flda"),
    )
    def test_registered_neural_detectors_forward(self, name):
        model = get_detector(name).eval()
        with torch.no_grad():
            out = model(_image())
        assert out.shape == (2, 2)
        assert torch.isfinite(out).all()

    def test_unfitted_srm_flda_via_registry_returns_features(self):
        out = get_detector("srm_flda")(_image())
        assert out.shape == (2, 3000)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown detector"):
            get_detector("absolutely_not_a_detector")
