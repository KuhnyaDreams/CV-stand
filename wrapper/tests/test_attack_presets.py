import unittest

from evaluation.attack_presets import (
    get_attack_preset_params,
    load_attack_presets,
    merge_attack_params,
)


class AttackPresetTests(unittest.TestCase):
    def test_default_preset_file_loads(self) -> None:
        presets = load_attack_presets()
        self.assertIn("gaussian_blur_attack", presets)
        self.assertIn("motion_blur_attack", presets)

    def test_get_attack_preset_params_returns_requested_preset(self) -> None:
        presets = load_attack_presets()
        params = get_attack_preset_params("compression_attack", "medium", presets)
        self.assertEqual(params, {"jpeg_quality": 15})

    def test_merge_attack_params_prefers_explicit_overrides(self) -> None:
        merged = merge_attack_params(
            {"kernel_size": 9, "angle_degrees": 0},
            {"kernel_size": 17},
        )
        self.assertEqual(merged, {"kernel_size": 17, "angle_degrees": 0})


if __name__ == "__main__":
    unittest.main()
