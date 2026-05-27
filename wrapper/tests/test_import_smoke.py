import importlib
import unittest


MODULES = [
    "api.model_functions",
    "utils.config",
    "utils.config_validator",
    "utils.io_utils",
    "utils.path_utils",
    "attacks.base_attacks",
    "attacks.coords_extractor",
    "attacks.bb.bb_attacks",
    "attacks.bb.video_attacks",
    "defenses.defense",
    "defenses.attack_classifier",
    "defenses.adaptive_defense",
    "defenses.video_defences",
    "evaluation.attack_presets",
    "evaluation.attack_eval",
    "evaluation.generate_attack_presentation",
    "evaluation.plot_detection_ratio_drop",
    "evaluation.plot_attack_strength_heatmap",
    "evaluation.plot_defense_effectiveness_heatmap",
    "evaluation.run_image_patch_experiments",
    "evaluation.video_attack_eval",
    "scripts.run_all_defenses",
    "scripts.run_full",
]


class ImportSmokeTests(unittest.TestCase):
    def test_key_modules_import(self) -> None:
        for module_name in MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
