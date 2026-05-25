import importlib
import unittest


MODULES = [
    "api.core_client",
    "api.model_functions",
    "utils.config",
    "utils.config_validator",
    "utils.io_utils",
    "utils.path_utils",
    "attacks.base_attacks",
    "attacks.coords_extractor",
    "attacks.bb.bb_attacks",
    "attacks.bb.video_attacks",
    "evaluation.attack_presets",
    "evaluation.attack_eval",
    "evaluation.generate_attack_presentation",
    "evaluation.run_image_patch_experiments",
    "evaluation.video_attack_eval",
    "training.learned_patch",
    "training.prepare_patch_data",
    "scripts.run_all_defenses",
    "scripts.run_full",
    "scripts.run_single_pixels_attack",
]


class ImportSmokeTests(unittest.TestCase):
    def test_key_modules_import(self) -> None:
        for module_name in MODULES:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
