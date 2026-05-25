import unittest

from evaluation.run_image_patch_experiments import _build_patch_placement, _to_core_data_path
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ImagePatchRunnerTests(unittest.TestCase):
    def test_to_core_data_path_returns_repo_relative_data_path(self) -> None:
        image_path = PROJECT_ROOT / "data" / "man_with_phone.jpg"
        self.assertEqual(_to_core_data_path(image_path), "man_with_phone.jpg")

    def test_on_object_placement_centers_patch(self) -> None:
        placement = _build_patch_placement(
            target_info={"x": 320, "y": 240, "bbox": [300, 220, 340, 260]},
            patch_size=64,
            placement_profile={"mode": "on_object"},
        )
        self.assertEqual(len(placement), 1)
        self.assertTrue(placement[0]["is_center"])
        self.assertEqual(placement[0]["x"], 320)
        self.assertEqual(placement[0]["y"], 240)

    def test_near_object_right_anchor_moves_patch_outside_bbox(self) -> None:
        placement = _build_patch_placement(
            target_info={"x": 320, "y": 240, "bbox": [300, 220, 340, 260]},
            patch_size=64,
            placement_profile={"mode": "near_object", "anchor": "right", "gap_ratio": 0.1},
        )
        self.assertEqual(len(placement), 1)
        self.assertTrue(placement[0]["is_center"])
        self.assertGreater(placement[0]["x"], 340)


if __name__ == "__main__":
    unittest.main()
