import unittest

from attacks.bb.video_attacks import VideoBlackBoxAttacks
from evaluation.video_attack_eval import VideoAttackEvaluator


class VideoAttackUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.video_attacks = VideoBlackBoxAttacks()
        self.video_evaluator = VideoAttackEvaluator()

    def test_temporal_mode_always_attacks_every_frame(self) -> None:
        for frame_index in range(8):
            with self.subTest(frame=frame_index):
                self.assertTrue(
                    self.video_attacks._should_apply_attack(
                        frame_index,
                        temporal_mode="always",
                    )
                )

    def test_temporal_mode_flicker_uses_expected_pattern(self) -> None:
        pattern = [
            self.video_attacks._should_apply_attack(
                frame_index,
                temporal_mode="flicker",
                flicker_period=4,
                flicker_active_ratio=0.5,
            )
            for frame_index in range(8)
        ]
        self.assertEqual(pattern, [True, True, False, False, True, True, False, False])

    def test_invalid_temporal_mode_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.video_attacks._should_apply_attack(0, temporal_mode="invalid")

    def test_extract_analysis_metrics_normalizes_missing_values(self) -> None:
        metrics = self.video_evaluator._extract_analysis_metrics(None)
        self.assertEqual(metrics["interval_count"], 0)
        self.assertEqual(metrics["total_time_with_phone"], 0.0)
        self.assertEqual(metrics["detection_ratio"], 0.0)
        self.assertEqual(metrics["avg_phone_confidence"], 0.0)

    def test_extract_analysis_metrics_uses_phone_intervals(self) -> None:
        metrics = self.video_evaluator._extract_analysis_metrics(
            {
                "total_time_with_phone": 12.5,
                "detection_ratio": 0.4,
                "total_frames_processed": 90,
                "duration_seconds": 30.0,
                "intervals": [
                    {"avg_phone_confidence": 0.8},
                    {"avg_phone_confidence": 0.6},
                ],
            }
        )
        self.assertEqual(metrics["interval_count"], 2)
        self.assertEqual(metrics["total_time_with_phone"], 12.5)
        self.assertEqual(metrics["detection_ratio"], 0.4)
        self.assertEqual(metrics["avg_phone_confidence"], 0.7)


if __name__ == "__main__":
    unittest.main()
