# BB Attack Pipeline

This document describes the current workflow for black-box video attacks.

## 1. What the pipeline does

The pipeline has two stages:

1. Apply a local attack to the input video.
2. Evaluate the original and attacked video through `analyze_video_phone(...)`.

The evaluation target is not generic detection. The final metric comes from:

- `wrapper/model_functions.py`
- `core/app.py`
- `core/utils.py`

The business target is "person with phone" detection over time.

## 2. Main entrypoint

All normal video BB runs start from:

```powershell
cd B:\CV-stand\wrapper
python video_attack_eval.py --video ../data/test.mp4 ...
```

## 3. Before running attacks

Make sure:

1. Core is running:

```powershell
cd B:\CV-stand
docker-compose up -d
```

2. The input video exists, for example:

- `B:\CV-stand\data\test.mp4`

## 4. Single attack

### 4.1 Run one attack with manual parameters

Example:

```powershell
python video_attack_eval.py --video ../data/test.mp4 --attack compression_attack --jpeg-quality 20
```

### 4.2 Run one attack with a preset

Example:

```powershell
python video_attack_eval.py --video ../data/test.mp4 --attack motion_blur_attack --preset strong
```

### 4.3 Run one attack with preset plus manual override

Example:

```powershell
python video_attack_eval.py --video ../data/test.mp4 --attack patch_attack --preset stealth_medium --patch-size 128
```

## 5. Multiple attacks in one run

### 5.1 Run all available BB attacks

```powershell
python video_attack_eval.py --video ../data/test.mp4 --all
```

Current `--all` includes:

- `gaussian_blur_attack`
- `motion_blur_attack`
- `random_noise_attack`
- `low_light_attack`
- `brightness_attack`
- `contrast_attack`
- `compression_attack`
- `downscale_upscale_attack`
- `frame_drop_attack`
- `blackout_attack`
- `patch_attack`

### 5.2 Run a custom subset of attacks

```powershell
python video_attack_eval.py --video ../data/test.mp4 --attacks motion_blur_attack compression_attack patch_attack
```

Important:

- `--all` and `--attacks` run attacks with default parameters.
- If you want exact presets or parameter sweeps, use `--experiment-config`.

## 6. Comparing attacks and parameter variants

### 6.1 Run a comparison config

Example:

```powershell
python video_attack_eval.py --video ../data/test.mp4 --experiment-config ./bb_realistic_suite.json
```

This is the preferred mode for:

- comparing different attacks
- comparing the same attack with different strengths
- building a reproducible benchmark

### 6.2 Config structure

Each experiment entry contains:

- `name`
- `attack_name`
- optional `preset`
- optional `attack_params`

Example:

```json
{
  "name": "compression_strong",
  "attack_name": "compression_attack",
  "preset": "strong"
}
```

If both `preset` and `attack_params` are present:

- preset is loaded first
- explicit `attack_params` override preset values

## 7. Presets

Presets are stored in:

- `B:\CV-stand\wrapper\bb_attack_presets.json`

Use them like this:

```powershell
python video_attack_eval.py --video ../data/test.mp4 --attack low_light_attack --preset medium
```

## 8. Temporal mode

Many video attacks support:

- `temporal_mode=always`
- `temporal_mode=flicker`

CLI example:

```powershell
python video_attack_eval.py --video ../data/test.mp4 --attack gaussian_blur_attack --kernel-size 9 --temporal-mode flicker --flicker-period 6 --flicker-active-ratio 0.5
```

Meaning:

- attack part of the frames
- leave the rest unchanged

This helps test temporal instability, not only spatial corruption.

## 9. Where results are saved

### 9.1 Attacked videos

Saved in:

- `B:\CV-stand\data`

Examples:

- `attack_motion_blur_attack_....mp4`
- `attack_patch_attack_....mp4`

### 9.2 JSON and CSV reports

Saved in:

- `B:\CV-stand\results\video_attack_reports`

Typical outputs:

- detailed JSON report
- flat CSV comparison table

### 9.3 Core-side video analysis

Core may also write:

- `B:\CV-stand\results\video_analysis`

with `phone_analysis.json` files.

## 10. How to interpret results

Primary metric:

- `detection_ratio_drop`

Supporting metrics:

- `phone_time_drop`
- `phone_confidence_drop`
- `interval_count_drop`

Recommended reading:

1. `detection_ratio_drop`
2. `phone_time_drop`
3. `phone_confidence_drop`

Interpretation:

- larger `detection_ratio_drop` means a stronger attack against the target pipeline
- larger `phone_time_drop` means the system sees the phone for less time
- larger `phone_confidence_drop` means the model is less confident even when detections still exist

## 11. Practical command examples

### Compare Gaussian vs Motion Blur

```powershell
python video_attack_eval.py --video ../data/test.mp4 --experiment-config ./bb_realistic_suite.json
```

### Check one realistic low-light run

```powershell
python video_attack_eval.py --video ../data/test.mp4 --attack low_light_attack --preset medium
```

### Check one stealth patch

```powershell
python video_attack_eval.py --video ../data/test.mp4 --attack patch_attack --preset stealth_medium
```

### Check one frame-drop scenario

```powershell
python video_attack_eval.py --video ../data/test.mp4 --attack frame_drop_attack --preset medium
```

## 12. Recommended workflow

1. Start with one attack and one preset.
2. Move to `--experiment-config` for comparison.
3. Read CSV first for ranking.
4. Open JSON only when you need full detail.
5. Review attacked videos visually if you need to judge realism vs effectiveness.
