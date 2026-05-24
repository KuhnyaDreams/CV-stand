import json
from pathlib import Path


DEFAULT_PRESETS_PATH = Path(__file__).with_name("bb_attack_presets.json")


def load_attack_presets(presets_path: str | Path | None = None) -> dict:
    """
    Load attack presets from JSON.

    The expected format is:
    {
      "attack_name": {
        "preset_name": {
          "... attack params ..."
        }
      }
    }
    """
    path = Path(presets_path) if presets_path is not None else DEFAULT_PRESETS_PATH
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Attack presets file must contain a JSON object: {path}")

    return data


def get_attack_preset_params(
    attack_name: str,
    preset_name: str | None,
    presets: dict,
) -> dict:
    """
    Resolve one preset for a given attack. Returns an empty dict if no preset is requested.
    """
    if not preset_name:
        return {}

    attack_presets = presets.get(attack_name)
    if not isinstance(attack_presets, dict):
        raise ValueError(f"No presets defined for attack '{attack_name}'")

    preset_params = attack_presets.get(preset_name)
    if not isinstance(preset_params, dict):
        raise ValueError(f"Preset '{preset_name}' not found for attack '{attack_name}'")

    return dict(preset_params)


def merge_attack_params(
    preset_params: dict | None = None,
    override_params: dict | None = None,
) -> dict:
    """
    Merge preset params with explicit overrides.

    Explicit overrides always win over preset values.
    """
    merged = {}
    if preset_params:
        merged.update(preset_params)
    if override_params:
        merged.update(override_params)
    return merged

