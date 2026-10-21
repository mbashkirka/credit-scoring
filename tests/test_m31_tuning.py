import importlib.util
import sys
from pathlib import Path


def _load_script_module():
    path = Path(__file__).parents[1] / "scripts" / "train_m31_tuning.py"
    spec = importlib.util.spec_from_file_location("train_m31_tuning", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_m31_tuning_grid_is_small_and_contains_exact_m3_baseline():
    module = _load_script_module()
    configs = module.tuning_configs()
    assert 8 <= len(configs) <= 12
    assert configs[0].name == "baseline_d6_lr005_l2_5"
    assert configs[0].model_kwargs() == {
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
    }
    assert len({c.name for c in configs}) == len(configs)
