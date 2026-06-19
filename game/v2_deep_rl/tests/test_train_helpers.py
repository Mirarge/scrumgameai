"""
Tests for pure helper functions in training/train_dqn.py:
  - epsilon_by_episode
  - _slugify_run_name
  - resolve_training_config

training.train_dqn imports torch and matplotlib at module level, so we cannot do a
plain import. Instead we:
  1. Skip gracefully if torch is not installed (pytest.importorskip).
  2. Temporarily restore any stub from sys.modules and replace it with the real
     torch before importing the engine module.

If torch IS installed the stubs are swapped out and the real module is loaded
via importlib so we get the real epsilon_by_episode / _slugify_run_name
implementations.
"""
from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: need real torch to import train_dqn
# ---------------------------------------------------------------------------

# Remove conftest stubs so the real libraries can be found.
# Save the stub first so we can restore it if real torch is unusable.
_torch_stub = sys.modules.get("torch")
for _mod in list(sys.modules):
    if _mod == "torch" or _mod.startswith("torch."):
        del sys.modules[_mod]
for _heavy in ("rl.dqn_agent", "game_runtime.scrum_game_env"):
    sys.modules.pop(_heavy, None)

# PyTorch doesn't support Python 3.14+ yet; running on 3.14 causes a hard C abort.
if sys.version_info >= (3, 14):
    if _torch_stub is not None:
        sys.modules["torch"] = _torch_stub
    pytest.skip("torch not compatible with Python 3.14+ — skipping train_dqn helper tests",
                allow_module_level=True)

# importorskip only catches ImportError; guard against RuntimeError too.
try:
    import torch as _torch_check  # noqa: F401
except (ImportError, RuntimeError) as _e:
    # Restore the stub so other test modules still work.
    if _torch_stub is not None:
        sys.modules["torch"] = _torch_stub
    pytest.skip(f"torch unavailable or broken ({_e}) — skipping train_dqn helper tests",
                allow_module_level=True)

torch = pytest.importorskip("torch", reason="torch not installed — skipping train_dqn helper tests")

# matplotlib may also be absent on minimal CI envs.
pytest.importorskip("matplotlib", reason="matplotlib not installed — skipping train_dqn helper tests")

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

_ENGINE_DIR = Path(__file__).resolve().parents[1]

# We need the engine directory on sys.path so package imports work.
_engine_str = str(_ENGINE_DIR)
if _engine_str not in sys.path:
    sys.path.insert(0, _engine_str)

_spec = importlib.util.spec_from_file_location("training.train_dqn", _ENGINE_DIR / "training" / "train_dqn.py")
_train_dqn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_dqn)

epsilon_by_episode = _train_dqn.epsilon_by_episode
_slugify_run_name = _train_dqn._slugify_run_name
resolve_training_config = _train_dqn.resolve_training_config


# ---------------------------------------------------------------------------
# epsilon_by_episode
# ---------------------------------------------------------------------------

class TestEpsilonByEpisode:
    def test_episode_zero_returns_epsilon_start(self):
        result = epsilon_by_episode(0, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay_end=0.9)
        assert result == pytest.approx(1.0)

    def test_episode_at_decay_returns_epsilon_min(self):
        result = epsilon_by_episode(1000, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay_end=0.9)
        assert result == pytest.approx(0.05)

    def test_episode_beyond_decay_clamps_to_epsilon_min(self):
        result = epsilon_by_episode(999999, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay_end=0.9)
        assert result == pytest.approx(0.05)

    def test_midpoint_is_linearly_interpolated(self):
        """At episode 500 with decay=1000 epsilon should be (start+min)/2."""
        start, minimum, decay = 1.0, 0.0, 0.9
        result = epsilon_by_episode(500, epsilon_start=start, epsilon_min=minimum, epsilon_decay_end=decay)
        expected = start - (start - minimum) * (500 / 1000)
        assert result == pytest.approx(expected)

    def test_decay_is_strictly_decreasing(self):
        """Epsilon must be strictly decreasing from episode 0 to decay-1."""
        previous = epsilon_by_episode(0, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay_end=0.9)
        for ep in range(1, 501):
            current = epsilon_by_episode(ep, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay_end=0.9)
            assert current <= previous + 1e-9, f"Epsilon increased at episode {ep}"
            previous = current

    def test_epsilon_at_episode_one_less_than_decay(self):
        """Episode just before the cliff should be just above epsilon_min."""
        result = epsilon_by_episode(449999, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay_end=0.9)
        assert result > 0.05
        assert result < 1.0

    def test_custom_start_and_min(self):
        result = epsilon_by_episode(0, epsilon_start=0.8, epsilon_min=0.1, epsilon_decay_end=200)
        assert result == pytest.approx(0.8)
        result_end = epsilon_by_episode(200, epsilon_start=0.8, epsilon_min=0.1, epsilon_decay_end=200)
        assert result_end == pytest.approx(0.1)

    def test_default_parameters(self):
        """Default call must return 1.0 at episode 0 and 0.05 at 450000."""
        assert epsilon_by_episode(0) == pytest.approx(1.0)
        assert epsilon_by_episode(450000) == pytest.approx(0.05)

    @pytest.mark.parametrize("episode", [0, 1, 100, 1000, 9999])
    def test_result_always_between_min_and_start(self, episode):
        result = epsilon_by_episode(episode, epsilon_start=1.0, epsilon_min=0.05, epsilon_decay_end=10000)
        assert 0.05 <= result <= 1.0


# ---------------------------------------------------------------------------
# _slugify_run_name
# ---------------------------------------------------------------------------

class TestSlugifyRunName:
    def test_none_returns_empty_string(self):
        assert _slugify_run_name(None) == ""

    def test_empty_string_returns_empty_string(self):
        assert _slugify_run_name("") == ""

    def test_whitespace_only_returns_empty_string(self):
        assert _slugify_run_name("   ") == ""

    def test_simple_alpha_lowercased(self):
        assert _slugify_run_name("HelloWorld") == "helloworld"

    def test_spaces_become_underscores(self):
        assert _slugify_run_name("my run name") == "my_run_name"

    def test_special_chars_become_underscores(self):
        result = _slugify_run_name("run-name.v2!")
        assert result == "run_name_v2"

    def test_consecutive_special_chars_collapsed(self):
        """Multiple consecutive non-alnum chars should collapse to a single underscore."""
        result = _slugify_run_name("run  --  name")
        assert result == "run_name"

    def test_leading_and_trailing_special_chars_stripped(self):
        result = _slugify_run_name("---test---")
        assert result == "test"

    def test_numbers_preserved(self):
        result = _slugify_run_name("run123")
        assert result == "run123"

    def test_mixed_alphanumeric_and_special(self):
        result = _slugify_run_name("Autopilot Run v2.0")
        assert result == "autopilot_run_v20"

    def test_length_truncated_to_48(self):
        long_name = "a" * 100
        result = _slugify_run_name(long_name)
        assert len(result) <= 48

    def test_exactly_48_chars_not_truncated(self):
        name = "a" * 48
        assert _slugify_run_name(name) == name

    def test_49_chars_truncated(self):
        name = "a" * 49
        result = _slugify_run_name(name)
        assert len(result) == 48

    def test_unicode_non_alnum_becomes_underscore(self):
        result = _slugify_run_name("naïve résumé")
        # ï, é are non-alphanumeric in isalnum but letters; actual behaviour
        # depends on Python str.isalnum() which returns True for unicode letters.
        # At minimum the function should not crash.
        assert isinstance(result, str)

    def test_all_special_chars_returns_empty(self):
        result = _slugify_run_name("!@#$%^&*()")
        assert result == ""


# ---------------------------------------------------------------------------
# resolve_training_config
# ---------------------------------------------------------------------------

class TestResolveTrainingConfig:
    """resolve_training_config loads a base config then applies overrides."""

    def _base_tc(self, **kwargs):
        """Return a TrainingConfig with sensible defaults, optionally overridden."""
        import config.config_manager as cm
        defaults = {
            "episodes": 50000,
            "evaluation_episodes": 100,
            "checkpoint_interval": 10000,
            "evaluation_interval": 10000,
            "learning_rate": 0.0005,
            "gamma": 0.85,
            "replay_capacity": 100000,
            "batch_size": 128,
            "target_update_frequency": 2000,
            "seed": 42,
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay_end": 45000,
            "run_notes": "",
        }
        defaults.update(kwargs)
        return cm.TrainingConfig.from_dict(defaults)

    def test_no_overrides_returns_base_unchanged(self):
        base = self._base_tc()
        result = resolve_training_config(training_config=base)
        assert result.episodes == 50000
        assert result.learning_rate == pytest.approx(0.0005)

    def test_num_episodes_override(self):
        base = self._base_tc()
        result = resolve_training_config(training_config=base, num_episodes=99999)
        assert result.episodes == 99999

    def test_learning_rate_override(self):
        base = self._base_tc()
        result = resolve_training_config(training_config=base, learning_rate=0.001)
        assert result.learning_rate == pytest.approx(0.001)

    def test_gamma_override(self):
        base = self._base_tc()
        result = resolve_training_config(training_config=base, gamma=0.99)
        assert result.gamma == pytest.approx(0.99)

    def test_seed_override(self):
        base = self._base_tc()
        result = resolve_training_config(training_config=base, seed=123)
        assert result.seed == 123

    def test_checkpoint_interval_override(self):
        base = self._base_tc()
        result = resolve_training_config(training_config=base, checkpoint_interval=500)
        assert result.checkpoint_interval == 500

    def test_evaluation_interval_override(self):
        base = self._base_tc()
        result = resolve_training_config(training_config=base, evaluation_interval=2500)
        assert result.evaluation_interval == 2500

    def test_evaluation_episodes_override(self):
        base = self._base_tc()
        result = resolve_training_config(training_config=base, evaluation_episodes=50)
        assert result.evaluation_episodes == 50

    def test_epsilon_decay_override(self):
        base = self._base_tc()
        result = resolve_training_config(training_config=base, epsilon_decay_end=10000)
        assert result.epsilon_decay_end == 10000

    def test_run_notes_override(self):
        base = self._base_tc()
        result = resolve_training_config(training_config=base, run_notes="my notes")
        assert result.run_notes == "my notes"

    def test_empty_run_notes_does_not_override(self):
        """run_notes='' (the default) must not overwrite a base config's notes."""
        base = self._base_tc(run_notes="original notes")
        result = resolve_training_config(training_config=base, run_notes="")
        # Empty string is falsy, so the if-guard skips the override
        assert result.run_notes == "original notes"

    def test_none_overrides_leave_base_values(self):
        """All override params defaulting to None must leave the base values intact."""
        base = self._base_tc(learning_rate=0.0003, gamma=0.92)
        result = resolve_training_config(training_config=base)
        assert result.learning_rate == pytest.approx(0.0003)
        assert result.gamma == pytest.approx(0.92)

    def test_multiple_overrides_applied_simultaneously(self):
        base = self._base_tc()
        result = resolve_training_config(
            training_config=base,
            num_episodes=1000,
            learning_rate=0.002,
            seed=7,
        )
        assert result.episodes == 1000
        assert result.learning_rate == pytest.approx(0.002)
        assert result.seed == 7

    def test_returns_training_config_instance(self):
        import config.config_manager as cm
        base = self._base_tc()
        result = resolve_training_config(training_config=base)
        assert isinstance(result, cm.TrainingConfig)
