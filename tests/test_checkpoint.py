import pytest
import torch
from safetensors.torch import save_file, save_model

from focus_vlwa.model.checkpoint import load_checkpoint_weights
from focus_vlwa.model.world_model import WorldModelExpert
from focus_vlwa.scripts import check_checkpoint


class Model(WorldModelExpert, torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_world_model = True
        self.world_model_total_horizon = 5
        self.world_model_horizon = 4
        self.world_model_dim = 2
        self.joint_experts = torch.nn.Module()
        self.joint_experts.action_expert = torch.nn.Linear(2, 2)
        self.joint_experts.world_model_expert = torch.nn.Linear(2, 2)
        for name in (
            "action_input_projection",
            "action_output_projection",
            "world_model_input_projection",
            "world_model_output_projection",
        ):
            setattr(self, name, torch.nn.Linear(2, 2))


def test_loaded_world_weights_are_preserved_until_explicitly_reinitialized(tmp_path):
    original = Model()
    with torch.no_grad():
        for name, value in original.named_parameters():
            value.fill_(7 if "world_model" in name else 1)
    save_model(original, tmp_path / "model.safetensors")
    loaded = Model()
    load_checkpoint_weights(loaded, tmp_path)
    assert (loaded.joint_experts.world_model_expert.weight == 7).all()
    load_checkpoint_weights(loaded, tmp_path, init_world_model_from_action=True)
    for parameter in loaded.parameters():
        assert (parameter == 1).all()


@pytest.mark.parametrize("prefix", ["backbone.action_expert.", "paligemma_with_expert.gemma_expert."])
def test_loader_and_checker_reject_old_names(tmp_path, monkeypatch, prefix):
    model = Model()
    state = model.state_dict()
    for key in list(state):
        if key.startswith("joint_experts.action_expert."):
            state[prefix + key.removeprefix("joint_experts.action_expert.")] = state.pop(key)
    save_file(state, tmp_path / "model.safetensors")
    (tmp_path / "model_config.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Checkpoint mismatch"):
        load_checkpoint_weights(Model(), tmp_path)
    monkeypatch.setattr(check_checkpoint, "FocusVLWA", lambda config: Model())
    missing, unexpected, shapes = check_checkpoint.validate_checkpoint(tmp_path)
    assert "joint_experts.action_expert.weight" in missing
    assert prefix + "weight" in unexpected
    assert shapes == []


def test_current_checkpoint_passes_checker(tmp_path, monkeypatch):
    save_model(Model(), tmp_path / "model.safetensors")
    (tmp_path / "model_config.json").write_text("{}")
    monkeypatch.setattr(check_checkpoint, "FocusVLWA", lambda config: Model())
    assert check_checkpoint.validate_checkpoint(tmp_path) == ([], [], [])


def test_missing_world_weights_require_explicit_initialization(tmp_path):
    state = {name: value for name, value in Model().state_dict().items() if "world_model" not in name}
    save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Checkpoint mismatch"):
        load_checkpoint_weights(Model(), tmp_path)
    initialized = Model()
    load_checkpoint_weights(initialized, tmp_path, init_world_model_from_action=True)
    torch.testing.assert_close(
        initialized.joint_experts.world_model_expert.weight, state["joint_experts.action_expert.weight"]
    )
