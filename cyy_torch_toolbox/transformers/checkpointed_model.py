import copy

import torch
import torch.nn as nn
import torch.utils.checkpoint
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.reproducible_env import global_reproducible_env

from cyy_torch_toolbox.model_util import ModelUtil


class _CheckPointBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = nn.Sequential(*[m[1] for m in block])
        self.__block_names = [m[0] for m in block]
        get_logger().debug("use checkpoint_block %s", self.__block_names)

    def forward(self, x):
        if not global_reproducible_env.enabled:
            return torch.utils.checkpoint.checkpoint(
                self.block, x, preserve_rng_state=False
            )
        return torch.utils.checkpoint.checkpoint(self.block, x)


def get_checkpointed_model(model) -> torch.nn.Module:
    checkpointed_blocks = ModelUtil(model).get_sub_module_blocks(
        block_types={(nn.Conv2d, nn.BatchNorm2d)},
        only_block_name=False,
    )
    assert checkpointed_blocks
    checkpointed_model = copy.deepcopy(model)
    checkpointed_model.load_state_dict(model.state_dict())
    checkpointed_model_util = ModelUtil(checkpointed_model)
    for checkpointed_block in checkpointed_blocks:
        for idx, submodule in enumerate(checkpointed_block):
            submodule_name = submodule[0]
            if idx == 0:
                checkpointed_model_util.set_attr(
                    submodule_name,
                    _CheckPointBlock(checkpointed_block),
                    as_parameter=False,
                )
            else:
                checkpointed_model_util.set_attr(
                    submodule_name, lambda x: x, as_parameter=False
                )
    return checkpointed_model
