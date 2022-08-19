import copy

import torch
import torch.nn as nn
import torch.utils.checkpoint
from cyy_naive_lib.log import get_logger
from cyy_torch_toolbox.model_util import ModelUtil
from cyy_torch_toolbox.reproducible_env import global_reproducible_env


class _CheckPointBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = nn.Sequential(*[m[1] for m in block])
        self.__block_names = [m[0] for m in block]
        get_logger().debug("use checkpoint_block %s", self.__block_names)

    def forward(self, x):
        return torch.utils.checkpoint.checkpoint(
            self.block, x, preserve_rng_state=global_reproducible_env.enabled
        )


def get_checkpointed_model(model) -> torch.nn.Module:
    checkpointed_blocks = ModelUtil(model).get_module_blocks(
        block_types={(nn.Conv2d, nn.BatchNorm2d)}
    )
    assert checkpointed_blocks
    checkpointed_model = copy.deepcopy(model)
    checkpointed_model.load_state_dict(model.state_dict())
    checkpointed_model_util = ModelUtil(checkpointed_model)
    for checkpointed_block in checkpointed_blocks:
        for idx, module in enumerate(checkpointed_block):
            module_name = module[0]
            if idx == 0:
                checkpointed_model_util.set_attr(
                    module_name,
                    _CheckPointBlock(checkpointed_block),
                    as_parameter=False,
                )
            else:
                checkpointed_model_util.set_attr(
                    module_name, lambda x: x, as_parameter=False
                )
    return checkpointed_model
