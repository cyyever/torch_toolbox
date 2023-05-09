#!/usr/bin/env python3

from cyy_torch_toolbox.data_structure.torch_task_queue import TorchTaskQueue
from cyy_torch_toolbox.device import get_cpu_device
from cyy_torch_toolbox.tensor import (assemble_tensors, disassemble_tensor,
                                      tensor_to)

from .torch_process_context import TorchProcessContext


class TorchProcessTaskQueue(TorchTaskQueue):
    ctx = None
    manager = None

    def __init__(
        self,
        send_tensor_in_cpu: bool = False,
        use_manager: bool = True,
        assemble_tensor: bool = False,
        **kwargs: dict
    ) -> None:
        self.__send_tensor_in_cpu = send_tensor_in_cpu
        if assemble_tensor:
            assert send_tensor_in_cpu
        self.__assemble_tensor = assemble_tensor
        super().__init__(mp_ctx=TorchProcessContext(use_manager=use_manager), **kwargs)

    # def __getstate__(self):
    #     state = super().__getstate__()
    #     state["_TorchProcessTaskQueue__manager"] = None
    #     return state

    def __process_tensor(self, data):
        if self.__send_tensor_in_cpu:
            data = tensor_to(data, device=get_cpu_device())
            if self.__assemble_tensor:
                data = assemble_tensors(data)
        return data

    def put_data(self, data, **kwargs):
        return super().put_data(data=self.__process_tensor(data), **kwargs)

    def get_data(self, **kwargs):
        data = super().get_data(**kwargs)
        if self.__assemble_tensor:
            data = disassemble_tensor(*data)
        return data
