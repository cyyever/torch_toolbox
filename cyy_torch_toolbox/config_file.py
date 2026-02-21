import sys
from pathlib import Path
from typing import Any

import hydra
from cyy_naive_lib.log import log_debug
from omegaconf import OmegaConf


def load_combined_config_from_files(
    config_path: str | Path, other_config_files: list[str | Path] | None = None
) -> Any:
    # disable hydra output dir
    for option in [
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]:
        sys.argv.append(option)
    if other_config_files is None:
        other_config_files = []
    if len(other_config_files) > 2:
        log_debug(
            "load config from auxiliary files: %s, you should make sure that they are from global to local",
            other_config_files,
        )
    other_confs = [OmegaConf.load(file) for file in other_config_files]
    conf_obj: Any = None
    resolved_path = Path(config_path).resolve()
    config_name_str: str | None = None
    config_path_str: str
    if resolved_path.suffix == ".yaml":
        config_name_str = resolved_path.stem
        config_path_str = str(resolved_path.parent)  # hydra.main requires str
    else:
        config_path_str = str(resolved_path)  # hydra.main requires str

    @hydra.main(
        config_path=config_path_str,
        config_name=config_name_str,
        version_base=None,
    )
    def load_config_hydra(conf: Any) -> None:
        nonlocal conf_obj
        conf_obj = conf

    load_config_hydra()

    other_confs.append(conf_obj)
    for idx, conf_obj in enumerate(other_confs):
        while "dataset_name" not in conf_obj and len(conf_obj) == 1:
            conf_obj = next(iter(conf_obj.values()))
        other_confs[idx] = conf_obj

    result_conf = other_confs[0]
    for o in other_confs[1:]:
        result_conf.merge_with(o)
    return result_conf
