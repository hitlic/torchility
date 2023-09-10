def load_state_dict(model, ckpt_path):
    """
    load state_dict for pytorch model from pytorch_lightning checkpoint.
    Args:
        model: pytorch model
        ckpt_path: path of pytorch_lightning checkpoint
    """
    from lightning_fabric.utilities.cloud_io import _load as pl_load
    from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE
    from pytorch_lightning.utilities.migration import pl_legacy_patch
    from typing import cast

    map_location = cast(_MAP_LOCATION_TYPE, lambda storage, loc: storage)
    with pl_legacy_patch():
        checkpoint = pl_load(ckpt_path, map_location=map_location)
    state_dict = {k.split('.', 1)[1] :v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    return model
