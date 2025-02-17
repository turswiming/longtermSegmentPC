import torch
from detectron2.checkpoint import DetectionCheckpointer
from typing import Dict, Any

class OurCheckpointer(DetectionCheckpointer):
    def _load_file(self, f: str) -> Dict[str, Any]:
        """
        Load a checkpoint file. Can be overwritten by subclasses to support
        different formats.

        Args:
            f (str): a locally mounted file path.
        Returns:
            dict: with keys "model" and optionally others that are saved by
                the checkpointer dict["model"] must be a dict which maps strings
                to torch.Tensor or numpy arrays.
        """
        return torch.load(f, weights_only=False,map_location=torch.device("cuda"))

