from typing import (
    Optional,
    Tuple,
    Union,
)

import cv2
import numpy as np
import torch

from loguru import logger

from src.models.networks import OpenPoseNet

from .base import PoseInferenceBase
from .decode import decode_pose


class PoseInference(PoseInferenceBase):
    def __init__(self, model_weight_path: str, device: Optional[str] = None):
        super().__init__()
        self.net = OpenPoseNet()
        if not device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        net_weights = torch.load(model_weight_path, map_location=self.device)
        keys = list(net_weights.keys())

        weights_load = {}
        for i in range(len(keys)):
            weights_load[list(self.net.state_dict().keys())[i]] = net_weights[list(keys)[i]]

        state = self.net.state_dict()
        state.update(weights_load)
        self.net.load_state_dict(state)
        self.net.eval()

        logger.info(f"Load model successfully to device '{self.device}' for inference")

    def preprocess(
        self,
        img: Union[np.ndarray, str],
        size: Tuple[int] = (368, 368),
        color_mean: Tuple[float] = (0.485, 0.456, 0.406),
        color_std: Tuple[float] = (0.229, 0.224, 0.225),
    ) -> torch.Tensor:
        if isinstance(img, str):
            original_img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            original_img = img
        else:
            raise ValueError("'img' parameter must be of type string or numpy array")

        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(original_img, size, interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0

        preprocessed_img = img.copy()

        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - color_mean[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / color_std[i]

        img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)

        return img, original_img

    def process(
        self,
        img: Union[np.ndarray, str],
        size: Tuple[int] = (368, 368),
        color_mean: Tuple[float] = (0.485, 0.456, 0.406),
        color_std: Tuple[float] = (0.229, 0.224, 0.225),
    ):
        preprocessed_img, original_img = self.preprocess(
            img=img, size=size, color_mean=color_mean, color_std=color_std
        )
        # Run model
        predicted_outputs, _ = self.net(preprocessed_img)

        shape = original_img.shape
        heatmaps = PoseInference._generate_heatmap(predicted_outputs, size, shape)
        pafs = PoseInference._generate_part_affinity_fields(predicted_outputs, size, shape)

        result_img = self.postprocess(original_img, heatmaps, pafs)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        return result_img

    def postprocess(
        self,
        oriImg: np.ndarray,
        heatmaps: np.ndarray,
        pafs: np.ndarray,
    ) -> np.ndarray:
        _, result_img, _, _ = decode_pose(oriImg, heatmaps, pafs)
        return result_img

    @staticmethod
    def _generate_heatmap(
        predicted_outputs: torch.Tensor,
        size: Tuple[int],
        oriImg_shape: Tuple[int],
    ) -> np.ndarray:
        _heatmaps = predicted_outputs[1][0].detach().numpy().transpose(1, 2, 0)
        _heatmaps = cv2.resize(_heatmaps, size, interpolation=cv2.INTER_CUBIC)
        _heatmaps = cv2.resize(
            _heatmaps,
            (oriImg_shape[1], oriImg_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        logger.info("Generate heatmap ...")
        return _heatmaps

    @staticmethod
    def _generate_part_affinity_fields(
        predicted_outputs: torch.Tensor,
        size: Tuple[int],
        oriImg_shape: Tuple[int],
    ) -> np.ndarray:
        _pafs = predicted_outputs[0][0].detach().numpy().transpose(1, 2, 0)
        _pafs = cv2.resize(_pafs, size, interpolation=cv2.INTER_CUBIC)
        _pafs = cv2.resize(
            _pafs,
            (oriImg_shape[1], oriImg_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        logger.info("Generate part affinity fields ...")
        return _pafs
