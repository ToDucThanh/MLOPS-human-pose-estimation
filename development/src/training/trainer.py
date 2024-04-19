import time

import mlflow
import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DictDotNotation
from logs import log
from src.models.loss import OpenPoseLoss
from src.models.networks import OpenPoseNet
from src.utils.augment_data import DataTransform
from src.utils.dataset import COCOKeypointsDataset
from src.utils.manage_data_path import retrieve_data_path
from src.utils.mlflow_utils import log_model

from .base import TrainerBase


class Trainer(TrainerBase):
    def __init__(self, cfg: DictDotNotation):
        super().__init__(cfg)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = OpenPoseNet()
        self.hyperparamters = DictDotNotation(cfg.hyperparameters)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hyperparamters.lr,
            betas=self.hyperparamters.betas,
            weight_decay=self.hyperparamters.weight_decay,
        )
        self.criterion = OpenPoseLoss()
        self.model.to(self.device)
        self.training_iterations = 1
        self.validation_iterations = 1

    def prepare_data(self, train_ratio: float = 0.8) -> None:
        _, _, _, img_list, mask_list, meta_list = retrieve_data_path(
            self.cfg.data_root_path,
            self.cfg.train_mask_data_path,
            self.cfg.val_mask_data_path,
            self.cfg.label_subset_file,
        )
        train_num = int(train_ratio * len(img_list))
        self.train_img_list = img_list[:train_num]
        self.train_mask_list = mask_list[:train_num]
        self.train_meta_list = meta_list[:train_num]

        self.val_img_list = img_list[train_num:]
        self.val_mask_list = mask_list[train_num:]
        self.val_meta_list = meta_list[train_num:]

        self.train_dataset = COCOKeypointsDataset(
            self.train_img_list,
            self.train_mask_list,
            self.train_meta_list,
            phase="train",
            transform=DataTransform(),
        )

        self.val_dataset = COCOKeypointsDataset(
            self.val_img_list,
            self.val_mask_list,
            self.val_meta_list,
            phase="val",
            transform=DataTransform(),
        )

    @property
    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.hyperparamters.train_batch_size,
            shuffle=True,
            drop_last=True,
        )
        return dataloader

    @property
    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.hyperparamters.val_batch_size,
            shuffle=False,
            drop_last=True,
        )
        return dataloader

    def train(self, epoch: int) -> None:
        self.model.train()
        log.info(f"---------- Train epoch {epoch+1} ----------")
        train_batch_size = self.train_dataloader.batch_size
        train_loss_1_epoch = 0.0
        t_iter_start = time.perf_counter()

        for imges, heatmap_target, heat_mask, paf_target, paf_mask in self.train_dataloader:
            imges = imges.to(self.device)
            heatmap_target = heatmap_target.to(self.device)
            heat_mask = heat_mask.to(self.device)
            paf_target = paf_target.to(self.device)
            paf_mask = paf_mask.to(self.device)

            self.optimizer.zero_grad()
            _, saved_for_loss = self.model(imges)

            loss = self.criterion(saved_for_loss, heatmap_target, heat_mask, paf_target, paf_mask)
            del saved_for_loss
            loss.backward()
            self.optimizer.step()

            if self.training_iterations % 2 == 0:
                t_iter_finish = time.perf_counter()
                duration = t_iter_finish - t_iter_start
                log.info(
                    "[EPOCH] {}/{}, [ITERATION]: {} || TRAINING_Loss: {:.4f} || 2iter: {:.4f} sec.".format(
                        epoch + 1,
                        self.hyperparamters.epochs,
                        self.training_iterations,
                        loss.item() / train_batch_size,
                        duration,
                    )
                )
                if self.training_iterations % 4 == 0:
                    loss_item = loss.item() / train_batch_size
                    mlflow.log_metric(
                        "TRAINING_Loss", f"{loss_item:.4f}", step=self.training_iterations
                    )
                t_iter_start = time.perf_counter()

            train_loss_1_epoch += loss.item()
            self.training_iterations += 1
        return train_loss_1_epoch

    def valid(self, epoch) -> float:
        self.model.eval()
        log.info(f"---------- Validate epoch {epoch+1} ----------")
        val_batch_size = self.val_dataloader.batch_size
        val_loss_1_epoch = 0.0

        with torch.no_grad():
            t_iter_start = time.perf_counter()
            for imges, heatmap_target, heat_mask, paf_target, paf_mask in self.val_dataloader:
                imges = imges.to(self.device)
                heatmap_target = heatmap_target.to(self.device)
                heat_mask = heat_mask.to(self.device)
                paf_target = paf_target.to(self.device)
                paf_mask = paf_mask.to(self.device)

                _, saved_for_loss = self.model(imges)

                val_loss = self.criterion(
                    saved_for_loss, heatmap_target, heat_mask, paf_target, paf_mask
                )
                del saved_for_loss
                if self.validation_iterations % 2 == 0:
                    t_iter_finish = time.perf_counter()
                    duration = t_iter_finish - t_iter_start
                    log.info(
                        "[EPOCH] {}/{}, [ITERATION]: {} || VALIDATION_Loss: {:.4f} || 2iter: {:.4f} sec.".format(
                            epoch + 1,
                            self.hyperparamters.epochs,
                            self.validation_iterations,
                            val_loss.item() / val_batch_size,
                            duration,
                        )
                    )
                    if self.validation_iterations % 4 == 0:
                        loss_item = val_loss.item() / val_batch_size
                        mlflow.log_metric(
                            "VALIDATION_Loss", f"{loss_item:.4f}", step=self.validation_iterations
                        )
                    t_iter_start = time.perf_counter()

                val_loss_1_epoch += val_loss.item()
                self.validation_iterations += 1
        return val_loss_1_epoch

    def fit(self, train_ratio: float = 0.8) -> None:
        self.prepare_data(train_ratio)

        num_train_imgs = len(self.train_dataloader.dataset)
        num_valid_imgs = len(self.val_dataloader.dataset)

        log.info(f"The number of training images: {num_train_imgs}")
        log.info(f"The number of validation images: {num_valid_imgs}")
        log.info(f"The number of training epochs: {self.hyperparamters.epochs}")

        global_val_loss = np.inf

        for epoch in tqdm(range(self.hyperparamters.epochs), desc="Training Loop"):
            epoch_train_loss = 0.0
            epoch_valid_loss = 0.0

            t_epoch_start = time.perf_counter()

            train_loss_1_epoch = self.train(epoch)
            epoch_train_loss += train_loss_1_epoch
            epoch_train_loss /= num_train_imgs

            val_loss_1_epoch = self.valid(epoch)
            epoch_valid_loss += val_loss_1_epoch
            epoch_valid_loss /= num_valid_imgs

            t_epoch_finish = time.perf_counter()
            log.info(
                "[EPOCH] {}/{} || TRAINING_loss_per_epoch: {:.4f} || VALIDATION_loss_per_epoch: {:.4f}".format(
                    epoch + 1, self.hyperparamters.epochs, epoch_train_loss, epoch_valid_loss
                )
            )
            log.info(
                "[TIME TO FINSIH] Epoch {}: {:.4f} sec.".format(
                    epoch + 1, t_epoch_finish - t_epoch_start
                )
            )
            mlflow.log_metric("TRAINING_loss_per_epoch", f"{epoch_train_loss:.4f}", step=epoch)
            mlflow.log_metric("VALIDATION_loss_per_epoch", f"{epoch_valid_loss:.4f}", step=epoch)

            tags = {
                "val_loss": round(epoch_valid_loss, 4),
                "model_name": "OpenPose",
                "version": "1.0.0",
            }

            if epoch_valid_loss < global_val_loss:
                global_val_loss = epoch_valid_loss
                log_model(
                    self.model.state_dict(), artifact_path="model_state_dict_best", tags=tags
                )

            if (epoch + 1) % 3 == 0:
                log_model(
                    self.model.state_dict(),
                    artifact_path=f"model_state_dict_epochs/{epoch+1}",
                )

            if epoch == self.hyperparamters.epochs - 1:
                log_model(self.model.state_dict(), artifact_path="model_state_dict_last")

            t_epoch_start = time.perf_counter()

    @property
    def get_hyperparamters(self):
        return self.hyperparamters
