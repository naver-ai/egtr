# Reference: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb

import argparse
import os
from glob import glob
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader

from data.open_image import OIDetection
from data.visual_genome import VGDetection
from lib.evaluation.coco_eval import CocoEvaluator
from lib.evaluation.oi_eval import OICocoEvaluator
from model.deformable_detr import (
    DeformableDetrConfig,
    DeformableDetrFeatureExtractor,
    DeformableDetrFeatureExtractorWithAugmentor,
    DeformableDetrForObjectDetection,
)
from util.misc import use_deterministic_algorithms

seed_everything(42, workers=True)


def collate_fn(batch, feature_extractor):
    pixel_values = [item[0] for item in batch]
    encoding = feature_extractor.pad_and_create_pixel_mask(
        pixel_values, return_tensors="pt"
    )
    labels = [item[1] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


class Detr(pl.LightningModule):
    def __init__(
        self,
        backbone_dirpath,
        auxiliary_loss,
        lr,
        lr_backbone,
        weight_decay,
        main_trained,
        id2label,
        num_queries,
        architecture,
        ce_loss_coefficient,
        coco_evaluator,
        oi_coco_evaluator,
        feature_extractor,
    ):
        super().__init__()
        # replace COCO classification head with custom head
        config = DeformableDetrConfig.from_pretrained(architecture)
        config.architecture = architecture
        config.auxiliary_loss = auxiliary_loss
        config.num_labels = max(id2label.keys()) + 1
        config.num_queries = num_queries
        config.ce_loss_coefficient = ce_loss_coefficient
        config.output_attention_states = False
        self.model = DeformableDetrForObjectDetection(config=config)
        self.model.model.backbone.load_state_dict(
            torch.load(f"{backbone_dirpath}/{config.backbone}.pt")
        )

        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.coco_evaluator = coco_evaluator
        self.oi_coco_evaluator = oi_coco_evaluator
        self.feature_extractor = feature_extractor
        if main_trained:
            state_dict = torch.load(main_trained, map_location="cpu")["state_dict"]
            for k in list(state_dict.keys()):
                state_dict[k[6:]] = state_dict.pop(k)  # "model."
            self.model.load_state_dict(state_dict, strict=False)

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = batch["labels"]

        outputs = self.model(
            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
        )
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        del outputs
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        log_dict = {
            "step": torch.tensor(self.global_step, dtype=torch.float32),
            "training_loss": loss.item(),
        }
        log_dict.update({f"training_{k}": v.item() for k, v in loss_dict.items()})
        self.log_dict(log_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        loss_dict["loss"] = loss
        del loss
        return loss_dict

    def validation_epoch_end(self, outputs):
        log_dict = {
            "step": torch.tensor(self.global_step, dtype=torch.float32),
            "epoch": torch.tensor(self.current_epoch, dtype=torch.float32),
        }
        for k in outputs[0].keys():
            log_dict[f"validation_" + k] = (
                torch.stack([x[k] for x in outputs]).mean().item()
            )
        self.log_dict(log_dict, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # get the inputs
        pixel_values = batch["pixel_values"].to(self.device)
        pixel_mask = batch["pixel_mask"].to(self.device)
        labels = [
            {k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]
        ]  # these are in DETR format, resized + normalized

        # forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        orig_target_sizes = torch.stack(
            [target["orig_size"] for target in labels], dim=0
        )
        results = self.feature_extractor.post_process(
            outputs, orig_target_sizes
        )  # convert outputs of model to COCO api
        res = {
            target["image_id"].item(): output for target, output in zip(labels, results)
        }
        if self.coco_evaluator is not None:
            self.coco_evaluator.update(res)
        if self.oi_coco_evaluator is not None:
            self.oi_coco_evaluator(labels, res)

    def test_epoch_end(self, outputs):
        # log OD
        if self.coco_evaluator is not None:
            self.coco_evaluator.synchronize_between_processes()
            self.coco_evaluator.accumulate()
            self.coco_evaluator.summarize()
            self.log("AP50", self.coco_evaluator.coco_eval["bbox"].stats[1])
        if self.oi_coco_evaluator is not None:
            self.log_dict(self.oi_coco_evaluator.aggregate_metrics())

    def configure_optimizers(self):
        diff_lr_params = ["backbone", "reference_points", "sampling_offsets"]
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if (not any(nd in n for nd in diff_lr_params)) and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in diff_lr_params) and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader


if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser()
    # Path
    parser.add_argument("--data_path", type=str, default="dataset/visual_genome")
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
    )
    parser.add_argument("--backbone_dirpath", type=str, required=True)

    # Architecture
    parser.add_argument("--architecture", type=str, default="SenseTime/deformable-detr")
    parser.add_argument("--auxiliary_loss", type=str2bool, default=True)

    # Hyperparameters
    parser.add_argument("--num_queries", type=int, default=200)
    parser.add_argument("--ce_loss_coefficient", type=float, default=2.0)

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--max_epochs_finetune", type=int, default=50)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_val", type=float, default=0.1)

    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument("--memo", type=str, default="")
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--finetune", type=str2bool, default=True)

    # Evaluation
    parser.add_argument("--skip_train", type=str2bool, default=False)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_when_train_end", type=str2bool, default=True)

    # Speed up
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])

    args = parser.parse_args()

    # Feature extractor
    feature_extractor = DeformableDetrFeatureExtractor.from_pretrained(
        args.architecture, size=800, max_size=1333
    )
    feature_extractor_train = (
        DeformableDetrFeatureExtractorWithAugmentor.from_pretrained(
            args.architecture, size=800, max_size=1333
        )
    )

    # Dataset
    if "visual_genome" in args.data_path:
        train_dataset = VGDetection(
            data_folder=args.data_path,
            feature_extractor=feature_extractor_train,
            split="train",
            debug=args.debug,
        )
        val_dataset = VGDetection(
            data_folder=args.data_path, feature_extractor=feature_extractor, split="val"
        )
        cats = train_dataset.coco.cats
        id2label = {k - 1: v["name"] for k, v in cats.items()}  # 0 ~ 149
    else:
        train_dataset = OIDetection(
            data_folder=args.data_path,
            feature_extractor=feature_extractor_train,
            split="train",
            debug=args.debug,
        )
        val_dataset = OIDetection(
            data_folder=args.data_path, feature_extractor=feature_extractor, split="val"
        )
        id2label = train_dataset.classes_to_ind  # 0 ~ 600
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    # Dataloader
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=lambda x: collate_fn(x, feature_extractor),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=lambda x: collate_fn(x, feature_extractor),
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    # Evaluator
    if args.eval_when_train_end:
        if "visual_genome" in args.data_path:
            coco_evaluator = CocoEvaluator(
                val_dataset.coco, ["bbox"]
            )  # initialize evaluator with ground truths
            oi_coco_evaluator = None
        elif "open-image" in args.data_path:
            oi_coco_evaluator = OICocoEvaluator(
                train_dataset.rel_categories, train_dataset.ind_to_classes
            )
            coco_evaluator = None
    else:
        coco_evaluator = None
        oi_coco_evaluator = None

    # Logger setting
    save_dir = (
        f"{args.output_path}/pretrained_detr__{args.architecture.replace('/', '__')}"
    )
    name = f"batch__{args.batch_size * args.gpus * args.accumulate}__epochs__{args.max_epochs}_{args.max_epochs_finetune}__lr__{args.lr_backbone}_{args.lr}"
    if args.memo:
        name += f"__{args.memo}"
    if args.debug:
        name += "__debug"
    if args.resume:
        version = args.version  # for resuming
    else:
        version = None  #  If version is not specified the logger inspects the save directory for existing versions, then automatically assigns the next available version.

    # Trainer setting
    logger = TensorBoardLogger(save_dir, name=name, version=version)
    if os.path.exists(f"{logger.log_dir}/checkpoints"):
        if os.path.exists(f"{logger.log_dir}/checkpoints/last.ckpt"):
            ckpt_path = f"{logger.log_dir}/checkpoints/last.ckpt"
        else:
            ckpt_path = sorted(
                glob(f"{logger.log_dir}/checkpoints/epoch=*.ckpt"),
                key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
            )[-1]
    else:
        ckpt_path = None

    # Module
    module = Detr(
        backbone_dirpath=args.backbone_dirpath,
        auxiliary_loss=args.auxiliary_loss,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
        main_trained="",
        id2label=id2label,
        num_queries=args.num_queries,
        architecture=args.architecture,
        ce_loss_coefficient=args.ce_loss_coefficient,
        coco_evaluator=coco_evaluator,
        oi_coco_evaluator=oi_coco_evaluator,
        feature_extractor=feature_extractor,
    )

    # Callback
    checkpoint_callback = ModelCheckpoint(
        monitor="validation_loss",
        filename="{epoch:02d}-{validation_loss:.2f}",
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="validation_loss", patience=args.patience, verbose=True, mode="min"
    )

    # Train
    trainer = None
    if not args.skip_train:
        # Main training
        if not Path(
            TensorBoardLogger(
                save_dir, name=f"{name}__finetune", version=version
            ).log_dir
        ).exists():
            # Training
            trainer = Trainer(
                precision=args.precision,
                logger=logger,
                gpus=args.gpus,
                max_epochs=args.max_epochs,
                gradient_clip_val=args.gradient_clip_val,
                strategy=DDPStrategy(find_unused_parameters=False),
                callbacks=[checkpoint_callback, early_stop_callback],
                accumulate_grad_batches=args.accumulate,
            )
            use_deterministic_algorithms()
            if trainer.is_global_zero:
                print("### Main training")
            trainer.fit(module, ckpt_path=ckpt_path)

            try:
                os.chmod(logger.log_dir, 0o0777)
            except PermissionError as e:
                print(e)

        # Finetuning
        if args.finetune:
            ckpt_path = sorted(
                glob(f"{logger.log_dir}/checkpoints/epoch=*.ckpt"),
                key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
            )[-1]

            # Finetune trainer setting
            logger = TensorBoardLogger(
                save_dir, name=f"{name}__finetune", version=version
            )
            if os.path.exists(f"{logger.log_dir}/checkpoints"):
                finetune_ckpt_path = f"{logger.log_dir}/checkpoints/last.ckpt"
            else:
                finetune_ckpt_path = None

            # Finetune module
            module = Detr(
                backbone_dirpath=args.backbone_dirpath,
                auxiliary_loss=args.auxiliary_loss,
                lr=args.lr * 0.1,
                lr_backbone=args.lr_backbone * 0.1,
                weight_decay=args.weight_decay,
                main_trained=ckpt_path,
                id2label=id2label,
                num_queries=args.num_queries,
                architecture=args.architecture,
                ce_loss_coefficient=args.ce_loss_coefficient,
                coco_evaluator=coco_evaluator,
                oi_coco_evaluator=oi_coco_evaluator,
                feature_extractor=feature_extractor,
            )

            # Finetune callback
            checkpoint_callback = ModelCheckpoint(
                monitor="validation_loss",
                filename="{epoch:02d}-{validation_loss:.2f}",
                save_last=True,
            )
            early_stop_callback = EarlyStopping(
                monitor="validation_loss",
                patience=args.patience,
                verbose=True,
                mode="min",
            )

            # Training
            trainer = Trainer(
                precision=args.precision,
                logger=logger,
                gpus=args.gpus,
                max_epochs=args.max_epochs_finetune,
                gradient_clip_val=args.gradient_clip_val,
                strategy=DDPStrategy(find_unused_parameters=False),
                callbacks=[checkpoint_callback, early_stop_callback],
                accumulate_grad_batches=args.accumulate,
            )
            use_deterministic_algorithms()
            if trainer.is_global_zero:
                print("### Finetune with smaller lr")
            trainer.fit(module, ckpt_path=finetune_ckpt_path)

        # load best model & save best model as pytorch_model.bin
        ckpt_path = sorted(
            glob(f"{logger.log_dir}/checkpoints/epoch=*.ckpt"),
            key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
        )[-1]
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            state_dict[k[6:]] = state_dict.pop(k)  # "model."
        module.model.load_state_dict(state_dict)
        if trainer.is_global_zero:
            module.model.save_pretrained(logger.log_dir)

        if trainer is not None:
            torch.distributed.destroy_process_group()
            try:
                os.chmod(logger.log_dir, 0o0777)
            except PermissionError as e:
                print(e)

    # Evaluation
    if args.eval_when_train_end and (trainer is None or trainer.is_global_zero):
        if args.skip_train and args.finetune:
            logger = TensorBoardLogger(
                save_dir, name=f"{name}__finetune", version=version
            )

        # Load best model
        ckpt_path = sorted(
            glob(f"{logger.log_dir}/checkpoints/epoch=*.ckpt"),
            key=lambda x: int(x.split("epoch=")[1].split("-")[0]),
        )[-1]
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        for k in list(state_dict.keys()):
            state_dict[k[6:]] = state_dict.pop(k)  # "model."
        module.model.load_state_dict(state_dict)

        # Eval
        trainer = Trainer(
            precision=args.precision, logger=logger, gpus=1, max_epochs=-1
        )
        if "visual_genome" in args.data_path:
            test_dataset = VGDetection(
                data_folder=args.data_path,
                feature_extractor=feature_extractor,
                split=args.split,
            )
        else:
            test_dataset = OIDetection(
                data_folder=args.data_path,
                feature_extractor=feature_extractor,
                split=args.split,
            )
        test_dataloader = DataLoader(
            test_dataset,
            collate_fn=lambda x: collate_fn(x, feature_extractor),
            batch_size=args.eval_batch_size,
            pin_memory=True,
            num_workers=args.num_workers,
            persistent_workers=True,
        )
        if trainer.is_global_zero:
            print("### Evaluation")
        trainer.test(module, dataloaders=test_dataloader)
