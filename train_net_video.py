# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# This file is based on the MaskFormer Training Script from detectron2.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# It has been adapted with custom video instance segmentation handling,
# custom attention mask annealing, and layer-wise learning rate scheduling.
# ---------------------------------------------------------------

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

# Models

from lvmt import (
    YTVISDatasetMapper,
    CocoClipDatasetMapper,
    PanopticDatasetVideoMapper,
    SemanticDatasetVideoMapper,
    YTVISEvaluator,
    VPSEvaluator,
    VSSEvaluator,
    TQPChunkedDataLoader,
    add_lvmt_config,
    build_combined_loader,
    build_detection_train_loader,
    build_detection_test_loader,
)

from lvmt.modeling.warmup_poly_schedule import WarmupPolySchedule
from detectron2.engine.hooks import HookBase
from torch.optim import AdamW
import wandb

class AttentionMaskAnnealingHook(HookBase):

    def mask_annealing(self, start_iter, current_iter, final_iter,dtype,device):
        poly_power = 0.9
        if current_iter < start_iter:
            
            return torch.ones(1, device=device, dtype=dtype)
        elif current_iter >= final_iter:
            return torch.zeros(1, device=device, dtype=dtype)
        else:
            progress = (current_iter - start_iter) / (final_iter - start_iter)
            progress = torch.tensor(progress, device=device, dtype=dtype)
            return (1.0 - progress).pow(poly_power) 

    def after_step(self):
        model = self.trainer.model
        device = model.device
        dtype = model.module.backbone.decoder.attn_mask_probs[0].dtype
        
        if model.module.backbone.decoder.masked_attn_enabled:
            for i in range(model.module.backbone.num_blocks):
                model.module.backbone.decoder.attn_mask_probs[i] = self.mask_annealing(
                    model.module.backbone.start_steps[i],
                    self.trainer.iter,
                    model.module.backbone.end_steps[i],
                    dtype,
                    device,
                )
            for i, prob in enumerate(model.module.backbone.decoder.attn_mask_probs):
                self.trainer.storage.put_scalar(f"attn_mask_prob_{i}", prob.item())
        

        if comm.is_main_process():  # Ensure only the main process logs to W&B
                wandb.log({"trainer/global_step": self.trainer.iter + 1})

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """     
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)

        evaluator_dict = {'vis': YTVISEvaluator, 'vss': VSSEvaluator, 'vps': VPSEvaluator}
        assert cfg.MODEL.BACKBONE.TEST.TASK in evaluator_dict.keys()
        return evaluator_dict[cfg.MODEL.BACKBONE.TEST.TASK](dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        assert len(cfg.DATASETS.DATASET_RATIO) == len(cfg.DATASETS.TRAIN) ==\
               len(cfg.DATASETS.DATASET_NEED_MAP) == len(cfg.DATASETS.DATASET_TYPE)
        mappers = []
        mapper_dict = {
            'video_instance': YTVISDatasetMapper,
            'video_panoptic': PanopticDatasetVideoMapper,
            'video_semantic': SemanticDatasetVideoMapper,
            'image_instance': CocoClipDatasetMapper,
        }
        for d_i, (dataset_name, dataset_type, dataset_need_map) in \
                enumerate(zip(cfg.DATASETS.TRAIN, cfg.DATASETS.DATASET_TYPE, cfg.DATASETS.DATASET_NEED_MAP)):
            if dataset_type not in mapper_dict.keys():
                raise NotImplementedError
            _mapper = mapper_dict[dataset_type]
            mappers.append(
                _mapper(cfg, is_train=True, is_tgt=not dataset_need_map, src_dataset_name=dataset_name, )
            )
        assert len(mappers) > 0, "No dataset is chosen!"

        if len(mappers) == 1:
            mapper = mappers[0]
            loader = build_detection_train_loader(cfg, mapper=mapper, dataset_name=cfg.DATASETS.TRAIN[0])
        else:
            loaders = [
                build_detection_train_loader(cfg, mapper=mapper, dataset_name=dataset_name)
                for mapper, dataset_name in zip(mappers, cfg.DATASETS.TRAIN)
            ]
            loader = build_combined_loader(cfg, loaders, cfg.DATASETS.DATASET_RATIO)

        # Wrap with TQP chunking if enabled
        tqp_chunk_size = cfg.MODEL.BACKBONE.TQP_CHUNK_SIZE
        if tqp_chunk_size > 0:
            total_frames = cfg.INPUT.SAMPLING_FRAME_NUM
            loader = TQPChunkedDataLoader(loader, tqp_chunk_size, total_frames)

        return loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, dataset_type):
        mapper_dict = {
            'video_instance': YTVISDatasetMapper,
            'video_panoptic': PanopticDatasetVideoMapper,
            'video_semantic': SemanticDatasetVideoMapper,
        }
        if dataset_type not in mapper_dict.keys():
            raise NotImplementedError
        mapper = mapper_dict[dataset_type](cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        encoder_param_names = {
            n for n, _ in model.backbone.encoder.backbone.named_parameters()
        }
        backbone_param_groups = []
        other_param_groups = []
        backbone_blocks = len(model.backbone.encoder.backbone.blocks)
        block_i = backbone_blocks

        for name, param in reversed(list(model.named_parameters())):
            if not param.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            if name.replace("backbone.encoder.backbone.", "") in encoder_param_names:
                name_list = name.split(".")
                is_block = False
                for i, key in enumerate(name_list):
                    if key == "blocks":
                        block_i = int(name_list[i + 1])
                        is_block = True
                if is_block or block_i == 0:
                    lr *= cfg.SOLVER.LLRD ** (backbone_blocks - 1 - block_i)
                backbone_param_groups.append(
                    {"params": [param], "lr": lr, "name": name}
                )
            else:
                other_param_groups.append(
                    {"params": [param], "lr": cfg.SOLVER.BASE_LR, "name": name}
                )

        param_groups = backbone_param_groups + other_param_groups
        optimizer = AdamW(param_groups, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    
        return optimizer

    def run_step(self):
        """Override run_step to handle dataloader-based TQP chunking.
        
        The TQPChunkedDataLoader yields chunks sequentially. This method
        consumes ALL chunks of a video in one call, so self.iter increments
        once per video (= once per optimizer step). This keeps LR schedule,
        eval/checkpoint periods, and ETA correct without config multipliers.
        
        Each chunk is a standard 5-frame batch — single forward, single backward.
        """
        import time
        from contextlib import nullcontext
        import torch.distributed as dist

        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        # Check if this is a TQP chunk (dataloader injects metadata)
        chunk_idx = data[0].get("tqp_chunk_idx", None)
        if chunk_idx is None:
            # No TQP — standard training step
            self._trainer.run_step()
            return

        num_chunks = data[0]["tqp_num_chunks"]
        assert chunk_idx == 0, f"Expected first chunk (idx=0), got idx={chunk_idx}"

        self._trainer.optimizer.zero_grad()
        accumulated_losses = {}

        for c in range(num_chunks):
            if c > 0:
                data = next(self._trainer._data_loader_iter)

            # Use no_sync for ALL chunks — some parameters (e.g. q.weight)
            # only participate in chunk 0's loss, which would trigger DDP's
            # unused-param error on later chunks. We manually allreduce after.
            sync_ctx = self.model.no_sync() if hasattr(self.model, 'no_sync') else nullcontext()

            with sync_ctx:
                loss_dict = self.model(data)
                losses = sum(loss_dict.values())
                (losses / num_chunks).backward()

            # Detach propagation hidden state for next chunk
            actual_model = self.model.module if hasattr(self.model, 'module') else self.model
            decoder = actual_model.backbone.decoder
            if hasattr(decoder, 'last_query_embed') and decoder.last_query_embed is not None:
                decoder.last_query_embed = decoder.last_query_embed.detach()

            # Accumulate losses across chunks (each scaled by 1/num_chunks)
            for k, v in loss_dict.items():
                val = v.detach() / num_chunks
                if k in accumulated_losses:
                    accumulated_losses[k] = accumulated_losses[k] + val
                else:
                    accumulated_losses[k] = val

        # Manual gradient allreduce (since we used no_sync for all chunks)
        if dist.is_initialized() and dist.get_world_size() > 1:
            for param in self.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(dist.get_world_size())

        # Metrics logging with accumulated losses from all chunks
        if self._trainer.async_write_metrics:
            self._trainer.concurrent_executor.submit(
                self._trainer._write_metrics, accumulated_losses,
                data_time, iter=self.iter
            )
        else:
            self._trainer._write_metrics(
                accumulated_losses, data_time, iter=self.iter
            )

        self._trainer.optimizer.step()

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return WarmupPolySchedule(
            optimizer,
            warmup_steps=cfg.SOLVER.WARMUP_STEPS,
            total_steps=cfg.SOLVER.MAX_ITER,
            poly_power=cfg.SOLVER.POLY_POWER,
        )

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        from torch.cuda.amp import autocast
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            dataset_type = cfg.DATASETS.DATASET_TYPE_TEST[idx]
            data_loader = cls.build_test_loader(cfg, dataset_name, dataset_type)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warning(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with autocast():
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_lvmt_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="lvmt")

    if comm.get_rank() == 0:
        wandb.init(
            id=cfg.OUTPUT_DIR.rstrip("/").rsplit("/", 1)[-1],
            project="lvmt",
            config=cfg,
            sync_tensorboard=True,
            resume="allow",
            dir=cfg.OUTPUT_DIR,
            settings=wandb.Settings(start_method="fork"),
        )

        wandb.define_metric("trainer/global_step")
        wandb.define_metric("*", step_metric="trainer/global_step")

    return cfg


def _maybe_compile_backbone(cfg, model):
    """Apply torch.compile to encoder blocks only (not the decoder).
    
    The decoder contains ConvTranspose2d (ScaleBlock), frame-by-frame loops,
    and mutable GRU state that cause stride mismatches in inductor's backward.
    Compiling individual encoder blocks avoids these issues while capturing
    the main compute benefit (frozen ViT blocks).
    """
    if cfg.MODEL.BACKBONE.TORCH_COMPILE:
        compile_mode = cfg.MODEL.BACKBONE.TORCH_COMPILE_MODE
        logger = logging.getLogger(__name__)
        # Unwrap DDP if needed
        raw_model = model.module if hasattr(model, "module") else model
        encoder_blocks = raw_model.backbone.encoder.backbone.blocks
        logger.info(
            f"Applying torch.compile to {len(encoder_blocks)} encoder blocks "
            f"(mode={compile_mode})"
        )
        for i in range(len(encoder_blocks)):
            encoder_blocks[i] = torch.compile(
                encoder_blocks[i],
                mode=compile_mode,
                fullgraph=True,
                dynamic=True,
            )
    return model

# def _maybe_compile_backbone(cfg, model):
#     """Apply torch.compile to encoder and decoder blocks individually.

#     Skips decoder.upscale (ConvTranspose2d stride mismatch with dynamic shapes)
#     and decoder.query_gru (mutable hidden state across frames).
#     """
#     if cfg.MODEL.BACKBONE.TORCH_COMPILE:
#         compile_mode = cfg.MODEL.BACKBONE.TORCH_COMPILE_MODE
#         logger = logging.getLogger(__name__)
#         raw_model = model.module if hasattr(model, "module") else model

#         # --- Encoder blocks (frozen ViT — main compute cost) ---
#         encoder_blocks = raw_model.backbone.encoder.backbone.blocks
#         logger.info(
#             f"Applying torch.compile to {len(encoder_blocks)} encoder blocks "
#             f"(mode={compile_mode})"
#         )
#         for i in range(len(encoder_blocks)):
#             encoder_blocks[i] = torch.compile(
#                 encoder_blocks[i],
#                 mode=compile_mode,
#                 fullgraph=True,
#                 dynamic=True,
#             )

#         # --- Decoder transformer blocks (safe, no convolutions/GRU) ---
#         decoder = raw_model.backbone.decoder
#         if hasattr(decoder, "blocks"):
#             logger.info(
#                 f"Applying torch.compile to {len(decoder.blocks)} decoder blocks "
#                 f"(mode={compile_mode})"
#             )
#             for i in range(len(decoder.blocks)):
#                 decoder.blocks[i] = torch.compile(
#                     decoder.blocks[i],
#                     mode=compile_mode,
#                     fullgraph=True,
#                     dynamic=True,
#                 )

#         # Skipping decoder.upscale (ConvTranspose2d) and
#         # decoder.query_gru (mutable recurrent state)
#     return model

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        # Apply fused QKV BEFORE compiling so the compiled graph sees merged weights
        raw_model = model.module if hasattr(model, "module") else model
        if cfg.MODEL.BACKBONE.get("FUSED_QKV", False):
            raw_model.backbone._maybe_apply_fused_qkv()
        model = _maybe_compile_backbone(cfg, model)

        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            raise NotImplementedError
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.register_hooks([AttentionMaskAnnealingHook()])
    if args.resume:
        _maybe_compile_backbone(cfg, trainer.model)
        trainer.resume_or_load(resume=True)
    else:
        trainer.resume_or_load(resume=False)
        _maybe_compile_backbone(cfg, trainer.model)
    
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()   
    if not args.dist_url:
        args.dist_url = 'tcp://127.0.0.1:50263'
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
