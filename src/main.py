import determinism  # noqa

determinism.i_do_nothing_but_dont_remove_me_otherwise_things_break()  # noqa

import argparse
import bisect
import copy
import os
import sys
import time
from argparse import ArgumentParser

import torch
import wandb
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import PeriodicCheckpointer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
import losses
import utils
from eval_utils import eval_unsupmf, get_unsup_image_viz, get_vis_header
from trainer import Trainer,setup
from ourcheckpointer import OurCheckpointer
import subprocess
import atexit
global masks_raw_prior

masks_raw_prior = None
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
logger = utils.log.getLogger('gwm')

def freeze(module, set=False):
    for param in module.parameters():
        param.requires_grad = set

def run_new_command(cfg,iou):
    new_command = ["python", "main.py", "--resume_path",cfg.OUTPUT_DIR+"/checkpoints/checkpoint_best.pth","--iou_best",str(iou)]  # 替换为你想要执行的命令和参数
    subprocess.Popen(new_command)


def main(args):
    cfg = setup(args)
    logger.info(f"Called as {' '.join(sys.argv)}")
    logger.info(f'Output dir {cfg.OUTPUT_DIR}')
    logger.info(f'GWM.FOCUS_DATA: {cfg.GWM.FOCUS_DATA}')
    random_state = utils.random_state.PytorchRNGState(seed=cfg.SEED).to(torch.device(cfg.MODEL.DEVICE))
    random_state.seed_everything()
    utils.log.checkpoint_code(cfg.OUTPUT_DIR)

    if not cfg.SKIP_TB:
        writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    else:
        writer = None

    # initialize model
    logger.info("model: "+cfg.GWM.MODEL)
    model = Trainer.build_model(cfg)
    optimizer = Trainer.build_optimizer(cfg, model)
    scheduler = Trainer.build_lr_scheduler(cfg, optimizer)

    logger.info(f'Optimiser is {type(optimizer)}')


    checkpointer = OurCheckpointer(model,
                                         save_dir=os.path.join(cfg.OUTPUT_DIR, 'checkpoints'),
                                         random_state=random_state,
                                         optimizer=optimizer,
                                         scheduler=scheduler)
    periodic_checkpointer = PeriodicCheckpointer(checkpointer=checkpointer,
                                                 period=cfg.SOLVER.CHECKPOINT_PERIOD,
                                                 max_iter=cfg.SOLVER.MAX_ITER,
                                                 max_to_keep=None if cfg.FLAGS.KEEP_ALL else 5,
                                                 file_prefix='checkpoint')
    checkpoint = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume_path is not None)
    iteration = 0 if args.resume_path is None else checkpoint['iteration']

    train_loader, val_loader = config.loaders(cfg)
    # batch size 1  batch size 1
    # overfit single batch for debug
    # sample = next(iter(loader))
    criterions = {}
    if cfg.GWM.LOSS_MULT.OPT>0.0:
        criterions['opticalflow'] = (losses.OpticalFlowLoss_3d(cfg, model), cfg.GWM.LOSS_MULT.OPT*cfg.GWM.LOSS_MULT.GLOBAL, lambda x: 1)
    if cfg.GWM.LOSS_MULT.OPT2>0.0:
        criterions['opticalflow_formula2'] = (losses.OpticalFlowLossFormula2(cfg, model), cfg.GWM.LOSS_MULT.OPT2*cfg.GWM.LOSS_MULT.GLOBAL, lambda x: 1)
    if cfg.GWM.LOSS_MULT.TRAJ>0.0:
        criterions['tragectory_3d'] = (losses.TrajectoryLoss_3d(cfg, model), cfg.GWM.LOSS_MULT.TRAJ*cfg.GWM.LOSS_MULT.GLOBAL, lambda x: 1)
    if cfg.GWM.LOSS_MULT.TRAJ3>0.0:
        criterions['tragectory_formula3'] = (losses.TrajectoryLossFormula3(cfg, model), cfg.GWM.LOSS_MULT.TRAJ3*cfg.GWM.LOSS_MULT.GLOBAL, lambda x: 1)


    criterion = losses.CriterionDict(criterions)

    if args.eval_only:
        if len(val_loader.dataset) == 0:
            logger.error("Training dataset: empty")
            sys.exit(0)
        model.eval()
        writer_train = writer
        if cfg.ABLATION.NAME != "default":
            writer_train = None
        iou = eval_unsupmf(cfg=cfg, val_loader=val_loader, model=model, criterion=criterion, writer=writer_train,
                           writer_iteration=iteration)
        logger.info(f"Results: iteration: {iteration} IOU = {iou}")
        return
    if len(train_loader.dataset) == 0:
        logger.error("Training dataset: empty")
        sys.exit(0)

    logger.info(
        f'Start of training: dataset {cfg.GWM.DATASET},'
        f' train {len(train_loader.dataset)}, val {len(val_loader.dataset)},'
        f' device {model.device}, keys {cfg.GWM.SAMPLE_KEYS}, '
        f'multiple flows {cfg.GWM.USE_MULT_FLOW}')

    iou_best = args.iou_best
    iou_train_best = args.iou_best
    timestart = time.time()

    total_iter = cfg.TOTAL_ITER if cfg.TOTAL_ITER else cfg.SOLVER.MAX_ITER  # early stop
    with torch.autograd.set_detect_anomaly(cfg.DEBUG) and \
         tqdm(initial=iteration, total=total_iter, disable=utils.environment.is_slurm()) as pbar:
        while iteration < total_iter:
            for sample in train_loader:

                if cfg.MODEL.META_ARCHITECTURE != 'UNET' and cfg.FLAGS.UNFREEZE_AT:
                    if hasattr(model.backbone, 'frozen_stages'):
                        assert cfg.MODEL.BACKBONE.FREEZE_AT == -1, f"MODEL initial parameters forced frozen"
                        stages = [s for s, m in cfg.FLAGS.UNFREEZE_AT]
                        milest = [m for s, m in cfg.FLAGS.UNFREEZE_AT]
                        pos = bisect.bisect_right(milest, iteration) - 1
                        if pos >= 0:
                            curr_setting = model.backbone.frozen_stages
                            if curr_setting != stages[pos]:
                                logger.info(f"Updating backbone freezing stages from {curr_setting} to {stages[pos]}")
                                model.backbone.frozen_stages = stages[pos]
                                model.train()
                    else:
                        assert cfg.MODEL.BACKBONE.FREEZE_AT == -1, f"MODEL initial parameters forced frozen"
                        stages = [s for s, m in cfg.FLAGS.UNFREEZE_AT]
                        milest = [m for s, m in cfg.FLAGS.UNFREEZE_AT]
                        pos = bisect.bisect_right(milest, iteration) - 1
                        freeze(model, set=False)
                        freeze(model.sem_seg_head.predictor, set=True)
                        if pos >= 0:
                            stage = stages[pos]
                            if stage <= 2:
                                freeze(model.sem_seg_head, set=True)
                            if stage <= 1:
                                freeze(model.backbone, set=True)
                        model.train()

                else:
                    logger.debug_once(f'Unfreezing disabled schedule: {cfg.FLAGS.UNFREEZE_AT}')

                sample = [e for s in sample for e in s]
                flow_key = 'flow'
                raw_sem_seg = False
                if cfg.GWM.FLOW_RES is not None:
                    flow_key = 'flow_big'
                    raw_sem_seg = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME == 'MegaBigPixelDecoder'

                flow = torch.stack([x[flow_key].to(model.device) for x in sample]).clip(-20, 20)
                logger.debug_once(f'flow shape: {flow.shape}')
                preds = model.forward_base(sample, keys=cfg.GWM.SAMPLE_KEYS, get_eval=True, raw_sem_seg=raw_sem_seg)
                masks_raw = torch.stack([x['sem_seg'] for x in preds], 0)
                logger.debug_once(f'mask shape: {masks_raw.shape}')
                masks_softmaxed_list = [torch.softmax(masks_raw, dim=1)]


                total_losses = []
                log_dicts = []
                for mask_idx, masks_softmaxed in enumerate(masks_softmaxed_list):

                    loss, log_dict = criterion(sample, flow, masks_softmaxed, iteration)

                    if cfg.GWM.USE_MULT_FLOW:
                        flow2 = torch.stack([x[flow_key + '_2'].to(model.device) for x in sample]).clip(-20, 20)
                        other_loss, other_log_dict = criterion(sample, flow2, masks_softmaxed, iteration)
                        loss = loss / 2 + other_loss / 2
                        for k, v in other_log_dict.items():
                            log_dict[k] = other_log_dict[k] / 2 + v / 2
                    total_losses.append(loss)
                    log_dicts.append(log_dict)

                loss_ws = cfg.GWM.LOSS_MULT.HEIR_W
                total_w = float(sum(loss_ws[:len(total_losses)]))
                log_dict = {}
                if len(total_losses) == 1:
                    log_dict = log_dicts[0]
                    loss = total_losses[0]
                else:
                    loss = 0
                    for i, (tl, w, ld) in enumerate(zip(total_losses, loss_ws, log_dicts)):
                        for k, v in ld.items():
                            log_dict[f'{k}_{i}'] = v * w / total_w
                        loss += tl * w / total_w

                train_log_dict = {f'train/{k}': v for k, v in log_dict.items()}
                del log_dict
                train_log_dict['train/learning_rate'] = optimizer.param_groups[-1]['lr']
                train_log_dict['train/loss_total'] = loss.item()
                for k, v in train_log_dict.items():
                    if writer:
                        writer.add_scalar(k, v, iteration + 1)

                optimizer.zero_grad()
                masks_raw.retain_grad()
                masks_softmaxed_list[0].retain_grad()
                loss.backward()
                global masks_raw_prior
                if masks_raw_prior is not None:
                    writer.add_scalar(f"masks_raw/diff_mean", (masks_raw-masks_raw_prior).mean(), iteration + 1)
                    writer.add_scalar(f"masks_raw/diff_std", (masks_raw-masks_raw_prior).std(), iteration + 1)
                writer.add_histogram(f"masks_raw/gradient", masks_raw.grad, iteration + 1)
                writer.add_histogram(f"masks_raw/value", masks_raw, iteration + 1)
                writer.add_histogram(f"masks_raw/softmaxed_gradient", masks_softmaxed_list[0].grad, iteration + 1)
                writer.add_histogram(f"masks_raw/softmaxed_value", masks_softmaxed_list[0], iteration + 1)
                masks_raw_prior = masks_raw
                optimizer.step()
                scheduler.step()

                pbar.set_postfix(loss=loss.item())
                pbar.update()

                # Sanity check for RNG state
                if (iteration + 1) % 1000 == 0 or iteration + 1 in {1, 50}:
                    logger.info(
                        f'Iteration {iteration + 1}. RNG outputs {utils.random_state.get_randstate_magic_numbers(model.device)}')

                if cfg.DEBUG or (iteration + 1) % 100 == 0:
                    logger.info(
                        f'Iteration: {iteration + 1}, time: {time.time() - timestart:.01f}s, loss: {loss.item():.02f}.')



                    if cfg.WANDB.ENABLE:
                        wandb.log(train_log_dict, step=iteration + 1)

                if (iteration + 1) % cfg.LOG_FREQ == 0 or (iteration + 1) in [1, 50]:
                    model.eval()
                    if writer:
                        flow = torch.stack([x['flow'].to(model.device) for x in sample]).clip(-20, 20)
                        image_viz, header_text = get_unsup_image_viz(model, cfg, sample, criterion)
                        header = get_vis_header(image_viz.size(2), cfg.VISUALIZE.RESOLUTION[1], header_text)
                        image_viz = torch.cat([header, image_viz], dim=1)
                        writer.add_image('train/images', image_viz, iteration + 1)
                    if cfg.WANDB.ENABLE and (iteration + 1) % 2500 == 0:
                        image_viz = get_unsup_image_viz(model, cfg, sample)
                        wandb.log({'train/viz': wandb.Image(image_viz.float())}, step=iteration + 1)
                    writer_train = writer
                    if cfg.ABLATION.NAME != "default":
                        writer_train = None
                    if cfg.MENTOR_TRAIN:
                        if iou_train := eval_unsupmf(cfg=cfg, val_loader=train_loader, model=model, criterion=criterion,
                                            writer=writer_train, writer_iteration=iteration + 1, use_wandb=cfg.WANDB.ENABLE,mode="train"):
                            if cfg.SOLVER.CHECKPOINT_PERIOD and iou_train > iou_train_best:
                                iou_train_best = iou_train
                                if not args.wandb_sweep_mode:
                                    checkpointer.save(name='checkpoint_best', iteration=iteration + 1, loss=loss,
                                                    iou=iou_train_best)
                                logger.info(f'New best IoU {iou_train_best:.02f} after iteration {iteration + 1}')
                            if cfg.GWM.REBOOST_WHEN_DECREASE:
                                logger.info(f'Current IoU {iou_train:.02f} is less than best IoU {iou_train_best:.02f} after iteration {iteration + 1}')
                                # load the last best model
                                run_new_command(cfg,iou_train_best)
                                sys.exit()
                            if cfg.WANDB.ENABLE:
                                wandb.log({'train/IoU_best': iou_train_best}, step=iteration + 1)
                            if writer:
                                writer.add_scalar('train/IoU_best', iou_train_best, iteration + 1)
                    if cfg.MENTOR_VAL:
                        if iou := eval_unsupmf(cfg=cfg, val_loader=val_loader, model=model, criterion=criterion,
                                            writer=writer_train, writer_iteration=iteration + 1, use_wandb=cfg.WANDB.ENABLE,mode="eval"):
                            if cfg.SOLVER.CHECKPOINT_PERIOD and iou > iou_best:
                                iou_best = iou
                                if not args.wandb_sweep_mode:
                                    checkpointer.save(name='checkpoint_best', iteration=iteration + 1, loss=loss,
                                                    iou=iou_best)
                                logger.info(f'New best IoU {iou_best:.02f} after iteration {iteration + 1}')
                            if cfg.GWM.REBOOST_WHEN_DECREASE:
                                logger.info(f'Current IoU {iou:.02f} is less than best IoU {iou_best:.02f} after iteration {iteration + 1}')
                                # load the last best model
                                run_new_command(cfg,iou_best)
                                sys.exit()
                            if cfg.WANDB.ENABLE:
                                wandb.log({'eval/IoU_best': iou_best}, step=iteration + 1)
                            if writer:
                                writer.add_scalar('eval/IoU_best', iou_best, iteration + 1)


                    model.train()

                periodic_checkpointer.step(iteration=iteration + 1, loss=loss)

                iteration += 1
                timestart = time.time()

    #this is the end of the training loop
    #save learning rate to txt file
    if cfg.ABLATION.RESULTSAVEPATH is not None:
        savedir_list = cfg.ABLATION.RESULTSAVEPATH.split('/')
        savedir = '/'.join(savedir_list[:-1])
        os.makedirs(savedir, exist_ok=True)
        with open(cfg.ABLATION.RESULTSAVEPATH, 'w') as f:
            f.write(str(iou_best))

def get_argparse_args():
    parser = ArgumentParser()
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--use_wandb', dest='wandb_sweep_mode', action='store_true')  # for sweep
    parser.add_argument('--config-file', type=str,
                        default='configs/Unet.yaml')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--iou_best', type=float, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
             "See config references at "
             "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = get_argparse_args().parse_args()
    if args.resume_path:
        args.config_file = "/".join(args.resume_path.split('/')[:-2]) + '/config.yaml'
        print(args.config_file)
    main(args)
