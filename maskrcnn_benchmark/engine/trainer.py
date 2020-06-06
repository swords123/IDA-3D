# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

from apex import amp

#import visdom
#import numpy as np

def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()
    norm = (clip_norm / max(totalnorm, clip_norm))
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    local_rank
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    '''
    if local_rank == 0:
        vis = visdom.Visdom()
        win_1 = vis.line(X=np.array([start_iter]),Y=np.array([0]))
        win_2 = vis.line(X=np.array([start_iter]),Y=np.array([0]))
        win_3 = vis.line(X=np.array([start_iter]),Y=np.array([0]))
        win_4 = vis.line(X=np.array([start_iter]),Y=np.array([0]))
        win_5 = vis.line(X=np.array([start_iter]),Y=np.array([0]))
        win_6 = vis.line(X=np.array([start_iter]),Y=np.array([0]))
        win_7 = vis.line(X=np.array([start_iter]),Y=np.array([0]))
        win_8 = vis.line(X=np.array([start_iter]),Y=np.array([0]))
        win_9 = vis.line(X=np.array([start_iter]),Y=np.array([0]))
    '''

    for iteration, (images_left, images_right, targets, calib, _) in enumerate(data_loader, start_iter):
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images_left = images_left.to(device)
        images_right = images_right.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images_left, images_right, targets, calib=calib)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
            clip_gradient(model, 10.)
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

            '''
            if local_rank == 0:
                vis.line(X=[iteration], Y=[meters.loss.median], win=win_1, update='append', opts=dict(title='loss'))
                vis.line(X=[iteration], Y=[meters.loss_dimention.median], win=win_2, update='append', opts=dict(title='loss_dimention'))
                vis.line(X=[iteration], Y=[meters.rot_regression_loss.median], win=win_3, update='append', opts=dict(title='rot_regression_loss'))
                vis.line(X=[iteration], Y=[meters.loss_cost_depth.median], win=win_4, update='append', opts=dict(title='loss_cost_depth'))
                vis.line(X=[iteration], Y=[meters.loss_center.median], win=win_5, update='append', opts=dict(title='loss_center'))
                
                vis.line(X=[iteration], Y=[meters.loss_box_reg.median], win=win_6, update='append', opts=dict(title='loss_box_reg'))
                vis.line(X=[iteration], Y=[meters.loss_classifier.median], win=win_7, update='append', opts=dict(title='loss_classifier'))
                vis.line(X=[iteration], Y=[meters.loss_objectness.median], win=win_8, update='append', opts=dict(title='loss_objectness'))
                vis.line(X=[iteration], Y=[meters.loss_rpn_box_reg.median], win=win_9, update='append', opts=dict(title='loss_rpn_box_reg'))
            '''
            
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
