__all__ = ["HaloExchangeFunction"]

import cupy as cp
import torch

from distdl.utilities.slicing import compute_nd_slice_shape
from distdl.utilities.torch import zero_volume_tensor


class HaloExchangeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, P_x, slices, buffers, neighbor_ranks):

        device = input.device
        ctx.slices = slices
        ctx.buffers = buffers
        ctx.neighbor_ranks = neighbor_ranks
        ctx.P_x = P_x
        ctx.device = device

        if not P_x.active:
            return zero_volume_tensor(input.shape[0], device=device)

        # TODO: mark_dirty() is buggy and does not work properly if halo exchange is
        # chained with certain operations like ReLU, MaxPool, etc. For now, we make
        # a memory copy of the input, rather than modifying the halo in place.
        # ctx.mark_dirty(input)
        output = torch.clone(input.detach())

        if P_x.size == 1:
            return input

        dim = P_x.dim
        for i in range(dim):

            lbs, lgs, rbs, rgs = slices[i]
            lbb, lgb, rbb, rgb = buffers[i]
            if lbb is not None:
                lbb = lbb.get_view(compute_nd_slice_shape(lbs))
            if lgb is not None:
                lgb = lgb.get_view(compute_nd_slice_shape(lgs))
            if rbb is not None:
                rbb = rbb.get_view(compute_nd_slice_shape(rbs))
            if rgb is not None:
                rgb = rgb.get_view(compute_nd_slice_shape(rgs))
            lrank, rrank = neighbor_ranks[i]

            if lbb is not None:
                lbb.copy_(output[lbs])
            if rbb is not None:
                rbb.copy_(output[rbs])

            # Communication
            stream = cp.cuda.stream.get_current_stream()
            cp.cuda.nccl.groupStart()
            if lgb is not None:
                P_x._nccl.recv(lgb, lrank, stream=stream)
                event_lgb = cp.cuda.Event()
                event_lgb.record()
            if rgb is not None:
                P_x._nccl.recv(rgb, rrank, stream=stream)
                event_rgb = cp.cuda.Event()
                event_rgb.record()
            if lbb is not None:
                P_x._nccl.send(lbb, lrank, stream=stream)
            if rbb is not None:
                P_x._nccl.send(rbb, rrank, stream=stream)
            cp.cuda.nccl.groupEnd()

            # Wait for receive calls to complete
            if rgb is not None:
                cp.cuda.runtime.eventSynchronize(event_rgb.ptr)
                output[rgs].copy_(rgb.detach())
                output[rgs].requires_grad_(input.requires_grad)

            if lgb is not None:
                cp.cuda.runtime.eventSynchronize(event_lgb.ptr)
                output[lgs].copy_(lgb.detach())
                output[lgs].requires_grad_(input.requires_grad)

        return output.requires_grad_(input.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):

        slices = ctx.slices
        buffers = ctx.buffers
        neighbor_ranks = ctx.neighbor_ranks
        P_x = ctx.P_x
        device = ctx.device

        assert grad_output.device == device

        if not P_x.active:
            return zero_volume_tensor(grad_output.shape[0], device=device), None, None, None, None

        if P_x.size == 1:
            return grad_output, None, None, None, None

        ctx.mark_dirty(grad_output)

        dim = P_x.dim
        for i in reversed(range(dim)):

            lbs, lgs, rbs, rgs = slices[i]
            lbb, lgb, rbb, rgb = buffers[i]
            if lbb is not None:
                lbb = lbb.get_view(compute_nd_slice_shape(lbs))
            if lgb is not None:
                lgb = lgb.get_view(compute_nd_slice_shape(lgs))
            if rbb is not None:
                rbb = rbb.get_view(compute_nd_slice_shape(rbs))
            if rgb is not None:
                rgb = rgb.get_view(compute_nd_slice_shape(rgs))
            lrank, rrank = neighbor_ranks[i]

            if lgb is not None:
                lgb.copy_(grad_output[lgs])
                grad_output[lgs] = 0.0
            if rgb is not None:
                rgb.copy_(grad_output[rgs])
                grad_output[rgs] = 0.0

            stream = cp.cuda.stream.get_current_stream()
            cp.cuda.nccl.groupStart()
            if lbb is not None:
                P_x._nccl.recv(lbb, lrank, stream=stream)
                event_lbb = cp.cuda.Event()
                event_lbb.record()
            if rbb is not None:
                P_x._nccl.recv(rbb, rrank, stream=stream)
                event_rbb = cp.cuda.Event()
                event_rbb.record()
            if lgb is not None:
                P_x._nccl.send(lgb, lrank, stream=stream)
            if rgb is not None:
                P_x._nccl.send(rgb, rrank, stream=stream)
            cp.cuda.nccl.groupEnd()

            # Wait for receive calls to complete
            if lbb is not None:
                cp.cuda.runtime.eventSynchronize(event_lbb.ptr)
                grad_output[lbs] += lbb

            if rbb is not None:
                cp.cuda.runtime.eventSynchronize(event_rbb.ptr)
                grad_output[rbs] += rbb

        return grad_output, None, None, None, None
