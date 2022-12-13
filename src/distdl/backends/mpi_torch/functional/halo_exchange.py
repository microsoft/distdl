__all__ = ["HaloExchangeFunction"]

import numpy as np
import torch
from mpi4py import MPI

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

        ctx.mark_dirty(input)

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
                ## np.copyto(lbb, input.detach()[lbs].cpu().numpy())
                ## cp.copyto(lbb, cp.array(input.detach()[lbs]))
                # TODO: Keep as torch tensor (should be alias or deep copy?)
                # lbb = input[lbs].contiguous()
                lbb.copy_(input[lbs])
            if rbb is not None:
                ## np.copyto(rbb, input.detach()[rbs].cpu().numpy())
                ## cp.copyto(rbb, cp.array(input.detach()[rbs]))
                # TODO: Keep as torch tensor (should be alias or deep copy?)
                # rbb = torch.tensor(input.detach()[rbs], device=input.device, dtype=input.dtype)
                # rbb = input[rbs].contiguous()
                rbb.copy_(input[rbs])

            ltag = 0
            rtag = 1

            lrecv_req = P_x._comm.Irecv(lgb, source=lrank, tag=rtag) if lgb is not None else MPI.REQUEST_NULL
            rrecv_req = P_x._comm.Irecv(rgb, source=rrank, tag=ltag) if rgb is not None else MPI.REQUEST_NULL
            lsend_req = P_x._comm.Isend(lbb.detach(), dest=lrank, tag=ltag) if lbb is not None else MPI.REQUEST_NULL
            rsend_req = P_x._comm.Isend(rbb.detach(), dest=rrank, tag=rtag) if rbb is not None else MPI.REQUEST_NULL

            reqs = [lrecv_req, rrecv_req, lsend_req, rsend_req]
            n_reqs_completed = 0

            while n_reqs_completed < len(reqs):
                status = MPI.Status()
                index = MPI.Request.Waitany(reqs, status)

                if index != MPI.UNDEFINED:
                    if index == 0:
                        # TODO: Should change these lines to zero copy
                        # input[lgs] = torch.tensor(lgb, device=device)
                        input[lgs].copy_(lgb.detach())
                        input[lgs].requires_grad_(input.requires_grad)
                    elif index == 1:
                        # input[rgs] = torch.tensor(rgb, device=device)
                        input[rgs].copy_(rgb.detach())
                        input[rgs].requires_grad_(input.requires_grad)

                n_reqs_completed += 1

        return input

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
                print("lbb.device: ", lbb.device)
            if lgb is not None:
                lgb = lgb.get_view(compute_nd_slice_shape(lgs))
            if rbb is not None:
                rbb = rbb.get_view(compute_nd_slice_shape(rbs))
            if rgb is not None:
                rgb = rgb.get_view(compute_nd_slice_shape(rgs))
            lrank, rrank = neighbor_ranks[i]

            if lgb is not None:
                ## np.copyto(lgb, grad_output.detach()[lgs].cpu().numpy())
                ## cp.copyto(lgb, cp.array(grad_output.detach()[lgs]))
                # TODO: Keep as torch tensor (should be alias or deep copy?)
                # lgb = torch.tensor(grad_output.detach()[lgs], device=grad_output.device)
                lgb.copy_(grad_output[lgs])
                grad_output[lgs] = 0.0
            if rgb is not None:
                ## np.copyto(rgb, grad_output.detach()[rgs].cpu().numpy())
                ## cp.copyto(rgb, cp.array(grad_output.detach()[rgs]))
                # TODO: Keep as torch tensor (should be alias or deep copy?)
                # rgb = torch.tensor(grad_output.detach()[rgs], device=grad_output.device)
                rgb.copy_(grad_output[rgs])
                grad_output[rgs] = 0.0

            ltag = 0
            rtag = 1

            lrecv_req = P_x._comm.Irecv(lbb, source=lrank, tag=rtag) if lbb is not None else MPI.REQUEST_NULL
            rrecv_req = P_x._comm.Irecv(rbb, source=rrank, tag=ltag) if rbb is not None else MPI.REQUEST_NULL
            lsend_req = P_x._comm.Isend(lgb.detach(), dest=lrank, tag=ltag) if lgb is not None else MPI.REQUEST_NULL
            rsend_req = P_x._comm.Isend(rgb.detach(), dest=rrank, tag=rtag) if rgb is not None else MPI.REQUEST_NULL

            reqs = [lrecv_req, rrecv_req, lsend_req, rsend_req]
            n_reqs_completed = 0

            while n_reqs_completed < len(reqs):
                status = MPI.Status()
                index = MPI.Request.Waitany(reqs, status)

                if index != MPI.UNDEFINED:
                    if index == 0:
                        grad_output[lbs] += lbb
                    elif index == 1:
                        grad_output[rbs] += rbb

                n_reqs_completed += 1

        return grad_output, None, None, None, None
