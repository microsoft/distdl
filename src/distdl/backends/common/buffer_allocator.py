import numpy as np
from distdl.utilities.slicing import compute_nd_slice_shape
from distdl.backends.common.convert import convert_torch_to_model_dtype


def allocate_repartition_buffers(buffer_manager, P_x_to_y_overlaps, P_y_to_x_overlaps, dtype):
    r"""Allocator for data movement buffers.

    Parameters
    ----------
    P_x_to_y_overlaps : list
        List of tuples (sl, sh, partner) for which current worker needs a send
        buffer.
    P_y_to_x_overlaps : list
        List of tuples (sl, sh, partner) for which current worker needs a
        receive buffer.
    dtype :
        Data type of input/output tensors.

    """

    model_dtype = convert_torch_to_model_dtype(dtype)

    # count the buffers we need
    count = 0
    for sl, sh, partner in P_x_to_y_overlaps:
        if sl is not None and partner != "self":
            count += 1
    for sl, sh, partner in P_y_to_x_overlaps:
        if sl is not None and partner != "self":
            count += 1

    buffers = buffer_manager.request_buffers(count, dtype=model_dtype)

    i = 0

    # For each necessary copy, allocate send buffers.
    P_x_to_y_buffers = []
    for sl, sh, partner in P_x_to_y_overlaps:
        buff = None
        if sl is not None and partner != "self":
            buff = buffers[i]
            buff.allocate_view(sh)
            i += 1

        P_x_to_y_buffers.append(buff)

    # For each necessary copy, allocate receive buffers.
    P_y_to_x_buffers = []
    for sl, sh, partner in P_y_to_x_overlaps:
        buff = None
        if sl is not None and partner != "self":
            buff = buffers[i]
            buff.allocate_view(sh)
            i += 1

        P_y_to_x_buffers.append(buff)

    return P_x_to_y_buffers, P_y_to_x_buffers


def allocate_halo_exchange_buffers(buffer_manager, slices, recv_buffer_shape, send_buffer_shape, dtype):

    dim = len(slices)
    buffers_out = []
    model_dtype = convert_torch_to_model_dtype(dtype)

    # Each dimension is performed sequentially.  Thus, we only need 4 buffers:
    # one each for left and right bulk and ghost.  The buffer shapes will be
    # viewed correctly for each dimension.
    count = 4

    buffers = buffer_manager.request_buffers(count, dtype=model_dtype)

    for i in range(dim):
        lbb_shape = compute_nd_slice_shape(slices[i][0]) if send_buffer_shape[i, 0] > 0 else 0
        lgb_shape = compute_nd_slice_shape(slices[i][1]) if recv_buffer_shape[i, 0] > 0 else 0
        rbb_shape = compute_nd_slice_shape(slices[i][2]) if send_buffer_shape[i, 1] > 0 else 0
        rgb_shape = compute_nd_slice_shape(slices[i][3]) if recv_buffer_shape[i, 1] > 0 else 0

        buffers_i = list()

        for j, shape in enumerate([lbb_shape, lgb_shape, rbb_shape, rgb_shape]):
            buff = None
            if np.prod(shape) > 0:
                buff = buffers[j]
                buff.allocate_view(shape)

            buffers_i.append(buff)

        buffers_out.append(buffers_i)

    return buffers_out
