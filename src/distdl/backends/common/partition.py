import numpy as np
from mpi4py import MPI

from distdl import backends
from distdl.backends.common.compare import check_identical_comm
from distdl.backends.common.compare import check_identical_group
from distdl.backends.common.compare import check_null_comm
from distdl.backends.common.compare import check_null_group
from distdl.backends.common.compare import check_null_rank
from distdl.utilities.debug import print_sequential
from distdl.utilities.dtype import intID_to_numpy_dtype_dict
from distdl.utilities.dtype import numpy_to_intID_dtype_dict
from distdl.utilities.index_tricks import cartesian_index_c
from distdl.utilities.index_tricks import cartesian_index_f
from distdl.utilities.slicing import assemble_index_filter
from distdl.utilities.slicing import filtered_range_index


class MPIPartition:
    r"""MPI-based implementation of unstructured tensor partition.

    This class provides the user interface for the MPI-based implementation of
    tensor partitions.  The MPI interface is provided by ``mpi4py``.

    Teams of workers are managed using MPI Groups and communication and data
    movement occurs within the MPI Communicator associated with that group.

    To handle situations where data movement occurs between two partitions, a
    ``root`` MPI Communicator is stored.  This communicator allows the
    creation of a union of the two partitions, so that collectives can be used
    across that union without impacting other workers in the root
    communicator.

    Most MPI-based tools work based on communicators, so the communicator is
    the primary object used to create the partition from the user perspective,
    but internally the group is the important object.

    If the communicator is a null communicator, a nullified partition is
    created, which is indicated by the ``active`` member.  Workers with
    ``active`` set to ``False`` have no mechanism to communicate with the rest
    of the team, except through the ``root`` communicator.  The ``active``
    flag is used within layers to determine which work and data movement needs
    to be performed by the current worker.

    Parameters
    ----------
    comm : MPI communicator, optional
        MPI Communicator to create the partition from.
    group : MPI group, optional
        MPI Group associated with ``comm``.
    root : MPI communicator, optional
        MPI communicator tracking the original communicator used to create
        all ancestors of this partition.
    device : Device name, optional
        This field is inherited through all the partition's descendents to
        make sure that the tensors are allocated on the same device.
    initialize_backend_comm : bool, optional
        Flag to create the backend communicator.

    Attributes
    ----------
    _comm : MPI communicator
        MPI Communicator for this partition.
    _root : MPI communicator
        MPI communicator tracking the original communicator used to create
        all ancestors of this partition.
    active : boolean
        Indicates if the worker participated in work and data movement for
        the partition.
    group : MPI group, optional
        MPI Group associated with ``comm``.
    rank :
        Lexicographic identifier for the worker in the partition.
    size :
        Number of workers active in this team for this partition.
    shape :
        Number of workers in each Cartesian dimension.
    index :
        Lexicographic identifiers in each Cartesian dimension.
    """

    def __init__(self, comm=MPI.COMM_NULL, group=MPI.GROUP_NULL, root=None, device=None,
                 requested_device=None, initialize_backend_comm=False):

        # MPI communicator to communicate within
        self._comm = comm
        self._nccl = None

        # root tracks a root communicator: any subpartition from this one
        # will have the same root as this one.
        # If it is not specified, take the current communicator to be the root.
        if root is None:
            self._root = comm
        else:
            self._root = root

        # If the communicator is not null, this worker is active and can
        # gather remaining information.  Otherwise, the worker is inactive
        # and members should be nullified.1
        if self._comm != MPI.COMM_NULL:
            self.active = True
            if group == MPI.GROUP_NULL:
                self._group = comm.Get_group()
            else:
                self._group = group
            self.rank = self._comm.Get_rank()
            self.size = self._comm.Get_size()
            self.index = self.rank
        else:
            self.active = False
            self._group = group
            self.rank = MPI.PROC_NULL
            self.size = -1
            self.index = None

        # For convenience, sometimes unstructured partitions can be treated
        # like Cartesian partitions, so we need to give a shape and index.
        self.shape = np.array([self.size], dtype=int)
        self.dim = len(self.shape)

        if device is not None:
            self.device = device
        else:
            if not self._comm == MPI.COMM_NULL:
                self.set_device()
            self.device = backends.backend.get_device()

    def initialize_backend_comm(self):
        if self._comm != MPI.COMM_NULL and self.active and backends.backend.__name__ == 'nccl_cupy':
            if self._nccl is None:

                # Check if NCCL comm for current communicator already exists.
                root_group = self._root.Get_group()
                try:    # temporary hot fix for mpi4py API change
                    global_rank = root_group.Translate_ranks(self._group, [self._group.rank])
                except Exception:
                    global_rank = root_group.Translate_ranks([self._group.rank], self._group)   # (for 4.0.0 and above)
                all_ranks = str(self.allgather_data(global_rank).reshape(-1))
                hash_comm = hash(all_ranks)

                # If not, create new NCCL comm and add to the store. Otherwise, use existing comm.
                if hash_comm in backends.backend.comm_store.keys():
                    self._nccl = backends.backend.comm_store[hash_comm]
                else:
                    self._nccl = backends.backend.nccl_comm.NCCLBackend(self._comm, self.size, self.rank)
                    backends.backend.comm_store.update({hash_comm: self._nccl})

    def set_device(self):
        self.device = backends.backend.set_device(rank=self.rank)

    def deactivate(self):
        r"""Deactivates this partition by releasing any resources and
        nullifying any other properties.

        Releases the MPI Communicator and Group that define this partition.

        All `MPIPartition*.create_*` routines act as factories, acquiring
        communicator and group resources for a new partition instance.

        Warning
        -------
        Assumes that the resources released are *able to be released*.  There
        is no way to guarantee safety.  Only call this if you know it is safe
        to release the resources.

        """

        if self.active:
            if (self._comm != MPI.COMM_NULL and  # noqa W504
                self._comm != MPI.COMM_WORLD):  # noqa E129
                self._comm.Free()

            if self._group != MPI.GROUP_NULL:
                self._group.Free()

            self._comm = MPI.COMM_NULL
            self._group = MPI.GROUP_NULL

            self.active = False
            self.rank = MPI.PROC_NULL
            self.size = -1

            self.shape = np.array([self.size], dtype=int)
            self.dim = len(self.shape)
            self.index = self.rank

    def __eq__(self, other):
        r"""Equality comparator for partitions.

        Parameters
        ----------
        other : MPIPartition
            Partition to compare with.

        Returns
        -------
        ``True`` if ``group`` and ``comm`` are *identical*, in the sense of
        ``MPI_IDENT``, otherwise ``False``.

        """

        # MPI_COMM_NULL is not a valid argument to MPI_Comm_compare, per the
        # MPI spec.  Because reasons.
        # We will require two partitions to have MPI_IDENT communicators to
        # consider them to be equal.
        if (check_null_comm(self._comm) or  # noqa W504
            check_null_comm(other._comm) or  # noqa W504
            check_null_group(self._group) or  # noqa W504
            check_null_group(other._group) or  # noqa W504
            check_null_rank(self.rank) or  # noqa W504
            check_null_rank(other.rank)):  # noqa E129
            return False

        return (check_identical_comm(self._comm, other._comm) and  # noqa W504
                check_identical_group(self._group, other._group) and  # noqa W504
                self.rank == other.rank and  # noqa W504
                self.device == other.device)

    def print_sequential(self, val):

        if self.active:
            print_sequential(self._comm, val)

    def create_partition_inclusive(self, ranks):
        r"""Creates new partition from a subset of workers in this Partition.

        Uses the iterable set of ``ranks``, the lexicographic identifiers of
        the workers *in the current partition* to include in the new
        partition, to create a new partition, ordered as the ranks is ordered.

        This uses ``MPI_Group_incl`` which does not invoke collectives, and
        ``MPI_Comm_create_group`` which is only collective across the workers
        *in the new group*.

        Parameters
        ----------
        ranks : iterable
            The ranks of the workers in this partition.

        Returns
        -------
        A new :any:`MPIPartition` instance.

        """

        ranks = np.asarray(ranks)
        group = self._group.Incl(ranks)

        comm = self._comm.Create_group(group)

        return MPIPartition(comm, group, root=self._root, device=self.device)

    def create_partition_union(self, other, initialize_backend_comm=False):
        r"""Creates new partition from the union of two partitions.

        The new partition is the union of the current and ``other`` partitions.
        The worker's ordering uses the current partition first, in their rank
        order.  Any worker in both partitions is included in the first set and
        is not repeated.

        This uses ``MPI_Group_union`` which does not invoke collectives and
        ``MPI_Comm_create_group`` which is only collective across the workers
        *in the new group*.

        Parameters
        ----------
        other : MPIPartition
            The partition to union with.
        initialize_backend_comm: Bool, optional.
            Flag to initialize the backend communicator.

        Returns
        -------
        A new :any:`MPIPartition` instance.

        """

        # Cannot make a union if the two partitions do not share a root
        if not check_identical_comm(self._root, other._root):
            raise Exception()

        group = MPI.Group.Union(self._group, other._group)

        comm = self._root.Create_group(group)

        P_union = MPIPartition(comm, group, root=self._root, device=self.device)
        if initialize_backend_comm:
            P_union.initialize_backend_comm()

        return P_union

    def create_cartesian_topology_partition(self, shape, **options):
        r"""Creates new partition with Cartesian topology.

        The new partition is a remapping of the current partition to a Cartesian
        topology with the given ``shape``.

        Warning
        -------
        Currently, all workers in the ``self`` Partition must be included in the
        new Cartesian partition.

        Parameters
        ----------
        shape : iterable
            Iterable containing the shape of the new Cartesian partition.
        options : dict, optional
            Options to pass along to ``MPI_Comm_create_cart``.

        Returns
        -------
        A new :any:`MPICartesianPartition` instance.

        """

        shape = np.asarray(shape)
        if self.active:
            comm = self._comm.Create_cart(shape, **options)
            group = comm.Get_group()

            if not check_identical_group(self._group, group):
                raise Exception()

            # group = self._group
            return MPICartesianPartition(comm, group, self._root, shape, device=self.device)

        else:
            comm = MPI.COMM_NULL
            return MPICartesianPartition(comm, self._group, self._root, shape, device=self.device)

    def _build_cross_partition_groups(self, P, P_union,
                                      root_index, src_indices, dest_indices):
        r"""Builds list of ranks and MPI group for communicating across partitions.

        Some partition building functions need to be able to create new
        partitions, with one worker a member of a Cartesian partition and the
        other a general partition.  Here, this is to support the DistDL
        broadcasting rules.

        A worker with the given ``root_index`` will be the root worker in a
        new partition.  That partition is constructed from that worker, plus
        all other workers in the ``P_union`` partition who's color matches the
        root index.

        The ``root_index`` is either the index of the current worker if it is
        in the send group or it is the index that the current index will
        receive from if it is in the receive group.

        Colors are found in ``src_indices`` and ``dest_indices``, which are
        NumPy arrays of the same size as ``P_union``, with locations
        corresponding to workers' ranks in ``P_union``.

        This operation is collective across ``P`` and ``P_union``.

        Parameters
        ----------
        P : MPICartesianPartition
            Partition containing root index.
        P_union : MPIPartition
            Partition that all ranks are a member of.
        root_index : int
            Cartesian index (color) of the root of the new partition.
        src_indices : numpy.ndarray
            All Cartesian indices (colors) in the entire source partition.
        dest_indices : numpy.ndarray
            All Cartesian indices (colors) in the entire destination partition.

        Returns
        -------
        ranks : list
            The list of ranks in the new partition, ordered with the root rank first.
        group : MPI_group
            An MPI group containing ``ranks``.

        """

        root_rank = MPI.PROC_NULL
        if P.active:
            # The ranks in the union that I will send data to (broadcast) or
            # receive data from (reduction), if this is the "send" group.
            # The ranks ranks in the union that will receive data from the
            # same place as me (broadcast) or send data to the same place as
            # me (reduction) if this is the "receive" group.
            dest_ranks = np.where(dest_indices == root_index)[0]
            # My rank in the union (send group for broadcast or receive group
            # for reduction) or the rank in the union I will receive data from
            # (recv group for broadcast) or send data to (send group for
            # reduction).
            root_rank = np.where(src_indices == root_index)[0][0]

        # Create the MPI group
        ranks = []
        group = MPI.GROUP_NULL
        if root_rank != MPI.PROC_NULL:
            # Ensure that the root rank is first, so it will be rank 0 in the
            # new communicator and that ranks are not repeated in the union
            ranks = [root_rank] + [rank for rank in dest_ranks if rank != root_rank]
            ranks = np.array(ranks)
            group = P_union._group.Incl(ranks)

        return ranks, group

    def _create_send_recv_partitions(self, P_union,
                                     send_ranks, group_send,
                                     recv_ranks, group_recv,
                                     initialize_backend_comm=False):
        r"""Creates the send and receive partitions for broadcasts and reductions.

        Uses ``MPI_Comm_create_group`` which is only collective across the
        workers *in the new group*.

        The output ``P_send`` for the current worker may be ``P_recv`` for another
        worker, and vice versa.  A worker may have either, both, or neither.  They
        may also be the same.

        Parameters
        ----------
        P_union : MPIPartition
            Partition that all ranks are a member of.
        send_ranks : iterable
            Set of ranks for new partition current worker sends to.
        group_send : MPI_Group
            MPI_Group containing the ``send_ranks`` workers.
        recv_ranks : iterable
            Set of ranks for new partition current worker receives within.
        group_recv : MPI_Group
            MPI_Group containing the ``recv_ranks`` workers.

        Returns
        -------
        P_send : MPIPartition
            The send partition for the current worker.
        P_recv : MPIPartition
            The receive partition for the current worker.

        """

        # We will only do certain work if certain groups were created.
        has_send_group = not check_null_group(group_send)
        has_recv_group = not check_null_group(group_recv)
        same_send_recv_group = check_identical_group(group_send, group_recv)

        P_send = MPIPartition()
        P_recv = MPIPartition()

        # Brute force the four cases, don't try to be elegant...this pattern
        # is prone to deadlock if we are not careful.
        if has_send_group and has_recv_group and not same_send_recv_group:

            # If we have to both send and receive, it is possible to deadlock
            # if we try to create all send groups first.  Instead, we have to
            # create them starting from whichever has the smallest root rank,
            # first.  This way, we should be able to guarantee that deadlock
            # cannot happen.  It may be linear time, but this is part of the
            # setup phase anyway.
            if recv_ranks[0] < send_ranks[0]:
                comm_recv = P_union._comm.Create_group(group_recv, tag=recv_ranks[0])
                P_recv = MPIPartition(comm_recv, group_recv, root=P_union._root,
                                      device=P_union.device)
                comm_send = P_union._comm.Create_group(group_send, tag=send_ranks[0])
                P_send = MPIPartition(comm_send, group_send, root=P_union._root,
                                      device=P_union.device)
            else:
                comm_send = P_union._comm.Create_group(group_send, tag=send_ranks[0])
                P_send = MPIPartition(comm_send, group_send, root=P_union._root,
                                      device=P_union.device)
                comm_recv = P_union._comm.Create_group(group_recv, tag=recv_ranks[0])
                P_recv = MPIPartition(comm_recv, group_recv, root=P_union._root,
                                      device=P_union.device)
        elif has_send_group and not has_recv_group and not same_send_recv_group:
            comm_send = P_union._comm.Create_group(group_send, tag=send_ranks[0])
            P_send = MPIPartition(comm_send, group_send, root=P_union._root,
                                  device=P_union.device)
        elif not has_send_group and has_recv_group and not same_send_recv_group:
            comm_recv = P_union._comm.Create_group(group_recv, tag=recv_ranks[0])
            P_recv = MPIPartition(comm_recv, group_recv, root=P_union._root,
                                  device=P_union.device)
        else:  # if has_send_group and has_recv_group and same_send_recv_group
            comm_send = P_union._comm.Create_group(group_send, tag=send_ranks[0])
            P_send = MPIPartition(comm_send, group_send, root=P_union._root,
                                  device=P_union.device)
            P_recv = P_send

        if initialize_backend_comm:
            P_send.initialize_backend_comm()
            P_recv.initialize_backend_comm()

        return P_send, P_recv

    def create_broadcast_partition_to(self, P_dest,
                                      transpose_src=False,
                                      transpose_dest=False,
                                      initialize_backend_comm=False):
        r"""Creates the send and receive partitions for broadcasts.

        Creates the send and receive partitions from the current partition to
        the ``P_dest`` partition, following the DistDL broadcast rules
        (:ref:`code_reference/nn/broadcast:Broadcast Rules`).

        The order and layout of the broadcast can be somewhat fluid, so support
        for treating the source and destination partitions is supported.

        Parameters
        ----------
        P_dest : MPICartesianPartition
            Destination partition for the broadcast operation.
        transpose_src : bool, optional
            Flag to transpose the source partition.
        transpose_dest : bool, optional
            Flag to transpose the destination partition.
        initialize_backend_comm: bool, optional
            Flag to initialize the backend communicator.

        Returns
        -------
        Tuple containing the send and receive partitions for this worker.

        """

        P_src = self

        P_send = MPIPartition()
        P_recv = MPIPartition()

        P_union = MPIPartition()
        if P_src.active or P_dest.active:
            P_union = P_src.create_partition_union(P_dest, initialize_backend_comm=False)

        # If we are not active in one of the two partitions, return null
        # partitions
        if not P_union.active:
            return P_send, P_recv

        # Get the rank and shape of the two partitions
        data = None
        if P_src.active:
            data = P_src.shape
        P_src_shape = P_union.broadcast_data(data, P_data=P_src)
        src_dim = len(P_src_shape)

        data = None
        if P_dest.active:
            data = P_dest.shape
        P_dest_shape = P_union.broadcast_data(data, P_data=P_dest)
        dest_dim = len(P_dest_shape)

        # The source must be smaller (or equal) in size to the destination.
        if src_dim > dest_dim:
            raise Exception("No broadcast: Source partition larger than "
                            "destination partition.")

        # Share the src partition dimensions with everyone.  We will compare
        # this with the destination dimensions, so we pad it to the left with
        # ones to make a valid comparison.
        src_shape = np.ones(dest_dim, dtype=int)
        src_shape[-src_dim:] = P_src_shape[::-1] if transpose_src else P_src_shape
        dest_shape = P_dest_shape[::-1] if transpose_dest else P_dest_shape

        # Find any location that the dimensions differ and where the source
        # dimension is not 1 where they differ.  If there are any such
        # dimensions, we cannot perform a valid broadcast.
        no_match_loc = np.where((src_shape != dest_shape) & (src_shape != 1))[0]

        if len(no_match_loc) > 0:
            raise Exception("No broadcast: Dimensions don't match or "
                            "source is not 1 where there is a mismatch.")

        # We will use the matching dimensions to compute the broadcast indices
        match_loc = np.where((src_shape == dest_shape))[0]

        # Compute the Cartesian index of the source rank, in the matching
        # dimensions only.  This index will be constant in the dimensions of
        # the destination that we are broadcasting along.
        src_flat_index = -1
        if P_src.active:
            src_cart_index = np.zeros_like(src_shape)
            c = P_src.index
            if transpose_src:
                src_cart_index[-src_dim:] = c[::-1]
                src_flat_index = cartesian_index_f(src_shape[match_loc],
                                                   src_cart_index[match_loc])
            else:
                src_cart_index[-src_dim:] = c
                src_flat_index = cartesian_index_c(src_shape[match_loc],
                                                   src_cart_index[match_loc])

        data = np.array([src_flat_index], dtype=int)
        src_flat_indices = P_union.allgather_data(data)

        # Compute the Cartesian index of the destination rank, in the matching
        # dimensions only.  This index will match the index in the source we
        # receive the broadcast from.
        dest_flat_index = -1
        if P_dest.active:
            dest_cart_index = P_dest.index
            if transpose_dest:
                dest_cart_index = dest_cart_index[::-1]
                dest_flat_index = cartesian_index_f(dest_shape[match_loc],
                                                    dest_cart_index[match_loc])
            else:
                dest_flat_index = cartesian_index_c(dest_shape[match_loc],
                                                    dest_cart_index[match_loc])
        data = np.array([dest_flat_index], dtype=int)
        dest_flat_indices = P_union.allgather_data(data)

        # Build partitions to communicate single broadcasts across subsets
        # of the union partition.

        # Send ranks are P_union ranks in the send group, the first entry
        # is the root of the group.
        send_ranks, group_send = self._build_cross_partition_groups(P_src,
                                                                    P_union,
                                                                    src_flat_index,
                                                                    src_flat_indices,
                                                                    dest_flat_indices)
        # Recv ranks are P_union ranks in the recv group, the first entry
        # is the root of the group.
        recv_ranks, group_recv = self._build_cross_partition_groups(P_dest,
                                                                    P_union,
                                                                    dest_flat_index,
                                                                    src_flat_indices,
                                                                    dest_flat_indices)

        P_send, P_recv = self._create_send_recv_partitions(P_union,
                                                           send_ranks, group_send,
                                                           recv_ranks, group_recv,
                                                           initialize_backend_comm)

        # Release temporary resources
        P_union.deactivate()

        return P_send, P_recv

    def create_allreduction_partition(self, axes_reduce, initialize_backend_comm=False):
        r"""Creates the partitions for all-reductions.

        Parameters
        ----------
        axes_reduce : tuple
            Partition dimensions along which the all-reduction takes place.
        initialize_backend_comm : bool, optional
            Flag to create the backend communicator.

        Returns
        -------
        The all-reduce partition containing this worker.

        """

        P_src = self

        P_allreduce = MPIPartition()

        # If we are not active return a null partitions
        if not P_src.active:
            return P_allreduce

        for dim in axes_reduce:
            if dim >= P_src.dim:
                raise ValueError(f"Dimension {dim} in axes_reduce exceeds partition dimesion.")

        # `axes_reduce` are eliminated, so in the new partition the dim is 1 in those dimensions
        allreduce_shape = [P_src.shape[k] if k in axes_reduce else 1 for k in range(P_src.dim)]

        # assemble_index_filter produces a mask for preserving dimensions in the 2nd argument,
        # but to get the indices of the ranks in the new partition, we actually need a mask of
        # the inverse of those ranks
        index_filter = assemble_index_filter(P_src.index, axes_reduce, invert=True)
        allreduce_ranks = [P_src._comm.Get_cart_rank(idx) for idx in filtered_range_index(P_src.shape, index_filter)]

        P_allreduce_base = P_src.create_partition_inclusive(allreduce_ranks)
        P_allreduce = P_allreduce_base.create_cartesian_topology_partition(allreduce_shape)

        # Release temporary resources
        P_allreduce_base.deactivate()

        # Create backend communicator
        if initialize_backend_comm:
            P_allreduce.initialize_backend_comm()

        return P_allreduce

    def create_reduction_partition_to(self, P_dest,
                                      transpose_src=False,
                                      transpose_dest=False,
                                      initialize_backend_comm=False):
        r"""Creates the send and receive partitions for reductions.

        Creates the send and receive partitions from the current partition to
        the ``P_dest`` partition, following the DistDL broadcast rules
        (:ref:`code_reference/nn/broadcast:Broadcast Rules`), in reverse.

        The order and layout of the reduction can be somewhat fluid, so support
        for treating the source and destination partitions is supported.

        Parameters
        ----------
        P_dest : MPICartesianPartition
            Destination partition for the reduction operation.
        transpose_src : bool, optional
            Flag to transpose the source partition.
        transpose_dest : bool, optional
            Flag to transpose the destination partition.
        initialize_backend_comm : bool, optional
            Flag to create the backend communicator.

        Returns
        -------
        Tuple containing the send and receive partitions for this worker.

        """

        P_src = self

        P_send = MPIPartition()
        P_recv = MPIPartition()

        P_union = MPIPartition()
        if P_src.active or P_dest.active:
            P_union = P_src.create_partition_union(P_dest, initialize_backend_comm=False)

        # If we are not active in one of the two partitions, return null
        # partitions
        if not P_union.active:
            return P_send, P_recv

        # Get the rank and shape of the two partitions
        data = None
        if P_src.active:
            data = P_src.shape
        P_src_shape = P_union.broadcast_data(data, P_data=P_src)
        src_dim = len(P_src_shape)

        data = None
        if P_dest.active:
            data = P_dest.shape
        P_dest_shape = P_union.broadcast_data(data, P_data=P_dest)
        dest_dim = len(P_dest_shape)

        # The source must be smaller (or equal) in size to the destination.
        if dest_dim > src_dim:
            raise Exception("No reduction: Source partition smaller than "
                            "destination partition.")

        src_shape = P_src_shape[::-1] if transpose_src else P_src_shape
        dest_shape = np.ones(src_dim, dtype=int)
        dest_shape[-dest_dim:] = P_dest_shape[::-1] if transpose_dest else P_dest_shape

        # Find any location that the dimensions differ and where the dest
        # dimension is not 1 where they differ.  If there are any such
        # dimensions, we cannot perform a valid reduction.
        no_match_loc = np.where((src_shape != dest_shape) & (dest_shape != 1))[0]

        if len(no_match_loc) > 0:
            raise Exception("No broadcast: Dimensions don't match or "
                            "source is not 1 where there is a mismatch.")

        # We will use the matching dimensions to compute the broadcast indices
        match_loc = np.where((src_shape == dest_shape))[0]

        # Compute the Cartesian index of the source rank, in the matching
        # dimensions only.  This index will be constant in the dimensions of
        # the destination that we are reducing along.
        src_flat_index = -1
        if P_src.active:
            src_cart_index = P_src.index
            if transpose_src:
                src_cart_index = src_cart_index[::-1]
                src_flat_index = cartesian_index_f(src_shape[match_loc],
                                                   src_cart_index[match_loc])
            else:
                src_flat_index = cartesian_index_c(src_shape[match_loc],
                                                   src_cart_index[match_loc])

        data = np.array([src_flat_index], dtype=int)
        src_flat_indices = P_union.allgather_data(data)

        # Compute the Cartesian index of the destination rank, in the matching
        # dimensions only.  This index will match the index in the source we
        # receive the broadcast from.
        dest_flat_index = -1
        if P_dest.active:
            dest_cart_index = np.zeros_like(dest_shape)
            c = P_dest.index
            if transpose_dest:
                dest_cart_index[:dest_dim] = c
                dest_cart_index = dest_cart_index[::-1]
                dest_flat_index = cartesian_index_f(dest_shape[match_loc],
                                                    dest_cart_index[match_loc])
            else:
                dest_cart_index[-dest_dim:] = c
                dest_flat_index = cartesian_index_c(dest_shape[match_loc],
                                                    dest_cart_index[match_loc])
        data = np.array([dest_flat_index], dtype=int)
        dest_flat_indices = P_union.allgather_data(data)

        # Build partitions to communicate single reductions across subsets
        # of the union partition.

        # Send ranks are P_union ranks in the send group, the first entry
        # is the root of the group.
        send_ranks, group_send = self._build_cross_partition_groups(P_src,
                                                                    P_union,
                                                                    src_flat_index,
                                                                    dest_flat_indices,
                                                                    src_flat_indices)
        # Recv ranks are P_union ranks in the recv group, the first entry
        # is the root of the group.
        recv_ranks, group_recv = self._build_cross_partition_groups(P_dest,
                                                                    P_union,
                                                                    dest_flat_index,
                                                                    dest_flat_indices,
                                                                    src_flat_indices)

        P_send, P_recv = self._create_send_recv_partitions(P_union,
                                                           send_ranks, group_send,
                                                           recv_ranks, group_recv,
                                                           initialize_backend_comm)

        # Release temporary resources
        P_union.deactivate()

        return P_send, P_recv

    def broadcast_data(self, data, root=0, P_data=None):
        r"""Copy arbitrary data from one worker to all workers in a partition.

        Note
        ----
        This is a general broadcast in the sense of traditional parallelism.
        This is not the broadcast in the context of partitioned tensors.

        The data *can* be broadcast within a partition that is a superset of
        one containing the data, as long as ``P_data`` is both a subset of the
        current partition and contains the root worker.  If ``P_data`` is not
        specified, it is assumed to be the current partition.

        Parameters
        ----------
        data :
            The data to be broadcast.
        root :
            The lexicographic identifier of the source worker for the broadcast.
        P_data : MPIPartition
            The partition that contains the data.

        Returns
        -------
        The broadcast data.

        """

        # If the data is coming from a different partition
        if not self.active:
            return None

        if P_data is None:
            P_data = self
            data_root = root
        else:
            # Find the root rank (on P_data) in the self communicator
            rank_map = -1 * np.ones(self.size, dtype=int)
            rank_map_data = np.array([-1], dtype=int)
            if P_data.active:
                rank_map_data[0] = P_data.rank
            self._comm.Allgather(rank_map_data, rank_map)

            if root in rank_map:
                data_root = np.where(rank_map == root)[0][0]
            else:
                raise ValueError("Requested root rank is not in P_data.")

        # Give everyone the dimension of the data array
        data_dim = np.zeros(1, dtype=int)
        if P_data.active and self.rank == data_root:
            # Ensure that data is a numpy array
            data = np.atleast_1d(data)
            data_dim[0] = len(data.shape)
        self._comm.Bcast(data_dim, root=data_root)

        # Give everyone the shape of the data
        data_shape = np.zeros(data_dim[0], dtype=int)
        if P_data.active and self.rank == data_root:
            # Ensure that data is a numpy array
            data = np.atleast_1d(data)
            data_shape[:] = np.asarray(data.shape)
        self._comm.Bcast(data_shape, root=data_root)

        data_dtype = np.zeros(1, dtype=int)
        if P_data.active and self.rank == data_root:
            # Ensure that data is a numpy array
            data_dtype[0] = numpy_to_intID_dtype_dict[data.dtype]
        self._comm.Bcast(data_dtype, root=data_root)
        data_dtype = intID_to_numpy_dtype_dict[data_dtype[0]]

        out_data = np.zeros(data_shape, dtype=data_dtype)
        if P_data.active and P_data.rank == root:
            out_data = data

        self._comm.Bcast(out_data, root=data_root)

        return out_data

    def allgather_data(self, data):
        r"""Gather information from all workers to all workers.

        Note
        ----
        This is a general all-gather in the sense of traditional parallelism.

        Parameters
        ----------
        data :
            The data to be gathered.

        Returns
        -------
        The output data, as a NumPy array, where the location of the entry
        corresponds to the rank of the worker that has that data.

        """

        data = np.atleast_1d(data)
        sz = len(data)

        out_data = -1 * np.ones(sz * self.size, dtype=int)
        self._comm.Allgather(data, out_data)
        out_data.shape = -1, sz

        return out_data

    def allreduce_data(self, data, op="sum"):
        r"""Reduce information from all workers to all workers.

        Note
        ----
        This is a general all-reduce in the sense of traditional parallelism.

        Parameters
        ----------
        data :
            The data to be reduced.
        op : string, optional
            The reduction operation.

        Returns
        -------
        The output data, as a NumPy array, where the contents is reduced using
        the `op` operation.

        """

        from distdl.backends.common import operation_map

        if data.dtype.kind in ['i', 'u']:
            dtype_info = np.iinfo(data.dtype)
        elif data.dtype.kind in ['f']:
            dtype_info = np.finfo(data.dtype)

        if op == "sum":
            output = np.zeros(data.shape, dtype=data.dtype)
        elif op == "prod":
            output = np.ones(data.shape, dtype=data.dtype)
        elif op == "max":
            output = np.full(data.shape, dtype_info.min, dtype=data.dtype)
        elif op == "min":
            output = np.full(data.shape, dtype_info.max, dtype=data.dtype)
        else:
            raise ValueError(f"Invalid operation {op}.")

        self._comm.Allreduce(data, output, operation_map[op])

        return output


class MPICartesianPartition(MPIPartition):
    r"""MPI-based implementation of Cartesian tensor partition.

    This class provides the user interface for the MPI-based implementation of
    tensor partitions with Cartesian topologies.  The MPI interface is
    provided by ``mpi4py``.

    Teams of workers are managed using MPI Groups and communication and data
    movement occurs within the MPI Cartesian Communicator associated with that
    group.

    To handle situations where data movement occurs between two partitions, a
    ``root`` MPI Communicator is stored.  This communicator allows the
    creation of a union of the two partitions, so that collectives can be used
    across that union without impacting other workers in the root
    communicator.  The root communicator is *not* necessarily a Cartesian
    communicator.

    The number of workers in each dimension of the partition is specified by
    the ``shape``.

    Parameters
    ----------
    comm : MPI Cartesian communicator
        MPI Cartesian Communicator describing partition.
    group : MPI group
        MPI Group associated with ``comm``.
    root : MPI communicator, optional
        MPI communicator tracking the original communicator used to create
        all ancestors of this partition.
    shape : iterable
        Number of workers in each dimension of the Cartesian partition.

    Attributes
    ----------
    shape :
        Number of workers in each Cartesian dimension.
    dim :
        Number of dimensions in the partition.
    index :
        Lexicographic identifiers in each Cartesian dimension.
    """

    def __init__(self, comm, group, root, shape, device):

        super(MPICartesianPartition, self).__init__(comm, group, root, device)

        self.shape = np.asarray(shape).astype(int)
        self.dim = len(self.shape)

        self.index = None
        if self.active:
            self.index = self.cartesian_index(self.rank)

    def deactivate(self):
        r"""Deactivates this partition by releasing any resources and
        nullifying any other properties.

        Releases the MPI Communicator and Group that define this partition.

        All `MPIPartition*.create_*` routines act as factories, acquiring
        communicator and group resources for a new partition instance.

        Warning
        -------
        Assumes that the resources released are *able to be released*.  There
        is no way to guarantee safety.  Only call this if you know it is safe
        to release the resources.

        """

        # Preserve the shape
        shape = self.shape

        super(MPICartesianPartition, self).deactivate()

        self.shape = np.asarray(shape).astype(int)
        self.dim = len(self.shape)
        self.index = None

    def create_cartesian_subtopology_partition(self, remain_shape):
        r"""Creates new partition with Cartesian topology in specific
        sub-dimensions.

        The new partition is a subset of the dimensions of the current
        Cartesian partition, following the behavior of ``MPI_Comm_sub``.

        This uses the ``MPI_Comm_sub`` routine to create the subpartition.

        Parameters
        ----------
        remain_shape : iterable
            Iterable containing boolean flags indicating if the dimension is
            to be preserved.

        Returns
        -------
        A new :any:`MPICartesianPartition` instance.

        """

        # remain_shape = np.asarray(remain_shape)
        if self.active:
            comm = self._comm.Sub(remain_shape)
            group = comm.Get_group()

            return MPICartesianPartition(comm, group,
                                         self._root,
                                         self.shape[remain_shape],
                                         self.device)

        else:
            comm = MPI.COMM_NULL
            return MPIPartition(comm, root=self._root, device=self.device)

    def cartesian_index(self, rank):
        r"""Given the rank, returns the Cartesian coordinates of the worker.

        Parameters
        ----------
        rank :
            Lexicographic identifier of the desired worker.

        Returns
        -------
        Cartesian lexicographic identifier of the desired worker.

        """

        if not self.active:
            raise Exception()

        return np.asarray(self._comm.Get_coords(rank))

    def neighbor_ranks(self, rank):
        r"""Given the rank, returns the ranks of the Cartesian neighboring
        workers.

        Parameters
        ----------
        rank :
            Lexicographic identifier of the desired worker.

        Returns
        -------
        neighbor_ranks :
            List of (left, right) pairs of ranks in each dimension.

        """

        if not self.active:
            raise Exception()

        index = self.cartesian_index(rank)

        # Resulting list
        neighbor_ranks = []

        # Loop over the dimensions and add the ranks at the neighboring index to the list
        for i in range(self.dim):
            lindex = [x - 1 if j == i else x for j, x in enumerate(index)]
            rindex = [x + 1 if j == i else x for j, x in enumerate(index)]
            lrank = MPI.PROC_NULL if -1 == lindex[i] else self._comm.Get_cart_rank(lindex)
            rrank = MPI.PROC_NULL if self.shape[i] == rindex[i] else self._comm.Get_cart_rank(rindex)
            neighbor_ranks.append((lrank, rrank))

        return neighbor_ranks
