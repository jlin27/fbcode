#! /usr/bin/env python3
import queue
import time
from threading import Thread
from typing import List, Optional, Union

import torch
import torch.distributed.autograd as dist_autograd
from caffe2.torch.fb.distributed.examples.sparsenn.common.sparsenn_training_data import (
    data_preprocessing,
)
from caffe2.torch.fb.distributed.model_parallel.dist_optim import (
    DistributedOptimizer as FunctionalDistributedOptimizer,
    walk_module_for_param_rrefs,
)
from caffe2.torch.fb.distributed.model_parallel.elastic_averaging import (
    ElasticAveragingClient,
    ElasticAveragingParameterSync,
)
from caffe2.torch.fb.distributed.model_parallel.rpc_utils import is_rref_local, sprint
from caffe2.torch.fb.distributed.model_parallel.share_memory import (
    ShareMemoryRPCPickler,
)
from caffe2.torch.fb.distributed.pytorch.adagrad_jit import (
    Adagrad as FunctionalAdagrad,
    RowWiseSparseAdagrad,
)
from caffe2.torch.fb.training_toolkit.backend.data.dpp_session import DppSession
from torch import multiprocessing, nn
from torch.distributed import rpc
from torch.distributed.rpc.api import _use_rpc_pickler
from torch.nn import functional as F

from .iteration_controller import IterationControllerFactory


# only support "file_system". See comments in comm.ShareMemory for detail
multiprocessing.set_sharing_strategy("file_system")
_BATCH_COUNT_PER_PRINT = 100


class Trainer:
    r"""
    Multi threading Hogwild trainer with EASGD and DPP
    """

    def __init__(
        self,
        model: nn.Module,
        ea_client: ElasticAveragingClient,
        use_multithread_hogwild: bool,
        hogwild_workers_names: List[str],
        iteration_controller_factory: IterationControllerFactory,
        loss_fn: Optional[torch.jit.ScriptModule] = None,
    ):
        r"""
        model: nn.Module,
        ea_client: Elastic Averaging Client
        use_multithread_hogwild:
        hogwild_workers_names: Names of Hogwild Workers
        iteration_controller_factory:
        loss_fn: Ignore for now
        """

        self._use_multithread_hogwild = use_multithread_hogwild
        self._model = model

        if use_multithread_hogwild:
            self._queue = queue.Queue()
        else:
            # Important, mutliprocess hogwild requires share_memory
            model.share_memory()

        # TODO: Once we have EASGD settings in TrainingOptions, make this configable
        self._elastic_averaging = ElasticAveragingParameterSync(
            ea_client=ea_client, moving_rate=0.1
        )

        self._iteration_controller_factory = iteration_controller_factory

        self._hogwild_workers_names = hogwild_workers_names
        self._row_count = 0

    @staticmethod
    def _hogwild_worker(
        model: nn.Module,
        iteration_controller_factory: IterationControllerFactory,
        dpp_session: DppSession,
        name: Optional[str] = None,
        # Queue only used in multithread mode to pass the QPS
        queue: Optional[queue.Queue] = None,
    ) -> int:
        sprint(f"Training epoch starts...")
        batch_count = 0
        last_print = time.time()

        param_rrefs = walk_module_for_param_rrefs(model)

        # We assume all dense parameters are local and sparse parameters are remote
        # This is valid as least for now. Since dense parameters are on the trainer
        # and embeddings are on embedding servers.
        local_rrefs = [rref for rref in param_rrefs if is_rref_local(rref)]
        # if the parameters are all local, we can use the local optimizer
        dense_optimizer = FunctionalDistributedOptimizer(
            FunctionalAdagrad, local_rrefs, lr=0.001
        )

        remote_rrefs = [rref for rref in param_rrefs if not is_rref_local(rref)]
        sparse_optimizer = FunctionalDistributedOptimizer(
            RowWiseSparseAdagrad, remote_rrefs, lr=0.001
        )

        iteration_controller = iteration_controller_factory.create(
            # pyre-fixme[6]: Expected `Iterable[_T]` for 1st param but got
            #  `DistributedDataLoader`.
            iter(dpp_session.get_data_loader())
        )

        for batch_raw_data in iteration_controller:
            input = data_preprocessing(batch_raw_data)
            with dist_autograd.context():
                logit, label = model(input)
                # TODO in next diff, use self.loss_fn instead
                loss = F.binary_cross_entropy_with_logits(logit, label)
                # pyre-fixme[16]: Module `autograd` has no attribute `backward`.
                dist_autograd.backward([loss])
                dense_optimizer.step()
                sparse_optimizer.step()

            batch_count += 1

            if batch_count % _BATCH_COUNT_PER_PRINT == 0:
                time_passed = time.time() - last_print
                row_count = _BATCH_COUNT_PER_PRINT * dpp_session.batch_size
                sprint(
                    f"{ (name + ':') if name is not None else ''}"
                    + f"Batch:{batch_count}. "
                    + f"Loss: {loss.item()}. "
                    + f"Row count this print: {row_count}. "
                    + f"Time passed: {time_passed}. "
                    + f"QPS: {row_count / time_passed}. "
                )
                last_print = time.time()
        total_row_count = batch_count * dpp_session.batch_size
        sprint(f"batch count: {batch_count}. row count: {total_row_count}.")

        if queue:
            queue.put(total_row_count)

        return total_row_count

    def _start_multiprocess_hogwild_workers(
        self, dpp_session: DppSession
    ) -> List[rpc.RRef]:
        with _use_rpc_pickler(ShareMemoryRPCPickler()):
            return [
                rpc.remote(
                    worker,
                    self._hogwild_worker,
                    kwargs={
                        "model": self._model,
                        "dpp_session": dpp_session,
                        "iteration_controller_factory": self._iteration_controller_factory,
                    },
                )
                for worker in self._hogwild_workers_names
            ]

    def _start_multithread_hogwild_workers(
        self, dpp_session: DppSession
    ) -> List[Thread]:
        workers_threads = []

        for name in self._hogwild_workers_names:
            thread = Thread(
                target=self._hogwild_worker,
                name=f"{name}",
                kwargs={
                    "name": name,
                    "model": self._model,
                    "dpp_session": dpp_session,
                    "iteration_controller_factory": self._iteration_controller_factory,
                    "queue": self._queue,
                },
            )

            workers_threads.append(thread)
            thread.start()

        return workers_threads

    def _wait_on_workers(self, workers: List[Union[Thread, rpc.RRef]]) -> None:
        print(f"_launch_workers_and_wait {self._use_multithread_hogwild}")
        if self._use_multithread_hogwild:
            for thread in workers:
                # pyre-fixme[16]: `RRef` has no attribute `join`.
                thread.join()
        else:
            # pyre-fixme[16]: `Thread` has no attribute `to_here`.
            row_count_per_trainer = [rref.to_here() for rref in workers]
            sprint(f"row_count_per_trainer :{row_count_per_trainer}")
            self._row_count = sum(row_count_per_trainer)

    def train(self, dpp_session: DppSession) -> int:
        r"""
        dpp_session: A dpp data loader
        """
        begin = time.time()

        workers = None
        if self._use_multithread_hogwild:
            workers = self._start_multithread_hogwild_workers(dpp_session=dpp_session)
        else:
            workers = self._start_multiprocess_hogwild_workers(dpp_session=dpp_session)

        # Use a thread to check if the workers are alive without blocking
        # the main thread.
        worker_status = Thread(target=self._wait_on_workers, args=(workers,))
        worker_status.start()

        # Reuse the main thread to sync parameters as frequently as possible.
        while worker_status.is_alive():
            self._elastic_averaging.sync(self._model)

        time_consumed = time.time() - begin

        if self._use_multithread_hogwild:
            while not self._queue.empty():
                self._row_count += self._queue.get()

        sprint(
            f"Total row count: {self._row_count}."
            + f"Time consumed: {time_consumed}."
            + f"QPS: {self._row_count / time_consumed}"
        )
        return self._row_count
