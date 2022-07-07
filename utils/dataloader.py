import math
import multiprocessing
import random
import threading
import time
from abc import abstractmethod
from contextlib import closing
from multiprocessing.pool import ThreadPool
from random import shuffle

import keras
import numpy as np
import six
from PIL import Image

try:
    import queue
except ImportError:
    import Queue as queue

from .utils import cvtColor, preprocess_input


class DCganDataset(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, batch_size):
        super(DCganDataset, self).__init__()

        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        
        self.input_shape        = input_shape
        self.batch_size         = batch_size

    def __len__(self):
        return math.ceil(self.length / float(self.batch_size))
    
    def on_epoch_end(self):
        shuffle(self.annotation_lines)

    def __getitem__(self, index):
        images = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i       = i % self.length
            image   = Image.open(self.annotation_lines[i].split()[0])
            image   = cvtColor(image).resize([self.input_shape[1], self.input_shape[0]], Image.BICUBIC)

            image   = np.array(image, dtype=np.float32)
            images.append(preprocess_input(image))
        return np.array(images, dtype=np.float32)


#----------------------------------------------------------#
#   多进程进行数据读取的代码，Copy From Keras==2.1.5
#   Training-related part of the Keras engine.
#----------------------------------------------------------#
_SHARED_SEQUENCES = {}
_SEQUENCE_COUNTER = None


def init_pool(seqs):
    global _SHARED_SEQUENCES
    _SHARED_SEQUENCES = seqs


def get_index(uid, i):
    """Get the value from the Sequence `uid` at index `i`.

    To allow multiple Sequences to be used at the same time, we use `uid` to
    get a specific one. A single Sequence would cause the validation to
    overwrite the training Sequence.

    # Arguments
        uid: int, Sequence identifier
        i: index

    # Returns
        The value at index `i`.
    """
    return _SHARED_SEQUENCES[uid][i]


class SequenceEnqueuer(object):
    """Base class to enqueue inputs.

    The task of an Enqueuer is to use parallelism to speed up preprocessing.
    This is done with processes or threads.

    # Examples

    ```python
        enqueuer = SequenceEnqueuer(...)
        enqueuer.start()
        datas = enqueuer.get()
        for data in datas:
            # Use the inputs; training, evaluating, predicting.
            # ... stop sometime.
        enqueuer.close()
    ```

    The `enqueuer.get()` should be an infinite stream of datas.

    """

    @abstractmethod
    def is_running(self):
        raise NotImplementedError

    @abstractmethod
    def start(self, workers=1, max_queue_size=10):
        """Starts the handler's workers.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`).
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self, timeout=None):
        """Stop running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called start().

        # Arguments
            timeout: maximum time to wait on thread.join()
        """
        raise NotImplementedError

    @abstractmethod
    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Returns
            Generator yielding tuples `(inputs, targets)`
                or `(inputs, targets, sample_weights)`.
        """
        raise NotImplementedError


class OrderedEnqueuer(SequenceEnqueuer):
    """Builds a Enqueuer from a Sequence.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        sequence: A `keras.utils.data_utils.Sequence` object.
        use_multiprocessing: use multiprocessing if True, otherwise threading
        shuffle: whether to shuffle the data at the beginning of each epoch
    """

    def __init__(self, sequence,
                 use_multiprocessing=False,
                 shuffle=False):
        self.sequence = sequence
        self.use_multiprocessing = use_multiprocessing

        global _SEQUENCE_COUNTER
        if _SEQUENCE_COUNTER is None:
            try:
                _SEQUENCE_COUNTER = multiprocessing.Value('i', 0)
            except OSError:
                # In this case the OS does not allow us to use
                # multiprocessing. We resort to an int
                # for enqueuer indexing.
                _SEQUENCE_COUNTER = 0

        if isinstance(_SEQUENCE_COUNTER, int):
            self.uid = _SEQUENCE_COUNTER
            _SEQUENCE_COUNTER += 1
        else:
            # Doing Multiprocessing.Value += x is not process-safe.
            with _SEQUENCE_COUNTER.get_lock():
                self.uid = _SEQUENCE_COUNTER.value
                _SEQUENCE_COUNTER.value += 1

        self.shuffle = shuffle
        self.workers = 0
        self.executor_fn = None
        self.queue = None
        self.run_thread = None
        self.stop_signal = None

    def is_running(self):
        return self.stop_signal is not None and not self.stop_signal.is_set()

    def start(self, workers=1, max_queue_size=10):
        """Start the handler's workers.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, workers could block on `put()`)
        """
        if self.use_multiprocessing:
            self.executor_fn = lambda seqs: multiprocessing.Pool(workers,
                                                                 initializer=init_pool,
                                                                 initargs=(seqs,))
        else:
            # We do not need the init since it's threads.
            self.executor_fn = lambda _: ThreadPool(workers)
        self.workers = workers
        self.queue = queue.Queue(max_queue_size)
        self.stop_signal = threading.Event()
        self.run_thread = threading.Thread(target=self._run)
        self.run_thread.daemon = True
        self.run_thread.start()

    def _wait_queue(self):
        """Wait for the queue to be empty."""
        while True:
            time.sleep(0.1)
            if self.queue.unfinished_tasks == 0 or self.stop_signal.is_set():
                return

    def _run(self):
        """Submits request to the executor and queue the `Future` objects."""
        sequence = list(range(len(self.sequence)))
        self._send_sequence()  # Share the initial sequence
        while True:
            if self.shuffle:
                random.shuffle(sequence)

            with closing(self.executor_fn(_SHARED_SEQUENCES)) as executor:
                for i in sequence:
                    if self.stop_signal.is_set():
                        return
                    self.queue.put(
                        executor.apply_async(get_index, (self.uid, i)), block=True)

                # Done with the current epoch, waiting for the final batches
                self._wait_queue()

                if self.stop_signal.is_set():
                    # We're done
                    return

            # Call the internal on epoch end.
            self.sequence.on_epoch_end()
            self._send_sequence()  # Update the pool

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Yields
            The next element in the queue, i.e. a tuple
            `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
        """
        try:
            while self.is_running():
                inputs = self.queue.get(block=True).get()
                self.queue.task_done()
                if inputs is not None:
                    yield inputs
        except Exception as e:
            self.stop()
            six.raise_from(StopIteration(e), e)

    def _send_sequence(self):
        """Send current Sequence to all workers."""
        # For new processes that may spawn
        _SHARED_SEQUENCES[self.uid] = self.sequence

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        # Arguments
            timeout: maximum time to wait on `thread.join()`
        """
        self.stop_signal.set()
        with self.queue.mutex:
            self.queue.queue.clear()
            self.queue.unfinished_tasks = 0
            self.queue.not_full.notify()
        self.run_thread.join(timeout)
        _SHARED_SEQUENCES[self.uid] = None
