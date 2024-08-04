import os
import logging
from datetime import datetime
import numpy as np

def setup_logger(
    logger_name, root, phase, level=logging.INFO, screen=False, tofile=False
):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )
    lg.setLevel(level)
    if tofile:
        # log_file = os.path.join(root, phase + "_{}.log".format(get_timestamp()))
        log_file = os.path.join(root, phase + ".log")
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)

def progress_bar(batch_idx, num_batches, report_interval, train_loss):
    """Neat progress bar to track training."""

    dec = int(np.ceil(np.log10(num_batches)))
    bar_size = 31 + dec
    progress = (batch_idx % report_interval) / report_interval
    fill = int(progress * bar_size) + 1
    print('\rBatch {:>{dec}d} [{}{}] Train loss: {:>1.5f}'.format(batch_idx + 1, 
                                                                  '=' * fill + '>', ' ' * (bar_size - fill), 
                                                                  train_loss, dec=str(dec)), end='', flush=True)
def show_on_report(batch_idx, num_batches, loss, elapsed):
    """Formats training stats."""

    print('\r{}'.format(' ' * 80), end='\r')
    dec = int(np.ceil(np.log10(num_batches)))
    print('\rBatch {:>{dec}d} / {:d} | Avg loss: {:>1.5f} | Avg train time / batch: {:d} ms'.format(batch_idx + 1, num_batches, 
                                                                                                    loss, int(elapsed), dec=dec), end='', flush=True)

def time_elapsed_since(start):
    """Computes elapsed time since start."""

    timedelta = datetime.now() - start
    string = str(timedelta)[:-7]
    ms = int(timedelta.total_seconds() * 1000)

    return string, ms