#!/usr/bin/env python
# Copyright Michael Fonder 2021. All rights reserved.
# ==============================================================================

class PipelineOptions:
    def __init__(self, args):
        self.cmdline = args

        # Global options
        self.cmdline.add_argument('--log_dir',
                                  default="",
                                  help="""Directory of summaries and checkpoints""")
        self.cmdline.add_argument('-b', '--batch_size',
                                  default=64, type=int,
                                  help="""Size of each minibatch for each GPU""")
        self.cmdline.add_argument('-g', '--num_gpus',
                                  default=1, type=int,
                                  help="""Number of GPUs to run on.""")
        self.cmdline.add_argument('--no_batch_scaling', default=False,
                                  action="store_true",
                                  help="""Minibatch size given is the total amount of batches run on all GPUs""")
        self.cmdline.add_argument('--no_nccl', default=False,
                                  action="store_true",
                                  help="""Disable NCCL (not recommended)""")

        # Train options
        self.cmdline.add_argument('--train_datadir',
                                  default=None,
                                  help="""Path to the serialized training dataset.""")
        self.cmdline.add_argument('--val_datadir',
                                  default=None,
                                  help="""Path to the serialized validation dataset.""")
        self.cmdline.add_argument('--num_batches',
                                  default=50, type=int,
                                  help="""Number of batches to run.""")
        self.cmdline.add_argument('--num_epochs',
                                  default=None, type=int,
                                  help="""Number of epochs to run (overrides --num_batches).""")
        self.cmdline.add_argument('--summary_interval_secs',
                                  default=60, type=int,
                                  help="""How often (in seconds) to update summaries.""")
        self.cmdline.add_argument('--validation_interval_secs',
                                  default=180, type=int,
                                  help="""How often (in seconds) to run validation.""")
        self.cmdline.add_argument('--save_interval_secs', default=300, type=int,
                                  help="""How often (in seconds) to save checkpoints.""")
        self.cmdline.add_argument('--display_every',
                                  default=50, type=int,
                                  help="""How often (in iterations) to print-out training information.""")

        # Test options
        self.cmdline.add_argument('--test_datadir',
                                  default=None,
                                  help="""Path to the serialized test dataset.""")
        self.cmdline.add_argument("--export_results",
                                  default=False, action="store_true",
                                  help="""Export results of the test using the export_result function of the model""")