#!/usr/bin/env python
# Copyright Michael Fonder 2021. All rights reserved.
# ==============================================================================

class M4DepthOptions:
    def __init__(self, args):

        # Global Options
        args.add_argument('--dataset',
                          default="",
                          help="""Dataset settings to use""")
        args.add_argument('--db_seq_len',
                          default=None, type=int,
                          help="""Dataset sequence length (frames)""")
        args.add_argument('--seq_len',
                          default=2, type=int,
                          help="""Sequence length (frames)""")
        args.add_argument('--arch_depth',
                          default=6, type=int,
                          help="""Depth of the architecture (number of levels)""")
        args.add_argument('--special_case',
                          default=0, type=int,
                          help="""Special network case; 0: No, 1: only using current data, 2: only using history up to t-1 """)
        args.add_argument('--data_aug',
                          default=False, action="store_true",
                          help="Perform data augmentation")
        args.add_argument('--cpu_matmul',
                          default=False, action="store_true",
                          help="Perform matmul ops on CPU (required for cuda 10.x)")

        # Eval Options
        args.add_argument("--mask",
                          default=False, action="store_true",
                          help="mask infinity during evaluation")
        args.add_argument('--eval_only_last_pic',
                          default=False, action="store_true",
                          help="Compute performance metrics only for the last picture of sequences")

        self.cmdline = args