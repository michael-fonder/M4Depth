import json
import os.path

from dataloaders import DataloaderParameters
from m4depth_network import M4depthAblationParameters


class M4DepthOptions:
    def __init__(self, args):
        # Global Options
        args.add_argument('--dataset',
                          default="",
                          choices=['midair', 'tartanair', 'kitti-raw'],
                          help="""Dataset to use (midair, tartanair or kitti-raw)""")
        args.add_argument('--ckpt_dir',
                          default="ckpt",
                          help="""Model checkpoint directory""")
        args.add_argument('--mode',
                          choices=['train', 'finetune', 'eval', 'validation', 'predict'],
                          help="""Model run mode (train, finetune or eval)""")
        args.add_argument('--disable_xla',
                          default=False, action="store_true",
                          help="Disable XLA optimization and compilation")

        # Dataloader Options
        args.add_argument('--db_path_config',
                          default=os.path.join(os.path.dirname(__file__),"datasets_location.json"),
                          help="""Json file with datasets path configuration""")
        args.add_argument('--batch_size',
                          default=3, type=int,
                          help="""Size of each minibatch for each GPU""")
        args.add_argument('--records_path',
                          default=None, type=str,
                          help="""csv files to use when loading dataset""")
        args.add_argument('--db_seq_len',
                          default=None, type=int,
                          help="""Dataset sequence length (frames) [Mandatory for training!]""")
        args.add_argument('--seq_len',
                          default=4, type=int,
                          help="""Sequence length (frames)""")

        # Train Options
        args.add_argument('--log_dir',
                          default=None,
                          help="""Tensorboard log directory""")
        args.add_argument('--summary_interval',
                          default=1200, type=int,
                          help="""How often (in batches) to update summaries.""")
        args.add_argument('--save_interval', default=2, type=int,
                          help="""How often (in epochs) to save checkpoints.""")
        args.add_argument('--conf_err_rate', default=0.05, type=float,
                          help="""Desired error rate on the confidence bounds.""")
        args.add_argument('--no_augmentation',
                          default=False, action="store_true",
                          help="Disable data augmentation")
        args.add_argument('--enable_validation',
                          default=False, action="store_true",
                          help="Perform validation after each training epoch")
        args.add_argument('--keep_top_n',
                          default=1, type=int,
                          help="""Amount of top performing checkpoints to keep.""")

        # Ablation Options
        args.add_argument('--arch_depth',
                          default=6, type=int,
                          help="""Depth of the architecture (number of levels)""")
        args.add_argument("--no_DINL",
                          default=False, action="store_true",
                          help="Remove Domain Invariant Normalization Layer")
        args.add_argument("--no_SNCV",
                          default=False, action="store_true",
                          help="Remove Spatial Neigborhood Cost Volumes from the decoder")
        args.add_argument("--no_time_recurr",
                          default=False, action="store_true",
                          help="Remove time recurrence from the decoder")
        args.add_argument("--no_feature_normalization",
                          default=False, action="store_true",
                          help="Don't normalize feature vectors before building a cost volume")
        args.add_argument("--no_feature_subdivision",
                          default=False, action="store_true",
                          help="Don't subdivide feature vectors and build multiple cost volumes")
        args.add_argument("--no_level_memory",
                          default=False, action="store_true",
                          help="Remove additional level memory")

        args = args

        cmd, test_args = args.parse_known_args()
        json_data = json.load(open(cmd.db_path_config))

        path_root = os.path.dirname(__file__)
        for dataset, path in json_data.items():
            if not os.path.isabs(path):
                abs_path = os.path.join(path_root, path)
                json_data[dataset] = os.path.normpath(abs_path)

        self.ablation_settings = M4depthAblationParameters(not cmd.no_DINL, not cmd.no_SNCV, not cmd.no_time_recurr,
                                                           not cmd.no_feature_normalization, not cmd.no_feature_subdivision,
                                                           not cmd.no_level_memory)

        self.dataloader_settings =  DataloaderParameters(json_data, cmd.records_path, cmd.db_seq_len,
                                                         cmd.seq_len, not cmd.no_augmentation )
