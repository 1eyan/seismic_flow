import argparse


def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument('--h5File', type=str, default='/NAS/czt/mount/chengzhitong/data/测试数据/h5/field1031_irregular.h5')
parser.add_argument('--h5File_regular', type=str, default='/NAS/czt/mount/chengzhitong/data/测试数据/h5/field1031_label.h5')
parser.add_argument('--train_idx_np', type=str, default='/home/chengzhitong/5d_regular/seis_flow_data12V2/generate_py/h5/segc3/segc3_1_info/kept_trace_indices_random_0.5.npy')
'''parser.add_argument('--h5File_2', type=str, default='/home/chengzhitong/5d_regular/seis_flow_data12V2/generate_py/h5/segc3/segc3_2.h5')
parser.add_argument('--h5File_regular_2', type=str, default='/home/chengzhitong/5d_regular/seis_flow_data12V2/generate_py/h5/segc3/segc3_2.h5')
parser.add_argument('--train_idx_np_2', type=str, default='/home/chengzhitong/5d_regular/seis_flow_data12V2/generate_py/h5/segc3/segc3_2_info/kept_trace_indices_random_0.5.npy')
parser.add_argument('--h5File_3', type=str, default='/home/chengzhitong/5d_regular/seis_flow_data12V2/generate_py/h5/segc3/segc3_3.h5')
parser.add_argument('--h5File_regular_3', type=str, default='/home/chengzhitong/5d_regular/seis_flow_data12V2/generate_py/h5/segc3/segc3_3.h5')
parser.add_argument('--train_idx_np_3', type=str, default='/home/chengzhitong/5d_regular/seis_flow_data12V2/generate_py/h5/segc3/segc3_3_info/kept_trace_indices_random_0.5.npy')
parser.add_argument('--h5File_4', type=str, default='/home/chengzhitong/5d_regular/seis_flow_data12V2/generate_py/h5/segc3/segc3_4.h5')
parser.add_argument('--h5File_regular_4', type=str, default='/home/chengzhitong/5d_regular/seis_flow_data12V2/generate_py/h5/segc3/segc3_4.h5')
parser.add_argument('--train_idx_np_4', type=str, default='/home/chengzhitong/5d_regular/seis_flow_data12V2/generate_py/h5/segc3/segc3_4_info/kept_trace_indices_random_0.5.npy')'''
parser.add_argument('--time_ps', type=int, default=1256)
parser.add_argument('--trace_ps', type=int, default=128)
parser.add_argument('--sample_num', type=int, default=1256)
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--expand', type=float, default=0.1)
parser.add_argument('--min_r', type=float, default=0.4)
parser.add_argument('--max_r', type=float, default=0.7)
parser.add_argument('--ovt_mask_mode', type=str, default='train')
parser.add_argument('--ovt_mask_default_mode', type=str, default='random_bin')
parser.add_argument('--ovt_mask_mixture_json', type=str, default=None)
parser.add_argument('--ovt_mask_seed', type=int, default=42)
parser.add_argument('--ovt_mask_min_keep_cells', type=int, default=1)
parser.add_argument('--ovt_mask_fallback_random', type=bool, default=True)
parser.add_argument('--ovt_features', action='store_true',
                    help='Use OVT kd-tree dataset instead of sliding window')
parser.add_argument('--dataset_mode', type=str, default='interp',
                    choices=['interp', 'ovtbin', 'queryctx', 'queryctx_v2'],
                    help='Dataset mode: interp (sliding window), ovtbin (OVT SSL), queryctx (query-context)')
parser.add_argument('--h5File_grid', type=str, default=None,
                    help='Grid H5 for OVT SSL (test_aligned.h5)')
parser.add_argument('--ovt_target_slots', type=int, default=32,
                    help='Number of target slots per patch for OVT SSL')
parser.add_argument('--ovt_kdtree_offset_weight', type=float, default=2.0,
                    help='Offset dimension weight in 4D KNN for OVT SSL')
# ── queryctx 数据集专用参数 ──
parser.add_argument('--dataset_neighbors_train', type=str, default=None,
                    help='train_pool_idx_2d.npz path for queryctx train mode')
parser.add_argument('--dataset_neighbors_test', type=str, default=None,
                    help='infer_query_context.npz path for queryctx infer mode')
parser.add_argument('--train_num_query', type=int, default=16,
                    help='Number of query traces per training patch')
parser.add_argument('--train_context_size', type=int, default=None,
                    help='Fixed context size (None = trace_ps - num_query)')
parser.add_argument('--patch_beta', type=float, default=0.3,
                    help='Diversity weight for diverse_topk (higher = more spread)')
parser.add_argument('--patch_metric_weights', type=str, default='1.0,1.0,0.5,0.5',
                    help='Comma-separated 4D metric weights (sx,sy,rx,ry)')
parser.add_argument('--force_anchor_query', type=str2bool, default=False,
                    help='Force anchor trace to be in query set')
parser.add_argument('--trace_sort_keys_queryctx', type=str, default='rx,ry,sx,sy',
                    help='Comma-separated trace sort keys for queryctx patches')
parser.add_argument('--epoch_repeat', type=int, default=1,
                    help='Repeat anchor pool N times per epoch')
parser.add_argument('--coord_aug_scale', type=float, default=0.0,
                    help='Coordinate augmentation magnitude')
parser.add_argument('--allow_coord_stats_fallback', type=str2bool, default=False,
                    help='Fallback to compute_coord_stats if coord_norm_stats.npz missing')
parser.add_argument('--regular_holdout_npz', type=str, default=None,
                    help='Optional holdout npz for mixed training (infer_query_context format)')
parser.add_argument('--regular_task_prob', type=float, default=0.3,
                    help='Probability of sampling holdout (0.0 = disabled, default 0.3)')
parser.add_argument('--target_mode', type=str, default='self',
                    choices=['self', 'supervised'],
                    help='Training mode: self (self-supervised, random query) or supervised (fixed query/context)')
# 使用 parse_known_args 避免与主训练脚本的参数冲突（主脚本的 --model_name 等会保留给 train_fpmV3_ddp）
args, _ = parser.parse_known_args()