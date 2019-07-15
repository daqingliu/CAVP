import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # Important argument
    parser.add_argument('--caption_model', type=str, default="cavp")
    parser.add_argument('--start_from', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--self_critical_after', type=int, default=37)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--scheduled_sampling_start', type=int, default=0)
    parser.add_argument('--val_images_use', type=int, default=-1)
    parser.add_argument('--save_checkpoint_every', type=int, default=-1)
    parser.add_argument('--checkpoint_path', type=str, default='save')
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--learning_rate_decay_start', type=int, default=0)
    parser.add_argument('--learning_rate_decay_every', type=int, default=55)
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.1)

    # Data input path
    parser.add_argument('--input_json', type=str, default='data/coco_with_gt.json')
    parser.add_argument('--input_att_dir', type=str, default='data/cocotalk_box_36')
    parser.add_argument('--input_label_h5', type=str, default='data/coco_label_with_gt.h5')

    # Model parameters settings
    parser.add_argument('--rnn_size', type=int, default=1300)
    parser.add_argument('--input_encoding_size', type=int, default=1000)
    parser.add_argument('--att_hid_size', type=int, default=1024)
    parser.add_argument('--fc_feat_size', type=int, default=2048)
    parser.add_argument('--att_feat_size', type=int, default=2048)
    parser.add_argument('--logit_layers', type=int, default=1)
    parser.add_argument('--use_bn', type=int, default=1,
                        help='If 1, then do batch_normalization first in att_embed,'
                             'if 2 then do bn both in the beginning and the end of att_embed')

    # feature manipulation
    parser.add_argument('--norm_att_feat', type=int, default=0)
    parser.add_argument('--use_box', type=int, default=1)
    parser.add_argument('--norm_box_feat', type=int, default=0)

    # Optimization: General
    parser.add_argument('--grad_clip', type=float, default=0.1)
    parser.add_argument('--drop_prob_lm', type=float, default=0.5)
    parser.add_argument('--seq_per_img', type=int, default=5)

    # Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                        help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                        help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                        help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight_decay')

    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5,
                        help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05,
                        help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25,
                        help='Maximum scheduled sampling prob.')

    # Evaluation/Checkpoint
    parser.add_argument('--language_eval', type=int, default=1,
                        help='Evaluate language as well (1 = yes, 0 = no)?'
                             'BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
    parser.add_argument('--losses_print_every', type=int, default=10,
                        help='How often do we print losses (0 = disable)')
    parser.add_argument('--losses_log_every', type=int, default=10,
                        help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--load_best_score', type=int, default=0,
                        help='Do we load previous best score when resuming training.')

    # misc
    parser.add_argument('--id', type=str, default='',
                        help='an id identifying this run/job.'
                             'used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0,
                        help='if true then use 80k, else use 110k')

    # Reward
    parser.add_argument('--cider_reward_weight', type=float, default=1.0,
                        help='The reward weight from cider')
    parser.add_argument('--bleu_reward_weight', type=float, default=0.0,
                        help='The reward weight from bleu4')

    args = parser.parse_args()

    # Check if args are valid
    assert args.rnn_size > 0, "rnn_size should be greater than 0"
    assert args.input_encoding_size > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert 0 <= args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.seq_per_img > 0, "seq_per_img should be greater than 0"
    assert args.beam_size > 0, "beam_size should be greater than 0"
    assert args.losses_log_every > 0, "losses_log_every should be greater than 0"
    assert args.language_eval == 0 or args.language_eval == 1, "language_eval should be 0 or 1"
    assert args.load_best_score == 0 or args.load_best_score == 1, "language_eval should be 0 or 1"
    assert args.train_only == 0 or args.train_only == 1, "language_eval should be 0 or 1"

    return args
