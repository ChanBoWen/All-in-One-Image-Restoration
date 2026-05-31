import argparse

parser = argparse.ArgumentParser()

# Training parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=130, help='maximum number of epochs to train the total model.')
parser.add_argument('--epochs_encoder', type=int, default=15, help='number of epochs to train encoder.')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of encoder.')
parser.add_argument('--batch_size', type=int, default=8, help="Batch size to use per GPU")

parser.add_argument('--patch_size', type=int, default=128, help='patcphsize of input.')
parser.add_argument('--encoder_dim', type=int, default=256, help='the dimensionality of encoder.')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers.')

# Degradations
parser.add_argument('--de_type', type=list, default=['lowlight', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')

# Data paths
parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='directory holding per-task filename manifests.')
parser.add_argument('--lowlight_dir', type=str, default='data/Train/LowLight/',
                    help='root of low-light training data (expects Low/ and Normal/ subtrees)')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')

parser.add_argument('--output_path', type=str, default="output/", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="ckpt/AllInOne/", help='checkpoint save path')

options = parser.parse_args()