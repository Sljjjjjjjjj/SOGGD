import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Hybrid Keypoint Detection Training')

    # Data path parameters
    parser.add_argument('--data_dir', type=str, default='./data/train',
                        help='Training dataset path')
    parser.add_argument('--val_dir', type=str, default='./data/val',
                        help='Validation dataset path')
    parser.add_argument('--test_dir', type=str, default='./data/test',
                        help='Test dataset path')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Model saving path')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume training from checkpoint')

    # Dataset related parameters
    parser.add_argument('--dataset_type', type=str, default='custom',
                        choices=['custom', 'imagenet'],
                        help='Dataset type (custom, imagenet)')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--topk', type=int, default=100,
                        help='Number of keypoints to detect per image')
    parser.add_argument('--heatmap_sigma', type=float, default=4.0,
                        help='Standard deviation of Gaussian kernel for heatmap')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr_keynet', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use mixed precision training')

    # Model parameters
    parser.add_argument('--pyramid_scales', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Feature pyramid scales')

    # Parse arguments
    args = parser.parse_args()
    return args 