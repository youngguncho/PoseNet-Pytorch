import os
import argparse
from data_loader import get_loader
from solver import Solver
import copy
from torch.backends import cudnn


def main(config):
    cudnn.benchmark = True
    data_loader = get_loader(config.model, config.image_path, config.metadata_path, config.mode, config.batch_size,
                             config.shuffle)

    solver = Solver(data_loader, config)
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--sequential_mode', type=str, default=None,
                        choices=[None, 'model', 'fixed_weight', 'batch_size', 'learning_rate', 'beta'])

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta', type=float, default=500)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--fixed_weight', type=bool, default=False)
    parser.add_argument('--model', type=str, default='Resnet', choices=['Googlenet', 'Resnet'])
    parser.add_argument('--pretrained_model', type=str, default=None)

    parser.add_argument('--image_path', type=str, default='/mnt/data2/image_based_localization/posenet/KingsCollege')
    parser.add_argument('--metadata_path', type=str,
                        default='/mnt/data2/image_based_localization/posenet/KingsCollege/dataset_train.txt')

    # Training settings
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0 1 2 3') # selection of gpu id (single gpu)
    # parser.add_argument('--dataset', type=str, default='Oxford', choices=['NCLT', 'VKITTI', 'Oxford', 'QUT'])
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8) # 16

    # Test settings
    parser.add_argument('--test_model', type=str, default='29_3000')

    # Misc
    parser.add_argument('--use_tensorboard', type=bool, default=True)

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--model_save_step', type=int, default=10)

    config_default = parser.parse_args()

    # -------------------------
    # -------------------------
    # Evaluate fixed weight
    # config = copy.deepcopy(config_default)
    #
    # config.sequential_mode = 'fixed_weight'
    # list_roi = [False, True]
    #
    # for val in list_roi:
    #     config.fixed_weight = val
    #     config.mode = 'train'
    #     main(config)
    #     config.mode = 'test'
    #     main(config)

    # Evaluate learning rate
    config = copy.deepcopy(config_default)

    config.sequential_mode = 'learning_rate'
    list_roi = [0.0001, 0.0005, 0.001]

    for val in list_roi:
        config.lr = val
        config.mode = 'train'
        main(config)
        config.metadata_path = '/mnt/data2/image_based_localization/posenet/KingsCollege/dataset_test .txt'
        config.mode = 'test'
        main(config)

    # Evaluate beta
    config = copy.deepcopy(config_default)

    config.sequential_mode = 'beta'
    list_roi = [500, 1000, 1500]

    for val in list_roi:
        config.beta = val
        config.mode = 'train'
        main(config)
        config.mode = 'test'
        config.metadata_path = '/mnt/data2/image_based_localization/posenet/KingsCollege/dataset_test .txt'
        main(config)
