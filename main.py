import argparse
from transformers import set_seed

from utils import *
from myframework import MyFramework
import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='./model_saved')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.",
    )
parser.add_argument('--dataset_idx', default=0, type=int)
parser.add_argument('--config_idx', default=0, type=int)
parser.add_argument('--max_len', default=100, type=int)
parser.add_argument('--training_epoch', default=30, type=int)
parser.add_argument('--patience', default=5, type=int)
parser.add_argument('--early_stop', action='store_true')
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--gpuid', default='0', type=str)
parser.add_argument('--id', default=0, type=int)
parser.add_argument('--bert_path', default='./bert/bert-base-uncased')
parser.add_argument('--freeze_bert', action='store_true')
parser.add_argument('--freeze_layer_num', type=int, default=6)
parser.add_argument('--max_class_words', type=int, default=10)
parser.add_argument('--train_task_num', default=800, type=int)
parser.add_argument('--test_task_num', default=600, type=int)
parser.add_argument('--eval_task_num', default=600, type=int)


args = parser.parse_args()

if __name__ == '__main__':
    dataset_list = ['FewAsp', 'FewAsp(single)', 'FewAsp(multi)']
    config_list = [[2, 5, 5], [1, 5, 10], [1, 10, 5], [1, 10, 10]]
    dataset = dataset_list[args.dataset_idx]
    B, N, K = config_list[args.config_idx]
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    device = 'cuda' if torch.cuda.is_available() else "cpu"

    model_name = dataset + '_seed_{}_id_{}'.format(args.seed, args.id)
    root_path = dataset + '_N_{}_K_{}'.format(N, K)
    save_path = os.path.join(args.save_path, root_path)
    root_dir_path = build_dir(model_name, save_path)

    Q = 5
    log_file = os.path.join(root_dir_path, 'test.txt')
    with open(log_file, 'w') as w:
        w.write(' ')
    framework = MyFramework(args, B=B, N=N, K=K, Q=Q, model_name=root_dir_path, dataset=dataset,
                            max_len=args.max_len, training_epoch=args.training_epoch, early_stop=args.early_stop, patience=args.patience,
                            shuffle=args.shuffle, device=device, log_file=log_file)
    framework.train()
    framework.test(tasks=args.test_task_num)


