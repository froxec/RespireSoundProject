from LungAttn import LungAttnBinary, accuracy_binary, one_hot, myDataset, makedirs
import torch.nn.utils.prune as prune
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import numpy as np
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_bs', type=int, default=64)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--input', '-i',
                    default="./pack/binary/tqwt1_4_train.p", type=str,
                    help='path to directory with input data archives')
parser.add_argument('--test', default="./pack/binary/tqwt1_4_test.p",
                    type=str, help='path to directory with test data archives')
parser.add_argument('--prunning_amount', default=0.4,
                    type=int)
args = parser.parse_args()


def get_mnist_loaders(batch_size=128, test_batch_size = 500, workers = 4, perc=1.0, binary=False):
    # ori, ck, wh, res, label
    ori, stftl, stfth, stftr, labels = joblib.load(open(args.input, mode='rb'))
    stftl, stfth, stftr = np.array(stftl), np.array(stfth), np.array(ori)
    if binary:
        labels = np.array(labels).reshape(-1, 1)
    else:
        labels = one_hot(np.array(labels), 4)
    stft = np.concatenate((stftl[:, np.newaxis], stfth[:, np.newaxis], stftr[:, np.newaxis]), 1)

    ori_tst, stftl_test, stfth_test, stftr_test, labels_test = joblib.load(open(args.test, mode='rb'))
    stftl_test, stfth_test, stftr_test = np.array(stftl_test), np.array( stfth_test), np.array(ori_tst)
    if binary:
        labels_test = np.array(labels_test).reshape(-1, 1)
    else:
        labels_test = one_hot(np.array(labels_test), 4)
    stft_test = np.concatenate((stftl_test[:, np.newaxis], stfth_test[:, np.newaxis], stftr_test[:, np.newaxis]), 1)

    train_loader = DataLoader(
        myDataset(stft, labels), batch_size=batch_size,
        shuffle=True, num_workers=workers, drop_last=True
    )
    train_eval_loader = DataLoader(
        myDataset(stft, labels), batch_size=test_batch_size,
        shuffle=False, num_workers=workers, drop_last=True
    )

    test_loader = DataLoader(
        myDataset(stft_test, labels_test),
        batch_size=test_batch_size, shuffle=False, num_workers=workers, drop_last=False
    )

    return train_loader, train_eval_loader, test_loader


if __name__ == "__main__":
    model_path = "./log/pruned/model_pruned_0.4/saved_model_params"
    save_dir = "./log/quantized/model_quantized"
    makedirs(save_dir)

    # get loaders
    train_loader, train_eval_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size,
                                                                     test_batch_size=args.test_bs, workers=args.workers,
                                                                     binary=True)
    # prepare criterion
    criterion = nn.BCELoss()

    # select device
    use_cuda = False
    device = torch.device("cuda:0" if use_cuda else "cpu")

    fp32_model = LungAttnBinary()
    fp32_model.load_state_dict(torch.load(model_path))
    dummy_input, label = next(iter(train_loader))
    # export onxx model
    torch.onnx.export(fp32_model, dummy_input, "lungattn.onnx", verbose=True)