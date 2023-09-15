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
parser.add_argument('--prunning_amount', default=0.0,
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
    model_path = "./log/details/rigl_alpha_03_sparsity_08_delta_t_5_t_end_50/saved_model_params"
    save_dir = "./log/pruned/model_rigled" + str(args.prunning_amount)
    makedirs(save_dir)
    model = LungAttnBinary()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.identity(module, 'weight')
        if isinstance(module, torch.nn.Linear):
            prune.identity(module, 'weight')
    model.load_state_dict(torch.load(model_path))
    print(model)
    # select device
    use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.to(device)
    model.eval()
    # get loaders
    train_loader, train_eval_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size, test_batch_size=args.test_bs, workers=args.workers,
                                                                     binary=True)

    criterion = nn.BCELoss()

    prunned_connections = 0
    connections_available_for_pruning = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weights_num = torch.numel(module.weight_mask)
            connections_available_for_pruning += weights_num
            prunned_connections += (weights_num - torch.count_nonzero(module.weight_mask))
    amount_prunned = 100 * (prunned_connections / connections_available_for_pruning)
    print("Pruned {} out of {} connections. {} % connections were prunned".format(prunned_connections,
                                                                                  connections_available_for_pruning,
                                                                                  amount_prunned))


    # evaluate model
    print("Scores before channel pruning")
    train_acc, train_Se, train_Sq, train_Score, _, train_confm = accuracy_binary(model, train_eval_loader,
                                                                                 criterion, device)
    test_acc, test_Se, test_Sq, test_Score, test_loss, test_confm = accuracy_binary(model, test_loader, criterion, device)


    print("Train Acc {:.4f} |  train Se {:.4f} | train Sq {:.4f} | train Score {:.4f} | Test Loss {:.4f} |Test Acc {:.4f} | test Se {:.4f} | test Sq {:.4f} | test Score {:.4f}".format(
        train_acc, train_Se, train_Sq, train_Score, test_loss, test_acc, test_Se,
            test_Sq, test_Score))
    print("Scores after channel prunning")

    model.train()
    prunned_connections = 0
    connections_available_for_pruning = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=args.prunning_amount, n=2, dim=0)
            weights_num = torch.numel(module.weight_mask)
            connections_available_for_pruning += weights_num
            prunned_connections += (weights_num - torch.count_nonzero(module.weight_mask))
    amount_prunned = 100 * (prunned_connections / connections_available_for_pruning)
    print("Pruned {} out of {} connections. {} % connections were prunned".format(prunned_connections,
                                                                                  connections_available_for_pruning,
                                                                                  amount_prunned))

    model.eval()
    train_acc, train_Se, train_Sq, train_Score, _, train_confm = accuracy_binary(model, train_eval_loader,
                                                                                 criterion, device)
    test_acc, test_Se, test_Sq, test_Score, test_loss, test_confm = accuracy_binary(model, test_loader, criterion,
                                                                                    device)

    print( "Train Acc {:.4f} |  train Se {:.4f} | train Sq {:.4f} | train Score {:.4f} | Test Loss {:.4f} |Test Acc {:.4f} | test Se {:.4f} | test Sq {:.4f} | test Score {:.4f}".format(
            train_acc, train_Se, train_Sq, train_Score, test_loss, test_acc, test_Se,
            test_Sq, test_Score))

    print("REMOVING REPARAMETRIZATION AND SAVING MODEL")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')
    torch.save(model.state_dict(), save_dir + '/saved_model_params')

    # print("REDUCING MODEL STRUCTURE AND SAVING MODEL")
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
    #         pass
    # torch.save(model.state_dict(), save_dir + "_reduced" + '/saved_model_params')

    # measure inference time

