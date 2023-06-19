import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import json
import argparse
import torch
from src.experiment import common_functions as cmf
from src.model.MCGVL import MCGVL
from src.utils import timer
import numpy as np
import random
from src.model.LOSS import LOSS
import torch.optim as optim



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

setup_seed(102)


def _get_argument_params():
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--config_path",
        default="./options/charades/config.yml", help="Path to config file.")
	parser.add_argument("--method_type",
        default="tgn_lgi", help="Method type among [||].")

	parser.add_argument("--dataset",
        default="charades", help="Dataset to train models [|].")
	parser.add_argument("--num_workers", type=int,
        default=0, help="The number of workers for data loader.")
	parser.add_argument("--debug_mode" , action="store_true", default=False,
		help="Train the model in debug mode.")

	params = vars(parser.parse_args())
	print(json.dumps(params, indent=4))
	return params

def train(config):

    dsets, L = cmf.get_loader(dataset, split=["train", "test"],
                              loader_configs=[config["train_loader"], config["test_loader"]],
                              num_workers=config["misc"]["num_workers"])
    config['model']['query_enc_emb_idim'] = len(list(dsets['train'].wtoi.keys()))
    num_step = config["optimize"].get("num_step", 30)
    net = MCGVL(config).cuda()
    loss_fn = LOSS()
    optimizer = optim.Adam(net.parameters(), lr=0.0004)
    for epoch in range(num_step):
        total_loss = 0.0
        idx = 1
        net.train()
        for batch in L['train']:
            optimizer.zero_grad()
            net_inps, gts = net.prepare_batch(batch)
            outputs = net(net_inps)
            loss = loss_fn(outputs, gts, count_loss=True)
            loss.backward()
            optimizer.step()
            if idx % 20 == 0:
                print("epoch:{}, batch: {}, loss: {:.3}".format(epoch, idx, loss))
            idx += 1
            total_loss += loss
        print("epoch:{}, total_loss: {:.3}".format(epoch, total_loss/len(L['train'])))


        net.eval()
        net.reset_status()
        for batch in L['test']:
            net_inps, gts = net.prepare_batch(batch)
            outputs = net(net_inps)
            net.compute_status(outputs, gts)
            net.performances()
        net.print_performances(epoch)
        net.reset_counters()



if __name__ == "__main__":
    params = _get_argument_params()
    global M, dataset
    M, dataset, config = cmf.prepare_experiment(params)

    train(config)
