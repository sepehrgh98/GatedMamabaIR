import logging
import time
import torch
import os
from os import path as osp
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs, results_tracker
from basicsr.utils.options import dict2str, parse_options


from basicsr.tools.scan_hook import SelectiveScanObserver
import numpy as np


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)

    # observer = SelectiveScanObserver()

    # net = model.net_g.module  # unwrap DataParallel

    # scan_module = (
    #     net.layers[2]
    #     .residual_group.layers[3]
    #     .assm.selectiveScan
    # )

    # scan_module.register_forward_pre_hook(observer.hook_in)
    # scan_module.register_forward_hook(observer.hook_out)

    git = results_tracker.git_info(root_path)
    results_csv = opt['val'].get('results_csv', osp.join(root_path, 'docs', 'results.csv'))
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        start = time.time()
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])
        runtime_s = time.time() - start

        if getattr(model, 'metric_results', None):
            row = results_tracker.make_row(opt, test_set_name, model.metric_results, len(test_loader.dataset),
                                           runtime_s, opt.get('opt_path', ''), git, tag=opt['val'].get('tag', ''))
            results_tracker.append_rows(results_csv, [row])
            logger.info(f'Appended {test_set_name} results to {results_csv}')

    # results = []

    # for i in range(len(observer.inputs)):
    #     Xin = observer.inputs[i][0].numpy()   # [L_i, C]
    #     Xout = observer.outputs[i][0].numpy() # [L_i, C]

    #     # --- compute metrics per image ---
    #     mu_in = Xin.mean(axis=0)
    #     var_in = Xin.var(axis=0)

    #     mu_out = Xout.mean(axis=0)
    #     var_out = Xout.var(axis=0)

    #     # correlation matrices
    #     def corr(X, eps=1e-6):
    #         X = X - X.mean(axis=0, keepdims=True)
    #         X = X / (X.std(axis=0, keepdims=True) + eps)
    #         return (X.T @ X) / (X.shape[0] - 1)

    #     Rin = corr(Xin)
    #     Rout = corr(Xout)

    #     corr_change = np.linalg.norm(Rin - Rout, ord="fro")

    #     results.append({
    #         "L": Xin.shape[0],
    #         "corr_change_frob": float(corr_change),
    #         "mu_diff": float(np.linalg.norm(mu_in - mu_out)),
    #         "var_diff": float(np.linalg.norm(var_in - var_out)),
    #     })

    # np.save("urban100_selectivescan_metrics.npy", results)
    # print("Saved per-image SelectiveScan metrics.")


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
