from src.config_model import load_data
from src.model import ForwardSDE
import torch
import numpy as np
import pandas as pd
import math
import glob
import os
import src.train as train
from src.emd import earth_mover_distance
from src.mio_losses import mioflow_emd2_loss
from types import SimpleNamespace


def _resolve_runtime_config(args, initial_config):
    if hasattr(args, "config_pt") and os.path.exists(args.config_pt):
        return args
    return initial_config(args)


def _move_time_series_to_device(x, device):
    return [x_i.to(device) if x_i.device != device else x_i for x_i in x]

def init_device(args):
    device_type = str(getattr(args, "device_type", "auto")).lower()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if device_type == "cuda":
        args.cuda = args.use_cuda and torch.cuda.is_available()
        device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    elif device_type == "mps":
        args.cuda = False
        device = torch.device("mps" if has_mps else "cpu")
    elif device_type == "cpu":
        args.cuda = False
        device = torch.device("cpu")
    else:
        args.cuda = args.use_cuda and torch.cuda.is_available()
        if args.cuda:
            device = torch.device("cuda:{}".format(args.device))
        elif has_mps:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    return device

def derive_model(args, ckpt_name='epoch_003000'):
    device = init_device(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y, config = load_data(args)
    x = _move_time_series_to_device(x, device)
    model = ForwardSDE(config)

    train_pt = "./" + config.train_pt.format(ckpt_name)
    checkpoint = torch.load(train_pt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    return model, x, y, device


def evaluate_fit(args, initial_config, use_loss='emd'):
    
    device = init_device(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = _resolve_runtime_config(args, initial_config)
    x, y, config = load_data(config)
    x = _move_time_series_to_device(x, device)
    config = SimpleNamespace(**torch.load(config.config_pt, map_location="cpu"))

    file_info = 'interpolate-' + use_loss + '.log'
    log_path = os.path.join(config.out_dir, file_info)
    
    if os.path.exists(log_path):
        print(log_path, 'exists. Skipping.')
        return              

    losses_xy = []
    train_pts = sorted(glob.glob(config.train_pt.format('*')))
    print(config.train_pt)
    print(train_pts)

    for train_pt in train_pts:

        model = ForwardSDE(config)
        checkpoint = torch.load(train_pt, map_location=device)
        print('Loading model from {}'.format(train_pt))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        name = os.path.basename(train_pt).split('.')[1]

        for t in config.train_t:
            loss_xy = _evaluate_impute_model(config, t, model, x, y,device, use_loss).item()
            losses_xy.append((name, 'train', y[t], loss_xy))
        try:
            for t in config.test_t: 
                loss_xy = _evaluate_impute_model(config, t, model, x, y,device, use_loss).item()
                losses_xy.append((name, 'test', y[t], loss_xy))
        except AttributeError:
            continue

    losses_xy = pd.DataFrame(losses_xy, columns = ['epoch', 'eval', 't', 'loss'])
    losses_xy.to_csv(log_path, sep = '\t', index = False)
    print(losses_xy)
    print('Wrote results to', log_path)
    


def evaluate_fit_leaveout(args,initial_config,leaveouts=None,use_loss='emd'):

    device = init_device(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.train_t is None or len(args.train_t) == 0:
        _, _, loaded_args = load_data(args)
        Train_ts = list(loaded_args.train_t)
    else:
        Train_ts = list(args.train_t)

    args.leaveout_t = 'leaveout' + '&'.join(map(str, leaveouts)) 
    args.train_t = list(sorted(set(Train_ts)-set(leaveouts)))   
    args.test_t = leaveouts 
    print('---------------Evaluation-------------------')                                   
    print('--------------------------------------------')
    print('----------leaveout_t=',leaveouts,'---------')
    print('----------train_t=', args.train_t)
    print('--------------------------------------------')

    config = _resolve_runtime_config(args, initial_config)
    x, y, config = load_data(config)
    x = _move_time_series_to_device(x, device)
    config = SimpleNamespace(**torch.load(config.config_pt, map_location="cpu"))

    if os.path.exists(os.path.join(config.out_dir, 'train.log')): 
        print(os.path.join(config.out_dir, 'train.log'), ' exists.')

        file_info = 'interpolate-' + use_loss + '-all.log'
        log_path = os.path.join(config.out_dir, file_info)
        
        if os.path.exists(log_path):
            print(log_path, 'exists. Skipping.')
            return
        model = ForwardSDE(config)

        losses_xy = []
        train_pts = sorted(glob.glob(config.train_pt.format('*')))
        print(config.train_pt)
        print(train_pts)

        for train_pt in train_pts:
            checkpoint = torch.load(train_pt, map_location=device)
            print('Loading model from {}'.format(train_pt))
            model.load_state_dict(checkpoint['model_state_dict'])

            del checkpoint
            
            model.to(device)
            print(model)

            name = os.path.basename(train_pt).split('.')[1]

            for t in config.train_t:
                loss_xy = _evaluate_impute_model(config, t, model, x, y, device, use_loss).item()
                losses_xy.append((name, 'train', y[t], loss_xy))
            try:
                for t in config.test_t: 
                    loss_xy = _evaluate_impute_model(config, t, model, x, y, device, use_loss).item()
                    losses_xy.append((name, 'test', y[t], loss_xy))
            except AttributeError:
                continue

        losses_xy = pd.DataFrame(losses_xy, columns = ['epoch', 'eval', 't', 'loss'])

        out_dir = config.out_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        losses_xy.to_csv(log_path, sep = '\t', index = False)
        print(losses_xy)
        print('Wrote results to', log_path)




        
def _evaluate_impute_model(config, t_cur, model, x, y,device, use_loss='emd'):

    torch.manual_seed(0)
    np.random.seed(0)

    x_0, a_0 = train.p_samp(x[0], config.evaluate_n)
    x_r_0 = train.build_initial_state(x_0, config.use_growth)

    x_r_s = []
    chunk_size = max(1, int(config.ns))
    n_chunks = max(1, math.ceil(int(config.evaluate_n) / chunk_size))
    for i in range(n_chunks):
        x_r_0_ = x_r_0[i * chunk_size:(i + 1) * chunk_size, ]
        if x_r_0_.shape[0] == 0:
            continue
        x_r_s_ = model( [np.float64(y[0])] + [np.float64(y[t_cur])], x_r_0_)
        x_r_s.append(x_r_s_[-1].detach())

    x_r_s = torch.cat(x_r_s)
    y_t = x[t_cur]
    print('y_t', y_t.shape)
    pred_x, _, pred_logw = train.unpack_state(x_r_s, config.x_dim, config.use_growth)
    pred_mass = train.normalized_mass_from_logw(pred_logw) if pred_logw is not None else a_0

    if use_loss == 'mioemd2':
        loss_xy = mioflow_emd2_loss(
            pred_x.contiguous(),
            y_t.contiguous(),
            source_mass=pred_mass,
            detach_weights=config.detach_ot_weights,
        )
    elif use_loss == 'ot':
        loss_xy = mioflow_emd2_loss(
            pred_x.contiguous(),
            y_t.contiguous(),
            source_mass=pred_mass,
            detach_weights=config.detach_ot_weights,
        )
    elif use_loss == 'emd':
        loss_xy = earth_mover_distance(pred_x.cpu().numpy(), y_t.cpu())
    
    return loss_xy 














































        
