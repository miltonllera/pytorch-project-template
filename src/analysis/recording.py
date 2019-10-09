import os
import numpy as np
import pandas as pd
import torch
from itertools import product

from ignite.engine import Engine, Events, _prepare_batch


def create_activity_recorder(model, device=None, non_blocking=False,
                        prepare_batch=_prepare_batch):
    if device:
        model.to(device)

    def _record(engine, batch):            
        model.eval()
        with torch.no_grad():
            inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)
            pred, hidden = model(inputs)
            
            try:
                hx, cx = hidden
                hx = hx.squeeze()
                cx = cx.squeeze()
            
            except:
                hx, cx = hidden[-1]
            
            engine.state.hidden.append(hx.cpu().numpy())
            engine.state.memcell.append(cx.cpu().numpy())

            return pred, targets

    engine = Engine(_record)
    
    def init_recorder(engine):
        engine.state.hidden = []
        engine.state.memcell = []
    
    engine.add_event_handler(Events.EPOCH_STARTED, init_recorder)

    return engine


def create_grad_recorder(model, device=None, non_blocking=False):
    if device:
        model.to(device)
        
    def prepare_batch(batch, device, non_blocking):
        inputs, targets = _prepare_batch(batch, device, non_blocking)
        inputs.requires_grad_(True)
        return inputs, targets

    def _record(engine, batch):
        model.train()
        
        inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)
        pred, _ = model(inputs)
        pred.sum().backward()
        
        engine.state.grads.append(inputs.grad.cpu().numpy()[:, :, 0])

        return pred, targets

    engine = Engine(_record)

    engine.add_event_handler(Events.EPOCH_STARTED, lambda e: setattr(e.state, 'grads', []))
    
    return engine
    

def grad_wrt_inputs(model, data):
    grad_recorder = create_grad_recorder(model, device)
    grad_recorder.run(data)
    return np.concatenate(grad_recorder.state.grads, axis=0)


def get_masks(data):
    masks = []
    for inputs, _ in data:
        masks.append(inputs.cpu().numpy()[:, :, 1])
        
    return np.concatenate(masks, axis=0)

def get_model_sensibility(models, data, params, level_names):
    sensibilities = []
    
    for model in models:
        masks = get_masks(data)
        grads = grad_wrt_inputs(model, data)

        n_instances, n_timesteps = masks.shape
        
        index = pd.MultiIndex.from_product([range(n_instances), range(n_timesteps)], names=['instance', 'timestep'])
        df = pd.DataFrame({'grad': grads.reshape(-1), 'masks': masks.reshape(-1).astype(bool)}, index=index)
    
        sensibilities.append(df)
        
    return pd.concat(sensibilities, keys=params, names=level_names)


def record_activations(model, data, device):
    raw_data = []
    model.to(device)

    # set up recording forward hook
    def acitvation_recorder(self, input, output):
        out, _ = output
        try:
            out = out.numpy()
        except TypeError:
            out = out.cpu().numpy()
        raw_data.append(out)

    hook = model.rnn.register_forward_hook(acitvation_recorder)

    # feed stimuli to network
    with torch.no_grad():
        for i, batch in enumerate(data):
            inputs, _ = batch
            inputs = inputs.to(device)

            outputs = model(inputs)[0]

    hook.remove()
    raw_data = np.concatenate(raw_data)

    # Transform data to Pandas DataFrame

    input_idx = range(raw_data.shape[0])
    timesteps = range(raw_data.shape[1])
    units =  range(raw_data.shape[2])

    s = pd.Series(
        data=raw_data.reshape(-1),
        index=pd.MultiIndex.from_product(
            [input_idx, timesteps, units],
            names=['input','timestep', 'unit']),
        name='activation')

    return s


def record_from_models(models, data, device):
    recordings = [
        record_activations(model, test_data, device) for model in models
    ]
    return recordings


def create_recorder(model, device=None, hidden=None, non_blocking=False,
                        prepare_batch=_prepare_batch):
    if device:
        model.to(device)

    def _record(engine, batch):
        model.eval()
        with torch.no_grad():
            inputs, targets = prepare_batch(batch, device=device, non_blocking=non_blocking)
            pred, hidden = model(inputs, hidden)
            
            engine.state.hidden = np.concatenate([engine.state.hidden, hidden.cpu().numpy()], axis=0)

            return pred, targets

    engine = Engine(_record)

    return engine


def get_encoding_recording(model, data, device):
    recorder = create_recorder(model, device)
    recorder.run(data)
    recordings = recorder.state.hidden
    return recordings


# def weighted_activity(model, recordings):
#     df = recordings.groupby(['input', 'unit']).last()
#     weights = model.linear.weight.detach().numpy().T.reshape(-1)

#     gb_act = df.set_index('class', append=True).groupby(['unit', 'class'])

#     def weigh(group):
#         # name in the same order given in groupby
#         unit, label = group.name
#         return group * weights[label, unit]

#     weighted_activity = gb_act.apply(weigh)

#     return weighted_activity


# def mean_weighted_activity(model, recordings):
#     df = recordings.groupby(['input', 'unit']).last()
#     df = df.set_index(
#         'class', append=True).groupby(['unit','class']).mean()
#     df.columns = ['mean activation']

#     df['weight'] = weights.T.reshape(-1)

#     wact = weighted_activity(recordings, model)

#     df['mean weighted activation'] = wact.groupby(['unit', 'class']).mean()

#     return df
