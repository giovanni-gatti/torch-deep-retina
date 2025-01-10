import os
import torch
import matplotlib.pyplot as plt
import pyret.filtertools as ft
from torchdeepretina.utils import compute_sta, revcor_sta
import torchdeepretina.stimuli as tdrstim
import torchdeepretina as tdr
from tqdm import tqdm
import pickle
import time


# Set either layer_type or layers to None
layer_type = "relu" # "conv", "bnorm", "relu"
layers = None #["sequential.3", "sequential.7"]
pickle_folder = None # Defaults to <layer_type>_pickles
n_samples = 5000
batch_size = 2000
contrast = 0.35
revcor_stas = False # use reverse correlation instead of gradient based

mps_device = torch.device("mps")

models = [
    #"convgc_49_dataset15-10-07_stim_typenaturalscene",
    #"convgc_53_dataset15-10-07_stim_typewhitenoise",
    "15-10-07_naturalscene.pt"
]
for model in tqdm(models):
    model_path = os.path.join("models/", model)
    model = tdr.io.load_model(model_path)
    model.eval()
    model.to(mps_device)
    print(model)

    rfs = dict()
    if layers is None:
        if layer_type.lower() == "conv":
            print("Collecting Convolutional Layers")
            layers = tdr.utils.get_conv_layer_names(model)[:-1]
        elif layer_type.lower() == "relu":
            print("Collecting ReLU Layers")
            layers = tdr.utils.get_layer_names(model, {torch.nn.ReLU})
        elif layer_type.lower() == "bnorm":
            print("Collecting BNorm/AbsBNorm Layers")
            layers = tdr.utils.get_bnorm_layer_names(model)[:-1]
        else:
            raise NotImplemented
    print("Using Layers:")
    print(layers)

    if pickle_folder is None:
        if layer_type is None:
            pickle_folder = "pickles"
        else:
            pickle_folder = layer_type.lower() + "_pickles"
    #layers = ['outputs']
    checkpt = tdr.io.load_checkpoint(model_path)
    seed = "" if "seed" not in checkpt else checkpt["seed"]
    pickle_save = "{}_{}_{}_revcor{}_nsamp{}_ctrst{}_stas.p".format(checkpt['exp_name'],
                                        str(checkpt["stim_type"])+str(checkpt['dataset']),
                                        seed,
                                        revcor_stas,
                                        n_samples,
                                        contrast)
    print("Saving to Folder:", pickle_folder)
    if not os.path.exists(pickle_folder):
        os.mkdir(pickle_folder)
    path = os.path.join(pickle_folder, pickle_save)
    print(pickle_save)
    if os.path.exists(pickle_save):
        continue
    
    spatials = []
    temporals = []
    stas = []
    fig = plt.figure(figsize=(40,10))
    sta_stim = tdrstim.repeat_white(n_samples,nx=model.img_shape[1], # White Noise Stimulus (4960x40x50x50)
                                           contrast=contrast,
                                           n_repeats=3,
                                           rand_spat=True)
    sta_stim = tdrstim.rolling_window(sta_stim, model.img_shape[0])
    sta_stim = torch.FloatTensor(sta_stim).to(mps_device)

    if layers[0] == "outputs":
        for idx in range(model.n_units):
            layer_shape = None
            print("Computing STA", layers[0])
            if revcor_stas:
                with torch.no_grad():
                    sta = revcor_sta(model, contrast=contrast, layer=layers[0],
                                                        cell_index=idx,
                                                        layer_shape=layer_shape,
                                                        n_samples=n_samples,
                                                        batch_size=batch_size,
                                                        X=sta_stim,
                                                        to_numpy=True,
                                                        verbose=True)
            else:
                sta = compute_sta(model, contrast=contrast, layer=layers[0],
                                                        cell_index=idx,
                                                        layer_shape=layer_shape,
                                                        n_samples=n_samples,
                                                        batch_size=batch_size,
                                                        verbose=True)
            spatial, temporal = ft.decompose(sta)
            spatials.append(spatial)
            temporals.append(temporal)
            stas.append(sta)
        rfs['outputs'] = {'spatials':spatials, 'temporals':temporals,"stas":stas}
        save_file = os.path.join(pickle_folder,pickle_save)
        with open(save_file, 'wb') as f:
            pickle.dump(rfs, f)
    else:
        for l in range(len(layers)):
            print("Layer", layers[l])
            spatials = []
            temporals = []
            stas = []
            for i in range(model.chans[l]):
                layer_shape = (model.chans[l], *model.shapes[l])
                row,col = int(layer_shape[1]//2),int(layer_shape[2]//2)
                idx = (i, row, col)
                print("Computing STA", layers[l], "chan", i)
                start_time = time.time()
                if revcor_stas:
                    with torch.no_grad():
                        sta = revcor_sta(model, contrast=contrast, layer=layers[l],
                                                        cell_index=idx,
                                                        layer_shape=layer_shape,
                                                        n_samples=n_samples,
                                                        batch_size=batch_size,
                                                        to_numpy=True,
                                                        X=sta_stim,
                                                        verbose=True)
                else:
                    sta = compute_sta(model, contrast=contrast, layer=layers[l],
                                                            cell_index=idx,
                                                            layer_shape=layer_shape,
                                                            n_samples=n_samples,
                                                            batch_size=batch_size,
                                                            verbose=True)
                    print(sta.shape)
                print("Exec Time:", time.time()-start_time)
                start_time = time.time()
                print("Decomposing")
                spatial, temporal = ft.decompose(sta)
                print("Exec Time:", time.time()-start_time)
                start_time = time.time()
                spatials.append(spatial)
                temporals.append(temporal)
                stas.append(sta)
            print("Layers:", layers[l])
            rfs[layers[l]] = {'spatials':spatials, 'temporals':temporals, "stas":stas}
            save_file = os.path.join(pickle_folder,pickle_save)
            with open(save_file, 'wb') as f:
                pickle.dump(rfs, f)
            