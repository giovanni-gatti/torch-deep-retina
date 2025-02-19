To train a model already defined in `torchdeepretina/models.py`, you can simply run the following line at the terminal from the `training_scripts` folder:

    $ python3 main.py hyperparams.json hyperranges.json

The hyperparams.json should have a list of all the desired user setting for the training. The hyperranges.json should have the values wished to be searched over for each desired user setting key.

# Possible Hyperparameters
* `exp_name`: str
    * The name of the main experiment. Model folders will be saved within a folder of this name. *WARNING:* do not use names that include an integer right after an underscore, i.e. do not use names like `myexp_1` or `my_exp_1` or `my_1_exp`. The first integer following an underscore is assumed to be the `exp_num` in some portions of the code.
* `save_root`: str (optional)
    * Path to the folder in which you would like to save your experiment data.
* `exp_num_offset`: int
    * This value is added to the determined experiment number. This allows for trainings to be performed on different machines without `exp_num` conflicts.
* `n_repeats`: int
    * The number of times to repeat any given hyperparameter set
* `cross_val`: bool
    * if true, will cross validate model with `n_cv_folds` train-val splits
* `cross_val_idx`: int or null
    * if `cross_val` is true, this parameter has no effect and will be cycled through during training. If `cross_val` is false, this will specify the train-val split to use during training.
* `n_cv_folds`: int
    * the number of cross validation folds to use
* `save_every_epoch`: bool
    * A boolean determining if the model `state_dict` should be saved for every epoch, or only the most recent epoch.
* `seed`: int, None, or `"exp_num"`
    * This is the random seed for both pytorch and numpy. If None is argued, the current value of `time.time()` is used. If the string `"exp_num"` is used, the seed takes on the value of the assigned experiment number. This makes comparitive searches a little bit easier.
* `startpt`: string or None
    * This is the file path to the desired starting checkpoint if any. Will load the model state dict from the argued checkpoint in addition to the optimizer state dict and the zero dict with the appropriate value for `zero_bias`.

* `model_type`: str
    * The string name of the main model class to be used for training. Options are each of the classes defined in `models.py`
* `n_layers`: int
    * the number of layers to be used. Only applies if using "VaryModel" model type. Must be greater than or equal to one.
* `bnorm`: bool
    * if true, model uses batch normalization where possible
* `bnorm_d`: int
    * determines if model uses batchnorm 2d or 1d. Only options are 1 and 2
* `bias`: bool
    * if true, model uses trainable bias parameters where possible
* `softplus`: bool
    * if true, a softplus activation function is used on the outputs of the model
* `chans`: list of ints
    * the number of channels to be used in the intermediary layers
* `ksizes`: list of ints
    * the kernel sizes of the convolutions corresponding to each layer
* `img_shape`: list of ints
    * the shape of the incoming stimulus to the model (do not include batchsize but do include depth of images)
* `stackconvs`: bool
    * if true, convolutions are trained using linear convolution stacking
* `finalstack`: bool
    * if true, final layer is trained using linear convolution stacking. only applies if fully convolutional model.
* `convgc`: bool
    * if true, ganglion cell layer is convolutional
* `rand_onehot`: bool
    * if true, the onehot layer in retinotopic models is randomly initialized to values between 0 and 1. If false, the values are inititialized as `1/(height*width)` in which the height and width are the last dims of the incoming activations.

* `dataset`: str
    * the name of the dataset to be used for training. code assumes the datasets are located in `~/experiments/data/`. The dataset should be a folder that contains h5 files.
* `cells`: str or list of int
    * if string, "all" is only option which collects all cell recordings from the dataset. Otherwise, you can argue only the specific cells you would like to train with. See `datas.py` for more details.
* `stim_type`: str
    * the name of the h5 file (without the `.h5` extension) contained within the dataset folder
* `shift_labels`: bool
    * if true, the training labels are grouped and shifted by a random amounts within the group. This acts as a random control.
* `lossfxn`: str
    * The name of the loss function that should be used for training the model. Currently options are "PoissonNLLLoss" and "MSELoss"
* `log_poisson`: bool
    * only relevant if using "PoissonNLLLoss" function. If true, inputs are exponentiated before poisson loss is calculated.
* `shuffle`: bool
    * boolean determining if the order of samples with in a batch should be shuffled. This does not shuffle the sequence itself.
* `rand_sample`: bool or None
    * boolean determining if the sampled batches should have randomized order. If None, defaults to value of `shuffle`. When used in conjunction with `shuffle`, allows you to train with a set randomized order of data, or allows you to split the data into training and validation sets with preserved order but still randomize the data during training. Generally, however, the value should be true or null.

* `batch_size`: int
    * the number of samples to used in a single step of SGD
* `n_epochs`: int
    * the number of complete training loops through the data
* `lr`: float
    * the learning rate
* `l2`: float
    * the l2 weight penalty
* `l1`: float
    * the l1 activation penalty applied on the final outputs of the model.
* `noise`: float
    * the standard deviation of the gaussian noise layers
* `drop_p`: float
    * the dropout probability used in between linearly stacked convolutions. Only applies if `stackconvs` is set to true.
* `gc_bias`: bool
    * if true, final layer has a bias parameter
* `bn_moment`: float
    * the momentum of the batchnorm layers

* `prune`: bool
    * if true, layers are pruned
* `prune_layers`: list of str
    * enumerates the layers that should be pruned. If empty list or string "all", all intermediary convolutional layers are pruned.
* `prune_tolerance`: float
    * the maximum drop in accuracy willing to be tolerated for a channel removal
* `prune_intvl`: int
    * the number of epochs to train for when trying a new dropped channel
* `alpha_steps`: int
    * the number of integration steps when calculating the integrated gradient
* `intg_bsize`: int
    * batch size of integrated gradient calculations
* `zero_bias`: bool
    * determines if bias should be zeroed in addition to pruned filter
* `reset_lr`: bool
    * determines if the learning rate should be reset to its initial value after every pruning attempt
* `abssum`: bool
    * if true, will take absolute value of integrated gradient prior to summing over channel. If false, will sum over channel first and then take absolute value. Either case will take mean over time.
* `reset_sd`: bool
    * If true, the model will be reset to the original state dict after each channel is pruned.
* `altn_layers`: bool
    * If true, the pruning will alternate layers for each channel pruned. If false, will exhaustively prune earliest layer first, then move on to subsequent layers in order.

* `retinotopic`: bool
    * determines if retinotopic training. Overwritten by `prune`
* `semantic_scale`: float
    * the weighting of the semantic loss in the total loss
* `semantic_l1`: float
    * the scaling of an l1 penalty applied to the weight matrix of the onehot layer.
