import torch
import torchdeepretina as tdr
import pandas as pd
import os
import gc


def set_device():
    """
    Set the accelerator.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        # Ideally use MPS when running on Mac
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = set_device()
# Configurations for the script
config = {
    "batch_size"    : 1000, # Batch size for computing model outputs and correlations
    "sim_folder"    : "csvs", # Folder to save comparison csv
    "save_ext"      : "cors.csv", # Extension for saving the csv
    "intr_data_path": "~/Desktop/MSc AI/neuro/neural_code_data/interneuron_data/", # Path to interneuron data folder
    "intr_files"    : ['bipolars_late_2012.h5', 'bipolars_early_2012.h5', 'amacrines_early_2012.h5', 'amacrines_late_2012.h5'], # Interneuron data files
    "slide_steps"   : 0 # Slides the interneuron stimulus in both directions to find better model correlations
}

def main():
    r"""
    For each of the amacrine and bipolar cells (33 in total, divided among 4 files), find the best correlated model hidden unit (corresponding to a model interneuron) among all model checkpoints.
    """
    if not os.path.exists(config["sim_folder"]):
        # Create the folder to save the .csv with the correlations if it does not exist
        os.mkdir(config["sim_folder"])

    # Path to folder containing .pt model checkpoints for each experiment
    prepath = os.path.expanduser("./models")
    # Get the paths to the model checkpoints
    naturalscene_checkpoints = os.listdir(os.path.join(prepath, "naturalscene"))
    naturalscene_checkpoints = [os.path.join("naturalscene", checkpoint) for checkpoint in naturalscene_checkpoints]
    whitenoise_checkpoints   = os.listdir(os.path.join(prepath, "whitenoise"))
    whitenoise_checkpoints   = [os.path.join("whitenoise", checkpoint) for checkpoint in whitenoise_checkpoints]
    # Build the full list of model paths
    model_files = naturalscene_checkpoints + whitenoise_checkpoints
    model_paths = [os.path.join(prepath, f) for f in model_files]

    # Clear the GPU cache
    torch.mps.empty_cache()

    # Build the save file path
    save_file = "interneuron_" + config["save_ext"]
    save_file = os.path.join(config["sim_folder"], save_file)
    print("Models:")
    print("\n".join(model_paths))
    print("Saving to:", save_file)

    main_intr_df = None

    print("\nLoading Interneuron Data")
    interneuron_data = tdr.datas.load_interneuron_data(
        root_path     = config["intr_data_path"],
        files         = config["intr_files"],
        filter_length = 40, # Rolling window size
        stim_keys     = {"boxes"}, # Only load the boxes stimulus (this is the one they used also in the other scripts)
        join_stims    = True, # Join the stimuli in a single dictionary
        window        = True # Apply rolling window of size filter_length
    )
    # Load dictionary of stimuli and membrane potentials, each key is a cell file, i.e., amacrines, bipolars, etc.
    # The data for each stimulus has shape (time, 40, 38, 38)
    # The data for each membrane potential has shape (num_cells, time)
    stim_dict, mem_pot_dict, _ = interneuron_data 

    # Iterate over the files, adding a level to the dictionary
    for k in stim_dict.keys():
        stim_dict[k]    = {"boxes": stim_dict[k]} # Specify the stimulus type
        mem_pot_dict[k] = {"boxes": mem_pot_dict[k]}

    # Iterate over the different models
    for i, model_path in enumerate(model_paths):
        print("\nBeginning model:", model_path, "| {}/{}\n".format(i+1, len(model_paths)))

        # Load a single model
        model = tdr.io.load_model(model_path)
        model.eval()

        # Collect interneuron correlations
        with torch.no_grad():
            print("Computing Interneuron Correlations")
            model.to(DEVICE)

            # Remove last layer because this is generally the ganglion layer, we are interested in the interneurons (amacrines, bipolars)
            layers = tdr.utils.get_conv_layer_names(model)
            layers = layers[:-1]
            print(f"Layers: {layers}")

            # Isolate the model checkpoint name
            path = ("./"+model_path).split("/")[-1]

            # Get a dataframe of correlations for each interneuron cell with the model hidden units for the selected layers
            intr_df = tdr.intracellular.get_intr_cors(
                model        = model, 
                stim_dict    = stim_dict,
                mem_pot_dict = mem_pot_dict,
                layers       = set(layers),
                batch_size   = config["batch_size"],
                slide_steps  = config["slide_steps"],
                abs_val      = True,
                verbose      = True,
                window       = False # Keep False since the rolling window is already done in the interneuron data
                )

            # Keep only best correlations for each interneuron cell
            intr_df = intr_df.sort_values(by="cor", ascending=False)
            # Keep only the first instance, which is the one with the highest correlation after sorting
            intr_df = intr_df.drop_duplicates(["cell_file", "cell_idx"])

            intr_df["save_folder"] = path
            if main_intr_df is None: 
                main_intr_df = intr_df
            else: 
                # Append the correlations for the following models to the same dataframe  
                main_intr_df = pd.concat([main_intr_df, intr_df], axis=0, ignore_index=True)

            # Save the dataframe to a .csv file
            main_intr_df.to_csv(save_file, sep="!", header=True, index=False)
            gc.collect()


if __name__=="__main__":
    main()