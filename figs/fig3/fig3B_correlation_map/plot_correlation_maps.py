import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

if __name__ == "__main__":
    cor_maps = h5py.File("convgc_cor_maps.h5", 'r')
    
    # Iterate through hierarchical data structure
    for cell_file in cor_maps.keys():
        for stim_key in cor_maps[cell_file].keys():
            for model_folder in cor_maps[f"{cell_file}/{stim_key}"].keys():
                for ci in cor_maps[f"{cell_file}/{stim_key}/{model_folder}"].keys():
                    for layer in cor_maps[f"{cell_file}/{stim_key}/{model_folder}/{ci}"].keys():
                        for chan in tqdm(cor_maps[f"{cell_file}/{stim_key}/{model_folder}/{ci}/{layer}"].keys()):
                            # Construct full path
                            path = f"{cell_file}/{stim_key}/{model_folder}/{ci}/{layer}/{chan}/"
                            
                            # Load data
                            cor_map = np.asarray(cor_maps[path + "cor_map"])
                            max_resp = np.asarray(cor_maps[path + "max_resp"])
                            real_resp = np.asarray(cor_maps[path + "real_resp"])
                            
                            # Create figure
                            fig = plt.figure(figsize=(9, 10), dpi=80, facecolor='w', edgecolor='k')
                            plt.clf()
                            gridspec.GridSpec(4, 1)
                            
                            # Plot correlation map
                            plt.subplot2grid((4, 1), (0, 0), rowspan=2)
                            high_cor = round(np.max(abs(cor_map)), 2)
                            low_cor = -high_cor
                            
                            if high_cor < 0.55:
                                continue
                                
                            plt.imshow(cor_map.squeeze(), cmap='seismic', clim=[low_cor, high_cor])
                            ax = plt.gca()
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])
                            
                            # Add colorbar
                            cbar = plt.colorbar()
                            cbar.ax.yaxis.set_major_locator(plt.LinearLocator(2))
                            cbar.ax.set_yticks([low_cor, high_cor])
                            cbar.ax.set_yticklabels([str(low_cor), str(high_cor)])
                            cbar.ax.tick_params(axis='both', which='major', labelsize=17)
                            cbar.set_label("Correlation", rotation=270, fontsize=20)
                            
                            # Plot response comparison
                            plt.subplot2grid((4, 1), (2, 0))
                            ax = plt.gca()
                            
                            # Z-score normalize responses
                            real_z = (real_resp.squeeze() - real_resp.mean()) / real_resp.std()
                            max_z = (max_resp.squeeze() - max_resp.mean()) / max_resp.std()
                            
                            # Plot responses
                            ax.plot(real_z[:200], 'k', linewidth=3)
                            ax.plot(max_z[:200], color='#0066cc', linewidth=7, alpha=.7)
                            
                            # Style the plot
                            ax.set_facecolor("#d4ebf2")
                            
                            # Improve x-axis ticks while keeping original y-axis
                            ax.set_xlim(0, 200)
                            ax.set_xticks([0, 100, 200])
                            ax.set_xticklabels(['0', '1', '2'])
                            ax.set_xlabel("Time (s)", fontsize=20)
                            ax.tick_params(axis='both', which='major', labelsize=17)
                            
                            # Save figures
                            plt.tight_layout()
                            plt.savefig(f"figs/{model_folder}_{ci}_{layer}{chan}.pdf")
                            plt.savefig(f"figs/{model_folder}_{ci}_{layer}{chan}.png")
    
    cor_maps.close()