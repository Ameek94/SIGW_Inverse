import subprocess
import matplotlib.pyplot as plt
from pdf2image import convert_from_path

# List of command line arguments to run the script with
num_nodes_list = [2, 3, 4, 5]

# Path to the plot_results.py script
script_path = '/Users/amk/Library/CloudStorage/OneDrive-SwanseaUniversity/Codes/SIGW_Inverse/NG15/plot_results.py'

# Run the script with each number of nodes and save the plots
# for num_nodes in num_nodes_list:
#     print(f"Running plot_results.py with {num_nodes} nodes...")
#     subprocess.run(['python', script_path, str(num_nodes)])
#     print(f"Completed running plot_results.py with {num_nodes} nodes.")

# Create a grid of plots with shared x-axis
fig, axs = plt.subplots(len(num_nodes_list), 1, figsize=(18, 6 * len(num_nodes_list)), sharex=True)

for i, num_nodes in enumerate(num_nodes_list):
    # Convert the generated PDF to images
    images = convert_from_path(f'NG15_reconstructed_spectra_{num_nodes}_nodes.pdf')
    
    # Plot the images in the grid
    axs[i].imshow(images[0])
    axs[i].axis('off')
    # axs[i, 0].set_title(f'NG15 reconstructed spectra, number of nodes = {num_nodes}')
    
    # axs[i, 1].imshow(images[0])
    # axs[i, 1].axis('off')
    # axs[i, 1].set_title(f'NG15 reconstructed spectra, number of nodes = {num_nodes}')

# Save the grid of plots to a PDF file
plt.tight_layout()
plt.savefig('NG15_reconstructed_spectra_grid.pdf', bbox_inches='tight')
plt.show()