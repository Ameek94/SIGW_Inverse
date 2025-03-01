import subprocess

# List of command line arguments to run the script with
num_nodes_list = [2, 3, 4, 5]

# Path to the plot_results.py script
script_path = '/Users/amk/Library/CloudStorage/OneDrive-SwanseaUniversity/Codes/SIGW_Inverse/NG15/plot_results.py'

# Run the script with each number of nodes
for num_nodes in num_nodes_list:
    print(f"Running plot_results.py with {num_nodes} nodes...")
    subprocess.run(['python', script_path, str(num_nodes)])
    print(f"Completed running plot_results.py with {num_nodes} nodes.")