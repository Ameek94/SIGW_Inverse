import numpy as np
import matplotlib.pyplot as plt

# Load data files
datadir = 'NANOGrav15yr_KDE-FreeSpectra_v1.0.0/30f_fs{cp}_ceffyl/'
density = np.load(datadir+"density.npy").squeeze(axis=0)          # shape: (n_frequencies, n_grid_points)
log10rhogrid = np.load(datadir+"log10rhogrid.npy")  # grid for log10rho values
freqs = np.load(datadir+"freqs.npy")                # GW frequencies

print(density.shape, log10rhogrid.shape, freqs.shape)

print(freqs)

# Number of samples to draw from each KDE distribution
n_samples = 50000

# Prepare a list to store sampled data for each frequency
data_list = []

# Assuming density has shape (n_frequencies, n_grid_points)
for i in range(density.shape[0]):
    log_pdf = density[i]
    # Exponentiate the log PDF (subtract max for numerical stability)
    pdf = np.exp(log_pdf - np.max(log_pdf))
    # Normalize the PDF so that its sum equals 1
    pdf /= np.sum(pdf)
    
    # Draw samples from the log10rho grid weighted by the PDF
    samples = np.random.choice(log10rhogrid, size=n_samples, p=pdf)
    data_list.append(samples)

# Create the violin plot using matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
# positions are set to the frequencies so that each violin appears at its GW frequency
v1 = ax.violinplot(data_list, np.log10(freqs),
               widths=0.05)
# Make the violins look good B)
for pc in v1['bodies']:
    pc.set_facecolor(('C0',0.25))
    pc.set_edgecolor('C0')
    pc.set_linestyle('solid')
    pc.set_alpha(0.25)
    pc.set_linewidth(1.5)
# Labeling the axes
ax.set_xlabel("GW Frequency (Hz)")
ax.set_ylabel("log10ρ")
ax.set_title("Violin Plots of ρ vs Frequency")
ax.set(xscale="linear", yscale="linear")
ax.set_ylim(-8.9, -5)
ax.set_xlim(-8.8, -7.68)# plt.show()
plt.savefig('violin_plot.pdf')