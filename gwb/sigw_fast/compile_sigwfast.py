import os
import subprocess

# Define the Fortran source file and output module name
fortran_sources = "constants.f90 interp.f90 power_spectrum.f90 integrator.f90 sigwfast.f90"  
module_name = "sigwfast"

# Define compiler flags for optimization and OpenMP
optimisation_flags = "-O3 -march=native -fopenmp"

build_dir = "build_fortran"

CC = "gcc-14"

dep = "openmp" # https://github.com/numpy/numpy/issues/27163

# Construct the f2py command
f2py_command = f"""
CC={CC} \
f2py -c {fortran_sources} -m {module_name} \
    --opt="{optimisation_flags}" --dep {dep} --build-dir {build_dir}
"""

print("Running compilation command:")
print(f2py_command)

# Execute the command
result = subprocess.run(f2py_command, shell=True)

if result.returncode == 0:
    print(f"Compilation successful! You can now import `{module_name}` in Python.")
else:
    print("Compilation failed. Check the error messages above.")
