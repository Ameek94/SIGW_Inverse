from ptarcade.sampler import get_user_args, initialize_pta, setup_sampler
from ptarcade import input_handler, pta_importer, signal_builder
from ptarcade.input_handler import bcolors
from ptarcade.models_utils import ParamDict
from ptarcade import console
from rich import print
from rich.console import Console
from rich.panel import Panel
from ceffyl import Sampler
import numpy as np
import logging
import platform
import shutil
import sys
import time
from functools import partial
from nautilus import Sampler
from HD_model import num_nodes, left_node, right_node, y_min, y_max
import dynesty
# from mpi4py.futures import MPIPoolExecutor

def prior1D(cube,free_nodes,left_node,right_node,y_min,y_max):
    # Order and transform nodes to be in the correct range, from Polychord SortedUniformPrior
    # params = np.atleast_2d(cube.copy())
    params = cube.copy()
    ys = params[free_nodes:]
    ys = ys*(y_max - y_min) + y_min
    # if free_nodes>1:
    #     x = params[:free_nodes]
        # x = params[:,:free_nodes]
        # Npoints = cube.shape[0]
    # x = params[:free_nodes]
    # N = free_nodes
    # t = np.zeros(N)
    # t[N-1] = x[N-1]**(1./N)
    # for n in range(N-2, -1, -1):
    #     t[n] = x[n]**(1./(n+1)) * t[n+1]
    # xs = t*(right_node - left_node) + left_node
    return ys
    # return np.concatenate([xs,ys]) # array

def log_likelihood(params,pta=None):
    # print(params)
    # print(params.shape)
    res = pta.ln_likelihood(params)
    # print(res.shape)
    # print(res)
    res = np.where(np.isnan(res),-1e10, res)
    res = np.where(res<-1e10,-1e10, res)
    return res

def main():
    """Read user inputs, set up sampler and models, and run sampler."""
    console = Console()
    start_cpu = time.process_time()
    start_real = time.perf_counter()

    # num_nodes = 3
    print(f"Running inference with number of nodes: {num_nodes}, free nodes: {num_nodes-2}")
    print(f"left_node = {left_node}, right_node = {right_node}, y_min = {y_min}, y_max = {y_max}")
    free_nodes = num_nodes - 2
    # left_node = -9.
    # right_node = -7.
    # y_min = -8.
    # y_max = 0.

    inputs, input_options = get_user_args()

    psrs = None
    noise_params = None
    emp_dist = None

    # if inputs["config"].mode == "enterprise":
    #     with console.status("Loading Pulsars and noise data...", spinner="bouncingBall"):

    #         # import pta data
    #         psrs, noise_params, emp_dist = get_user_pta_data(inputs)

    #         console.print(f"[bold green]Done loading [blue]{len(psrs)}[/] Pulsars and noise data :heavy_check_mark:\n")


    with console.status("Initializing PTA...", spinner="bouncingBall"):
        pta = initialize_pta(inputs, psrs, noise_params)
        console.print("[bold green]Done initializing PTA :heavy_check_mark:\n")


    # with console.status("Initializing Sampler...", spinner="bouncingBall"):
    #     sampler, x0 = setup_sampler(inputs, input_options, pta, emp_dist)
    #     console.print(f"Initial point: {x0}, shape {len(x0)}\n")
    #     console.print("[bold green]Done initializing Sampler :heavy_check_mark:\n")

    console.print("Done with all initializtions.\nSetup times (including first sample) {:.2f} seconds real, {:.2f} seconds CPU\n".format(
        time.perf_counter()-start_real, time.process_time()-start_cpu));

    start_cpu = time.process_time()
    start_real = time.perf_counter()

    # do_sample(inputs, sampler, x0)

    # real_time = time.perf_counter()-start_real
    # cpu_time = time.process_time()-start_cpu

    N_samples = inputs["config"].N_samples
    print(f"Running {N_samples} samples...\n")

    loglike = partial(log_likelihood,pta=pta)
    prior = partial(prior1D,
                    free_nodes=free_nodes,
                    left_node=left_node,
                    right_node=right_node,
                    y_min=y_min,
                    y_max=y_max)

    ndim = free_nodes + num_nodes
    print(f"ndim = {ndim}")

    # seed the random number generator
    start = time.time()
    rstate = np.random.default_rng(5647)
    # sampler = dynesty.NestedSampler(loglike, prior, ndim, nlive=1500,
    #                            rstate=rstate)
    # sampler.run_nested(dlogz=0.01)
    # results = sampler.results
    # print(results.summary())
    sampler = Sampler(prior, loglike, ndim, pass_dict=False,vectorized=False
                      ,filepath=f'ptarcade_{num_nodes}_linear.h5'
                      ,pool=None )
    sampler.run(verbose=True,f_live=0.005,n_like_max=int(1e6))
    end = time.time()
    print('Sampling complete, time taken: {:.4f} s'.format(end-start))
    print('log Z: {:.4f}'.format(sampler.log_z))
    samples, logl, logwt = sampler.posterior()
    np.savez(f'ptarcade_{num_nodes}_linear.npz',samples=samples,logl=logl,logwt=logwt,logz=sampler.log_z)


    # summary_table = rich.table.Table(title="Run Summary", title_justify="left", box=rich.box.ROUNDED)

    # summary_table.add_column("Time (real)", style="cyan")
    # summary_table.add_column("Time (real)/sample", style="cyan")
    # summary_table.add_column("Time (CPU)", style="magenta")
    # summary_table.add_column("Time (CPU)/sample", style="magenta")


    # summary_table.add_row(f"{real_time:.2f}", f"{real_time/N_samples:.4f}", f"{cpu_time:.2f}", f"{cpu_time/N_samples:.4f}")
    # console.print(summary_table)

if __name__ == "__main__":

    main()
