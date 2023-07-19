import os

import numpy as np
import h5py
import subprocess


def make_and_change_dir(dirname):
    """Make a directory and change to it."""
    try:
        os.chdir(dirname)
    except FileNotFoundError:
        os.makedirs(dirname)
        os.chdir(dirname)
    print(os.getcwd())

def launch_slurm_job(
    job_name: str,
    job_dir: str,
    root_dir: str,
    env: str,
    script_name: str,
    args: str = "",
    num_tasks: int = 1,
    launch_mpi_if_multi_core=True,
    dependency=None,
    launch_script_name: str = "run.sh",
    time_limit=None,
) -> int:
    """Launch slurm job

    Parameters
    ----------
    job_name : str
        Name describing job.
    job_dir : str
        Directory to run job in.
    root_dir : str
        Root directory where python script to run lives.
    env : str
        Conda environment to load
    script_name : str
        Name of python script to run.
    args : str
        Arguments to python script. Optional. Default = ''.
    num_tasks : int
        Number of cores to run job on (slurm-ntasks). Optional. Default = 1.
    dependency : int
        Slurm job dependency. Optional. Default None.
    launch_script_name : str
        Name of slurm batch script to write to.
    time_limit : int
        Time limit in hours for job. Default None (infinite).

    Returns
    -------
    slurm_job_id : int
        Slurm job id.
    """
    top_dir = os.getcwd()
    make_and_change_dir(job_dir)
    num_nodes = max(1, num_tasks // 30)
    tasks = 2 * num_tasks  # hyperthreading
    if dependency is not None:
        slurm_job_dep = f"--dependency=afterok:{dependency}"
    else:
        slurm_job_dep = ""
    if time_limit is not None:
        time_limit_string = f"#SBATCH  --time={time_limit}:0:0"
    else:
        time_limit_string = ""
    launch_script=f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --ntasks={tasks} # Request full node
#SBATCH -N {num_nodes}  # Run on a single CPU
#SBATCH -o {job_name}.out
#SBATCH -e {job_name}.err
{time_limit_string}

pwd; hostname; date
source ~/miniconda3/etc/profile.d/conda.sh
conda activate {env}
export OMP_NUM_THREADS=8
python -u {root_dir}/{script_name} {args} > {job_name}.dat"""
    launch = f'{launch_script_name}'
    with open(launch, 'w') as f:
        f.write(launch_script)
    output = subprocess.check_output(f'sbatch {slurm_job_dep} {launch}'.split())
    slurm_jobid = int(output.split()[-1])
    os.chdir(top_dir)
    return slurm_jobid
    #return True


if __name__ == "__main__":
    # set root directory
    root_dir = os.getcwd()
    print("Script directory: ", root_dir)
    
    # set conda env
    conda_env = "pyscf"

    ppd = np.array([2, 3, 4, 5, 6, 7])
    N = ppd**3
    eta = [2, 3, 4, 5]
    timeval = 0.65
    for idx, Nid in enumerate(N):
        make_and_change_dir(f"N_{Nid}")
        Nid_dir = os.getcwd()
        # make_and_change_dir(f"eta_{etaval}")
        for etaval in eta:
            job_name = f"strang_{Nid}_{etaval}"
            thc_jobid = launch_slurm_job(
                job_name,
                f"eta_{etaval}",
                root_dir,
                conda_env,
                f"grid_ham_strang_spectral_norm_exe.py",
                args=f"{timeval} {ppd[idx]} {etaval} 5.0",
                num_tasks=30,
            )
        os.chdir(root_dir)
