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
    root_dir = os.getcwd()
    print("Script directory: ", root_dir)
    nel = [2, 4, 6, 8, 10, 12, 14]
    conda_env = "mec"
    nbasis = [7, 9, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]
    rs_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    for rs in rs_vals:
        make_and_change_dir(f"rs_{rs}")
        rs_dir = os.getcwd()
        for ne in nel:
            make_and_change_dir(f"nel_{ne}")
            sys_dir = os.getcwd()
            for nmo in nbasis:
                print("nmo change", os.getcwd())
                make_and_change_dir(f"nmo_{nmo}")
                get_data_jobid = None
                print("Basis set: ", ne, nmo)
                job_name = f"berry_{ne}_{nmo}"
                thc_jobid = launch_slurm_job(
                    job_name,
                    f"{job_name}",
                    root_dir,
                    conda_env,
                    f"run_trotter_berry.py",
                    args=f"{ne} {nmo} {rs}",
                    num_tasks=30,
                )
            os.chdir(rs_dir)
            print("nmo loop", os.getcwd())
        os.chdir(root_dir)
        print("rs loop", os.getcwd())
