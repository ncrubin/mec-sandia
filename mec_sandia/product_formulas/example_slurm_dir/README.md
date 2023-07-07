### Launching slurm job

In the directory you want your data to live (here `example_slurm_directory`) do

```
$ python ~/path/to/mec-sandia/mec_sandia/product_formulas/launch_trotter_numerics.py
```

which will run a range of eta, N and rs values you can modify and create a directory
structure like


```
rs_1.0/:
nel_2  nel_4

rs_1.0/nel_2:
nmo_7

rs_1.0/nel_2/nmo_7:
berry_2_7

rs_1.0/nel_2/nmo_7/berry_2_7:
berry_2_7.dat  berry_2_7.err  berry_2_7.out  berry_spectral_norms.npy  run.sh
```
