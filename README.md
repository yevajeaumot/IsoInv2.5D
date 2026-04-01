# IsoInv2.5D

A 2.5D numerical model constrained by radar horizons which determines the age and flow of ice along a flowline.

## Background

This model is based on ice flow equations derived by Parrenin and Hindmarsh, 2007. The forward model was then developed in Python by Parrenin et al (2025, preprint) and is available to download here https://github.com/parrenin/age_flow_line/. This code is for an inverse model which applies the "age_flow_line" forward model and uses radar observed isochrones to constrain the optimisation of ice flow parameters. The details of the inverse model are presented in Chung et al. (2024, preprint).

#### References
Chung, A., Parrenin, F., Mulvaney, R., Vittuari, L., Frezzotti, M., Zanutta, A., Lilien, D. A., Cavitte, M., & Eisen, O. (2024). Age, thinning and spatial origin of the Beyond EPICA ice from a 2.5D ice flow model. EGUsphere Preprint Repository. https://doi.org/10.5194/egusphere-2024-1650

Parrenin F, Hindmarsh R. Influence of a non-uniform velocity field on isochrone geometry along a steady flowline of an ice sheet. Journal of Glaciology. 2007;53(183):612-622. https://doi.org/10.3189/002214307784409298

Parrenin, F., Chung, A., & Martín, C. (2025). age_flow_line-1.0: a fast and accurate numerical age model for a pseudo-steady flow tube of an ice sheet. EGUsphere Preprint Repository. https://doi.org/10.5194/egusphere-2024-3411

## Python environment
This code was developed using an anaconda environment with Python 3.9. The following python modules are required:

* sys
* math
* numpy
* matplotlib
* scipy
* yaml
* time

## Running the model
Type the following commands into your command line

```
cd path-to-age_flow_line_inv
run age_flow_line_inv.py exp_directory/
```

where `path-to-age_flow_line_inv` is the directory containing the model code and `exp_directory` is the name of your experiment directory.

## Experiment directory structure
The parameter file `parameters.yml` contains general parameters for the
experiment. Parameters include: 
- `imax`: number of nodes on the forward model grid
- `ninv`: number of nodes on the grid of inferred parameters ($\bar{a}$ - steady accumulation, $p$ - Lliboutry parameters and $H$ - mechanical ice thickness)
- `nobs`: number of nodes on the observation grid (in this caese observations are radar isochrones)
- `njac`: number of nodes on jacobian grid used to calculate uncertainties
- `inversion`: if set to False, it will run as the forward model (https://github.com/parrenin/age_flow_line/)

For the grid resolutions, in general, this rule should be followed
`imax` $>$ `nobs` $>$ `ninv` $>=$ `njac`

For the forward model, there are several input file in `.txt` format:
- `accumulation.txt`: the surface accumulation along the flow line (inital guess for optimisation)
- `thickness.txt`: the thickness of the ice sheet (from radar observation) along the flow line
- `surface.txt`: the surface elevation along the flow line
- `tube_width.txt`: the flow tube width along the flow line
- `sliding.txt`: the basal sliding along the flow line
- `p_Lliboutry.txt`: the p exponent of the Lliboutry velocity profile along the flow line (inital guess for optimisation)
- `relative_density.txt`: a depth vs relative density profile
- `temporal_factor.txt`: the accumulation/melting relative temporal variations

The inverse model requires (the names of these files can be changed in `parameters.yml`): 
- `isochrones.txt`: the depths of horizons to be compared to modelled isochrones
- `ages.txt`: the ages of horizons to be compared to modelled isochrones
- `agescale.txt`: the age scale of an ice core for comparison with modelled ice core results

An example directory `DC_test` is provided. The results of running this experiment can be found in Chung et al. (2024, preprint).

## Model outputs
If the run went smoothly, these data files for the flowline were produced: 
- `flow_line_output.txt`: is the output for the whole flow line (total flux, accumulation and surface velocity)
- `inverted_results.txt`: the best fit values for inverted parameters with uncertainties and all other inferred values (eg. melting) along the flow line
- `stagnant.txt`: mechanical and observed ice thicknesses with stganant ice layer and melting along flow line

These figures are produced (same as the forward model): 
- `age_pi_theta.pdf`: figure for the age field in the (pi,theta) coordinate system
- `age_x_z.pdf`: is the same using the (x,z) coordinate system
- `age_x_depth.pdf`: is the same using the (x,depth) coordinate system
- `boundary_conditions_x.pdf`: figure with the boundary conditions as a function of x
- `calculated_quantities_x.pdf`: figure with some 1D quantities along the flow line
- `iso-omega_lines_x_z.pdf`: the iso-omega lines in (x,z)
- `iso-omega_lines_x_depth.pdf`: the iso-omega lines in (x,depth)
- `mesh_pi_theta.pdf`: the mesh in (pi,theta)
- `mesh_x_z.pdf`:the mesh in (x,z)
- `mesh_x_depth.pdf`: the mesh in (x,depth)
- `meshs.pdf`: comparison of x distance between nodes in forwards and inverse models and observations
- `R_temporal_factor.pdf`: is the figure with the accu/melting temporal factor
- `stream_lines.pdf`: are the stream lines / trajectories in (x,depth)
- `thinning_analytical_x_z.pdf`: is the thinning function calculated analytically in (x,z)

These figures are produced after running the inverse model:
- `age_misfit.pdf`: the difference between modelled ischrone age and observed isochrone age in the (x,depth) coordinate system
- `age_sigma.pdf`: the age uncertainty in the (x,depth) coordinate system
- `inverted_params.pdf`: the iso-omega lines in (x,depth)

If you have defined virtual ice cores, it has also created some output files:
- `IC_ice_core_output.txt`: is the output for the _IC_ ice core
- `IC_ice_core_vs_depth.pdf`: figure with a few quantities as a function of the depth in the _IC_ ice core
- `IC_ice_core_vs_age.pdf`: figure with a few quantities as a function of the age in the _IC_ ice core

# IsoInv2.5D
