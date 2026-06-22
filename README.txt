# Lagrangian-Particle-Tracking
supplementary code along with paper (Fan et al., 2026. A distinct material isolation and pole-to-Pole teleconnection on Mars.)

Authors
-------

Cong Sun
(sunc3@sustech.edu.cn)

Chen-Shuo Fan
(fancs@sustech.edu.cn)

Version: 1.0
Release date: 2026-06-22


1 OVERVIEW
----------

This code performs Lagrangian particle tracking in the Martian atmosphere to study material isolation and pole-to-pole teleconnection.

By default, the simulation initializes particles randomly and globally in the atmosphere and tracks them for 12 Martian hours.


2 PACKAGE CONTENTS
------------------

The package should contain:

- README.txt
- supplementary.code.py
- demo_input_MY29_Ls270.nc
- demo_output_pts_seeding_global_mode_0.nc
- demo_output_pts_seeding_global_mode_1.nc
- LICENSE

File descriptions:

- supplementary.code.py (Main Lagrangian particle-tracking source code.)
- demo_input_MY29_Ls270.nc (Small demonstration input dataset.)
- demo_output_pts_seeding_global_mode_0.nc (Reference output containing unweighted particle tracking results.)
- demo_output_pts_seeding_global_mode_1.nc (Reference output containing weighted particle tracking results.)


3 SYSTEM REQUIREMENTS
---------------------

3.1 Operating systems

The code is intended to run on a standard 64-bit desktop computer or workstation.


3.2 Python version

Required version: 

Python 3.10 (Other Python 3 versions may work, but only tested version is reported.)


3.3 Python dependencies

The code imports the following packages:

- numpy (2.2.5)
- xarray (2025.4.0)
- tqdm (4.67.1)
- joblib (1.5.2)
- scipy (1.15.2)
- netCDF4 (1.7.2)


3.4 Hardware requirements

No non-standard hardware is required. The code uses joblib for parallel processing. A multicore processor can reduce runtime, but it is not required.


4 INSTALLATION GUIDE
--------------------

4.1 Obtain the code

Clone the repository:

git clone https://github.com/cfan0018/Lagrangian-Particle-Tracking.git

Alternatively, download and extract the code archive.


4.2 Install dependencies

Run:

python -m pip install numpy xarray tqdm joblib scipy netCDF4


4.3 Typical installation time

Typical installation time on a normal desktop computer:

~5 minutes

Installation time depends on the operating system, internet connection, and whether compiled packages are already available.


5 DEMONSTRATION
---------------

5.1 Demo files

The files contain:

- demo_input_MY29_Ls270.nc
- demo_output_pts_seeding_global_mode_0.nc
- demo_output_pts_seeding_global_mode_1.nc

The demo input file is a reduced atmospheric dataset used to test the complete code workflow. The two demo output files serve as reference results.


5.2 Run the demo

From the repository root directory:

python supplementary.code.py


5.3 Expected demo output

The demo should create two files:

- demo_output_pts_seeding_global_mode_0.nc
- demo_output_pts_seeding_global_mode_1.nc


5.4 Expected demo runtime

Measured demo runtime on a normal desktop computer:

- Runtime:          15 minutes
- Operating system: Linux x86_64
- Processor:        Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz
- Memory:           1 TB
- Python:           3.10
- Demo particles:   5000
- Tracking period:  12 Martian hours
- Parallel workers: 1


6 INSTRUCTIONS FOR USE
----------------------

6.1 Configure system settings

In the section labeled:

====== System Setting ======

Set items such as:

- Ls (input filename for hourly data)
- Experiment (experiment name for particle initialization strategy)

Do not use more parallel workers than the available CPU cores.


6.2 Configure the experiment

In the section labeled:

====== Experiment Design ======

Set items such as:

- new_lat (initialized tracer latitudes)
- new_lon (initialized tracer longitudes)
- new_height (initialized tracer altitudes)


6.3 Configure input data

In the section labeled:

====== Input Data ======

Set items such as:

- Coordinate names
- Variable names

Adjust these names to match the input dataset.


6.4 Run the code on your data

From the repository directory:

python supplementary.code.py


6.5 Reproduce the results in the manuscript

In the section labeled:

====== System Setting ======

Set item:

- teststep = 720 + 1 (number of timesteps to run; 720 martian hours)

Ensure that the input dataset for each Martian season contains a time dimension of the required length.


7 LICENSE
---------

This code is distributed under the MIT license. See the LICENSE file for details.
