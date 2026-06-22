"""
Project: A distinct material isolation and pole-to-pole teleconnection on Mars
Description: Lagrangian particle-tracking code

Authors:
  - Cong Sun (Department of Earth and Space Sciences, Southern University of Science and Technology, Shenzhen, China)
  - Chen-Shuo Fan (Department of Earth and Space Sciences, Southern University of Science and Technology, Shenzhen, China)
Corresponding: Cong Sun <sunc3@sustech.edu.cn>; Chen-Shuo Fan <fancs@sustech.edu.cn>

Version: 1.0 
Date: 2026.06.22
Language/Runtime: Python 3.10

Instruction: 
By default, the simulation initializes particles randomly and globally in the atmosphere, and tracks them for 12 Martian hours. 
Modify the blocks labeled ">>>>>> System Setting <<<<<<", ">>>>>> Experiment Design <<<<<<", and ">>>>>> Input Data <<<<<<" as needed before running a personalized tracking experiment. 
"""


import numpy as np
import xarray as xr
from tqdm import tqdm 
from joblib import Parallel, delayed
from scipy.interpolate import interp1d, RegularGridInterpolator


### >>>>>> System setting <<<<<< (modify as needed)
seconds_one_sol 		= 24 * 60 * 60 + 39 * 60 + 35.244 # seconds per sol
steps_per_delta_time 	= 12
radius 					= 3396200 # planetary radius (m)
g 						= 3.72 # gravity constant
Ls          			= ['demo_input_MY29_Ls270'] # filename for hourly data (rename as needed)
Experiment        	 	= ['pts_seeding_global'] # experiment name for particle initialization strategy (rename as needed)
Mode 					= 0 # particle weighting flag (0: unweighted, 1: weighted)  
num_pts     			= 5000 # number of particles for initialization
random_seed 			= 1000000 # random seed for particle initialization
teststep   				= 12 + 1 # number of timesteps to run (default is 12 martian hours)
n_jobs 					= 1 # number of parallel jobs to run (based on available CPU cores)
### >>>>>> System setting <<<<<< (modify as needed)

### >>>>>> Experiment design <<<<<< (modify as needed)
def scatter_points_on_lat_lon_alt_grid_psi_random(exp, lat_start, lat_end, alt_start, alt_end, topography, num_points, random_seed):

	np.random.seed(random_seed)

	points = []

	for _ in range(num_points):
		
		new_lat    = np.random.uniform(lat_start, lat_end)
		new_lon    = np.random.uniform(-180, 180)
		new_height = np.random.uniform(alt_start, alt_end)

		if exp == 'pts_seeding_global' and new_height >= topography.sel(latitude=new_lat, longitude=new_lon, method="nearest").values: 
			points.append((new_lon, new_lat, new_height))
	
	return points, len(points)
### >>>>>> Experiment design <<<<<< (modify as needed)

def interpolate_along_time(data_subset, time_old, time_new):
	f = interp1d(time_old, data_subset, kind='linear', axis=0, fill_value='extrapolate')
	return f(time_new)

def parallel_interpolation_along_time_axis(data, time_old, time_new):
	num_lon = data.shape[3]
	
	results = Parallel(n_jobs=100)(
		delayed(interpolate_along_time)(data[:, :, :, ilon], time_old, time_new)
		for ilon in range(num_lon)
	)
	
	interpolated_data = np.stack(results, axis=-1)
	return interpolated_data

def calculate_longitude_offset(latitude_deg, distance_m, radius=radius):
	R = radius
	latitude_rad = np.deg2rad(latitude_deg)
	
	delta_lambda_rad = distance_m / ( R * np.cos(latitude_rad))
	
	delta_lambda_deg = np.rad2deg(delta_lambda_rad)
	
	return delta_lambda_deg

def calculate_latitude_offset(distance_m, radius=radius):
	return calculate_longitude_offset(0, distance_m, radius)

def mod_longitude(lon):
	return ((lon + 180) % 360) - 180

def mod_latitude(lat): # boundary of latitude is 90 degree, reflect the latitude if it is out of the boundary
	if lat > 90:
		return 90 - (lat % 90)
	elif lat < -90:
		return - (lat % 90)
	else:
		return lat

def track_particles(time, u, v, w, u_time_interp, v_time_interp, w_time_interp, need_track_initial_points, 
				   alt, lat, lon, sh, time_delta_in_seconds, steps_per_delta_time, lower_boundary, upper_boundary):
	"""
	Track particles through a wind field using Lagrangian tracking.
	
	Parameters:
		time: Time array
		u, v, w: Wind components
		u_time_interp, v_time_interp, w_time_interp: Time interpolated wind fields
		need_track_initial_points: Initial points to track
		alt, lat, lon: Grid coordinates
		time_delta_in_seconds: Time step in seconds
		steps_per_delta_time: Number of steps per time delta
		lower_boundary: Lower altitude boundary
		upper_boundary: Upper altitude boundary
		
	Returns:
		new_position_dict_list: List of dictionaries containing particle positions
	"""
	new_position_dict_list = [] # chain list of new_position_dict
	do_init_dict = {}

	for point_lon, point_lat, point_alt in need_track_initial_points:
		do_init_dict[point_lon, point_lat, point_alt] = True
	wind_dict = {}
	
	for itime in tqdm(range(len(time))[:-1], desc='Lagrangian tracking along time axis'):
		new_position_dict = {}
		
		for point_lon, point_lat, point_alt in need_track_initial_points:
			do_init = do_init_dict[point_lon, point_lat, point_alt]
			if do_init:
				now_lon = point_lon
				now_lat = point_lat
				now_alt = point_alt

				u_grid_interp_function = RegularGridInterpolator((alt, lat, lon), u[itime,:,:,:], method='linear', bounds_error=False, fill_value=None)
				v_grid_interp_function = RegularGridInterpolator((alt, lat, lon), v[itime,:,:,:], method='linear', bounds_error=False, fill_value=None)
				w_grid_interp_function = RegularGridInterpolator((alt, lat, lon), w[itime,:,:,:], method='linear', bounds_error=False, fill_value=None)

				if now_lon < lon.min(): now_lon = lon.min()  # Set to minimum if below
				elif now_lon > lon.max(): now_lon = lon.max()  # Set to maximum if above
				
				if now_lat < lat.min(): now_lat = lat.min()  # Set to minimum if below
				elif now_lat > lat.max(): now_lat = lat.max()  # Set to maximum if above

				interp_point = np.array([now_alt, now_lat, now_lon]).T
				wind_dict[point_lon, point_lat, point_alt] = [u_grid_interp_function(interp_point)[0],\
													  v_grid_interp_function(interp_point)[0], \
														w_grid_interp_function(interp_point)[0]]
				
				do_init_dict[point_lon, point_lat, point_alt] = False
			
			else:
				now_lon, now_lat, now_alt = new_position_dict_list[-1][point_lon, point_lat, point_alt]

			skip_flag = False

			if now_alt == -np.inf or now_alt == np.inf:
				new_position_dict[point_lon, point_lat, point_alt] = (now_alt, now_alt, now_alt)
				wind_dict[point_lon, point_lat, point_alt] = (0,0,0)
				skip_flag = True

			if not skip_flag:
				for itime_step in range(1,steps_per_delta_time+1):
					now_lon_previous = now_lon
					now_lat_previous = now_lat
					now_alt_previous = now_alt

					### calculate the offset of the point with the wind field at the previous time step
					previous_u, previous_v, previous_w = wind_dict[point_lon, point_lat, point_alt]
					delta_x = previous_u * time_delta_in_seconds
					delta_y = previous_v * time_delta_in_seconds
					delta_z = previous_w * time_delta_in_seconds

					longitude_offset = calculate_longitude_offset(point_lat, delta_x)
					latitude_offset = calculate_latitude_offset(delta_y)
					altitude_offset = delta_z 
			
					if now_lat > 90 or now_lat < -90:
						now_lon = mod_longitude(now_lon + longitude_offset + 180)
					else:
						now_lon = mod_longitude(now_lon + longitude_offset)
					now_lat = mod_latitude(now_lat + latitude_offset)

					lat_idx = np.abs(lat - now_lat).argmin()
					lon_idx = np.abs(lon - now_lon).argmin()
					
					tmp_alt = now_alt + altitude_offset
					
					if tmp_alt < sh[lat_idx,lon_idx].values:
						tmp_alt = -np.inf
					elif tmp_alt > upper_boundary:
						tmp_alt = np.inf
					
					now_alt = tmp_alt

					if now_alt == -np.inf or now_alt == np.inf:
						new_position_dict[point_lon, point_lat, point_alt] = (now_lon, now_lat, now_alt)
						wind_dict[point_lon, point_lat, point_alt] = (0,0,0)
						break

					### calculate the offset of the point with the wind field at the current time step    
					u_grid_interp_function = RegularGridInterpolator((alt, lat, lon), u_time_interp[itime*steps_per_delta_time+itime_step,:,:,:], method='linear', bounds_error=False, fill_value=None)
					v_grid_interp_function = RegularGridInterpolator((alt, lat, lon), v_time_interp[itime*steps_per_delta_time+itime_step,:,:,:], method='linear', bounds_error=False, fill_value=None)
					w_grid_interp_function = RegularGridInterpolator((alt, lat, lon), w_time_interp[itime*steps_per_delta_time+itime_step,:,:,:], method='linear', bounds_error=False, fill_value=None)

					if now_lon < lon.min(): now_lon = lon.min()  # Set to minimum if below
					elif now_lon > lon.max(): now_lon = lon.max()  # Set to maximum if above
					
					if now_lat < lat.min(): now_lat = lat.min()  # Set to minimum if below
					elif now_lat > lat.max(): now_lat = lat.max()  # Set to maximum if above

					interp_point = np.array([now_alt, now_lat, now_lon]).T
					u_value = u_grid_interp_function(interp_point)[0]
					v_value = v_grid_interp_function(interp_point)[0]
					w_value = w_grid_interp_function(interp_point)[0]

					delta_x = (u_value + previous_u) * time_delta_in_seconds / 2
					delta_y = (v_value + previous_v) * time_delta_in_seconds / 2
					delta_z = (w_value + previous_w) * time_delta_in_seconds / 2
					wind_dict[point_lon, point_lat, point_alt] = [(u_value + previous_u)/2,\
														(v_value + previous_v)/2, \
														(w_value + previous_w)/2]

					longitude_offset = calculate_longitude_offset(point_lat, delta_x)
					latitude_offset = calculate_latitude_offset(delta_y)
					altitude_offset = delta_z

					if now_lat_previous > 90 or now_lat_previous < -90:
						now_lon = mod_longitude(now_lon_previous + longitude_offset + 180)
					else:
						now_lon = mod_longitude(now_lon_previous + longitude_offset)
					now_lat = mod_latitude(now_lat_previous + latitude_offset)

					lat_idx = np.abs(lat - now_lat).argmin()
					lon_idx = np.abs(lon - now_lon).argmin()
					
					tmp_alt = now_alt_previous + altitude_offset
					
					if tmp_alt < sh[lat_idx,lon_idx].values:
						tmp_alt = -np.inf
					elif tmp_alt > upper_boundary:
						tmp_alt = np.inf
					
					now_alt = tmp_alt
					new_position_dict[point_lon, point_lat, point_alt] = (now_lon, now_lat, now_alt)

		new_position_dict_list.append(new_position_dict)

	return new_position_dict_list

def track_particles_segment(initial_points_segment, time, u, v, w, alt, lat, lon, sh,
						  u_time_interp, v_time_interp, w_time_interp,
						  time_delta_in_seconds, steps_per_delta_time,
						  lower_boundary, upper_boundary):
	"""Track a segment of initial points"""
		
	# Rest of tracking logic from original function
	# Returns new_position_dict_list for this segment
	return track_particles(time, u, v, w, u_time_interp, v_time_interp, w_time_interp, initial_points_segment, 
				   alt, lat, lon, sh, time_delta_in_seconds, steps_per_delta_time, lower_boundary, upper_boundary)


for ls_idx, ls_str in enumerate(Ls):

	### >>>>>> Input data <<<<<< (modify as needed)
	ds 		= xr.open_dataset('./' + ls_str + '.nc') # dimensions in increasing order: time, altitude, latitude, longitude
	lon  	= ds['longitude'].values
	lat  	= ds['latitude'].values 
	alt  	= ds['altitude'].values # unit (m)
	lev  	= ds['altitude_to_pressure'].values # unit (Pa)
	time 	= ds['time'].values # unit (hour)
	sh 		= ds['surface_height'] # unit (m)
	u    	= ds['zonal_wind'].values # unit (m/s)
	v    	= ds['meridional_wind'].values # unit (m/s)
	w    	= ds['vertical_wind'].values # unit (m/s)
	t  		= ds['temperature'].values # unit (K)
	### >>>>>> Input data <<<<<< (modify as needed)

	time_array = time / 24 * seconds_one_sol # unit from hours to seconds

	time       	= time[:teststep]
	u          	= u[:teststep,:,:,:]
	v          	= v[:teststep,:,:,:]
	w          	= w[:teststep,:,:,:]
	time_array 	= time_array[:teststep]

	# Define the new time points for interpolation
	new_time_points = np.linspace(time_array[0], time_array[-1] , steps_per_delta_time * len(time))
	time_delta_in_seconds = (time_array[1] - time_array[0]) / steps_per_delta_time 

	### interp the time series to the new time points with time step of steps_per_delta_time,
	u_time_interp = parallel_interpolation_along_time_axis(u, time_array, new_time_points)
	v_time_interp = parallel_interpolation_along_time_axis(v, time_array, new_time_points)
	w_time_interp = parallel_interpolation_along_time_axis(w, time_array, new_time_points)

	lower_boundary = np.min(alt)
	upper_boundary = np.max(alt)

	for exp_idx, exp_str in enumerate(Experiment):

		print('Ls=%s,Exp=%s' %(ls_str,exp_str))

		### >>>>>> Experiment design <<<<<< (modify as needed)
		scattered_points, scattered_points_len  = scatter_points_on_lat_lon_alt_grid_psi_random(exp_str, lat[0], lat[-1], alt[1], alt[-1], sh, num_pts, random_seed)
		### >>>>>> Experiment design <<<<<< (modify as needed)

		scatter_lon = np.array([item[0] for item in scattered_points])
		scatter_lat = np.array([item[1] for item in scattered_points])
		scatter_alt = np.array([item[2] for item in scattered_points])

		trajectory_dict = {}
		need_track_initial_points = []
		for i, (point_lon, point_lat, point_alt) in enumerate(scattered_points[:scattered_points_len]):
			need_track_initial_points.append((point_lon, point_lat, point_alt))
			trajectory_dict[point_lon, point_lat, point_alt] = (point_lon, point_lat, point_alt)

		# Split initial points into segments
		initial_points_list = list(need_track_initial_points)
		segment_size = len(initial_points_list) // n_jobs ### the number of points in each segment
		if segment_size == 0: segment_size = 1
		segments = [initial_points_list[i:i + segment_size] for i in range(0, len(initial_points_list), segment_size)]

		# Run tracking in parallel
		results = Parallel(n_jobs=n_jobs)(
			delayed(track_particles_segment)(
				segment, time, u, v, w, alt, lat, lon, sh,
				u_time_interp, v_time_interp, w_time_interp,
				time_delta_in_seconds, steps_per_delta_time, 
				lower_boundary, upper_boundary
			) for segment in segments
		)

		# Combine results from all segments
		new_position_dict_list = []
		for timestep in tqdm(range(len(results[0])), desc='Combining results from all segments'):  # For each timestep
			combined_dict = {}
			for result in results:  # For each segment result
				combined_dict.update(result[timestep])
			new_position_dict_list.append(combined_dict)

		weight_particle 		= np.zeros(scattered_points_len)
		passing_by_points_count = np.zeros((len(time)-1, u.shape[1],u.shape[2],u.shape[3]))
		trajectory_record_lat 	= np.zeros((scattered_points_len,len(time)));trajectory_record_lat[:] = np.nan 
		trajectory_record_lon 	= np.zeros((scattered_points_len,len(time)));trajectory_record_lon[:] = np.nan 
		trajectory_record_alt 	= np.zeros((scattered_points_len,len(time)));trajectory_record_alt[:] = np.nan

		for itime in range(len(time)-1):
			for ipoint, one_time_dict_key in enumerate(new_position_dict_list[itime].keys()): # check the passing by points in the space grid in each time step
				point_lon, point_lat, point_alt = list(new_position_dict_list[itime][one_time_dict_key])

				if point_alt == -np.inf or point_alt == np.inf or np.isnan(point_alt):

					trajectory_record_lat[ipoint,itime] = trajectory_record_lat[ipoint,itime-1]
					trajectory_record_lon[ipoint,itime] = trajectory_record_lon[ipoint,itime-1]
					trajectory_record_alt[ipoint,itime] = trajectory_record_alt[ipoint,itime-1]
				
				else:
					
					trajectory_record_lat[ipoint,itime] = point_lat
					trajectory_record_lon[ipoint,itime] = point_lon
					trajectory_record_alt[ipoint,itime] = point_alt
					
					lat_idx = np.abs(lat - point_lat).argmin()
					lon_idx = np.abs(lon - point_lon).argmin()
					alt_idx = np.abs(alt - point_alt).argmin()

					if itime == 0:
					
						weight_lat = np.cos(np.deg2rad(lat[lat_idx]))

						if Mode == 0: 

							weight_particle[ipoint] = weight_lat
						
						if Mode == 1: 

							weight_particle[ipoint] = weight_lat * lev[alt_idx,lat_idx,lon_idx] / t[alt_idx,lat_idx,lon_idx]							

					passing_by_points_count[itime, alt_idx, lat_idx, lon_idx] = passing_by_points_count[itime, alt_idx, lat_idx, lon_idx] + weight_particle[ipoint]
	
		# ------------------------------------------------------------------------------------
		# Summary of output variables from the Lagrangian particle-tracking experiment:
		# ------------------------------------------------------------------------------------
		# passing_by_points_count: 4D array [time, altitude, latitude, longitude]
		# -> Number of particles passing through each grid cell at each time step.
		#
		# trajectory_record_lat: 2D array [point_index, time]
		# -> Latitude of each tracked particle over time.
		#
		# trajectory_record_lon: 2D array [point_index, time]
		# -> Longitude of each tracked particle over time.
		#
		# trajectory_record_alt: 2D array [point_index, time]
		# -> Altitude of each tracked particle over time.
		# ------------------------------------------------------------------------------------

		tracer = xr.DataArray(passing_by_points_count, dims=['time', 'alt', 'lat', 'lon'], coords={'time': np.arange(0,len(time)-1,1), 'alt': alt, 'lat': lat, 'lon':lon})

		ds_output = xr.Dataset({
            'tracer': tracer,
        })

		ds_output.to_netcdf('./demo_output_' + exp_str + '_mode_' + str(Mode) + '.nc')

print("Lagrangian tracking experiment is complete!")
