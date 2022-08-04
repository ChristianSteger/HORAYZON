// Copyright (c) 2022 ETH Zurich, Christian R. Steger
// MIT License

#include "shadow_comp.h"
#include <cstdio>
#include <embree3/rtcore.h>
#include <stdio.h>
#include <math.h>
#include <limits>
#include <stdio.h>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <string.h>
#include <tbb/parallel_for.h>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace shapes;

//#############################################################################
// Auxiliary functions
//#############################################################################

// Compute linear index from subscripts (2D-array)
inline size_t lin_ind_2d(size_t dim_1, size_t ind_0, size_t ind_1) {
	/* Parameters
       ----------
	   dim_1: second dimension length of two-dimensional array [-]
	   ind_0: first array indices [-]
	   ind_1: second array indices [-]
	   
	   Returns
       ----------
	   ind_lin: linear index of array [-]*/
	return (ind_0 * dim_1 + ind_1);
}

// Convert degree to radian
inline float deg2rad(float ang) {
	/* Parameters
       ----------
	   ang: angle [degree]
	   
	   Returns
       ----------
	   ang: angle [radian]*/
	return ((ang / 180.0) * M_PI);
}

// Convert radian to degree
inline float rad2deg(float ang) {
	/* Parameters
       ----------
	   ang: angle [radian]
	   
	   Returns
       ----------
	   ang: angle [degree]*/
	return ((ang / M_PI) * 180.0);
}

// Convert from Kelvin to degree Celsius
inline float K2degC(float temp) {
	/* Parameters
       ----------
	   temp: temperature [Kelvin]
	   
	   Returns
       ----------
	   temp: temperature [degree Celsius]*/
	return (temp - 273.15);
}

// Cross product
inline void cross_prod(float a_x, float a_y, float a_z, float b_x, float b_y,
	float b_z, float &c_x, float &c_y, float &c_z) {
	/* Parameters
       ----------
	   a_x: x-component of vector a [arbitrary]
	   a_y: y-component of vector a [arbitrary]
	   a_z: z-component of vector a [arbitrary]
	   b_x: x-component of vector b [arbitrary]
	   b_y: y-component of vector b [arbitrary]
	   b_z: z-component of vector b [arbitrary]
	   c_x: x-component of vector c [arbitrary]
	   c_y: y-component of vector c [arbitrary]
	   c_z: z-component of vector c [arbitrary]*/
	c_x = a_y * b_z - a_z * b_y;
    c_y = a_z * b_x - a_x * b_z;
    c_z = a_x * b_y - a_y * b_x;
}

// Unit vector
inline void vec_unit(float &v_x, float &v_y, float &v_z) {
	/* Parameters
       ----------
	   v_x: x-component of vector [arbitrary]
	   v_y: y-component of vector [arbitrary]
	   v_z: z-component of vector [arbitrary]*/
	   float mag = sqrt(v_x * v_x + v_y * v_y + v_z * v_z);
	   v_x = v_x / mag;
	   v_y = v_y / mag;
	   v_z = v_z / mag;
}

// Vector rotation (according to Rodrigues' rotation formula)
inline void vec_rot(float k_x, float k_y, float k_z, float theta,
	float &v_x, float &v_y, float &v_z) {
	/* Parameters
       ----------
	   k_x: x-component of unit vector perpendicular to rotation plane [-]
	   k_y: y-component of unit vector perpendicular to rotation plane [-]
	   k_z: z-component of unit vector perpendicular to rotation plane [-]
	   theta: rotation angle [radian]
	   v_x: x-component of rotated vector [-]
	   v_y: y-component of rotated vector [-]
	   v_z: z-component of rotated vector [-]*/
	float cos_theta = cos(theta);
	float sin_theta = sin(theta);
	float part = (k_x * v_x + k_y * v_y + k_z * v_z) * (1.0 - cos_theta);
	float v_x_rot = v_x * cos_theta + (k_y * v_z - k_z * v_y) * sin_theta 
		+ k_x * part;
	float v_y_rot = v_y * cos_theta + (k_z * v_x - k_x * v_z) * sin_theta
		+ k_y * part;
	float v_z_rot = v_z * cos_theta + (k_x * v_y - k_y * v_x) * sin_theta
		+ k_z * part;
	v_x = v_x_rot;
	v_y = v_y_rot;
	v_z = v_z_rot;
}

// Estimate atmospheric refraction
inline float atmos_refrac(float elev_ang_true, float temp, float pressure) {
	/* Parameters
       ----------
	   elev_ang_true: true solar elevation angle [degree]
	   temp: temperature [degree Celsius]
	   pressure: atmospheric pressure [kPa]
	   
	   Returns
       ----------
	   refrac_cor: refraction correction [degree]

	   Reference
       ----------
	   - Saemundsson, P. (1986). "Astronomical Refraction". Sky and Telescope.
	   	 72: 70
	   - Meeus, J. (1998): Astronomical Algorithm - Second edition, p. 106*/
	float lower = -1.0;
	float upper = 90.0;
	elev_ang_true = std::max(lower, std::min(elev_ang_true, upper));
	float refrac_cor = (1.02 / tan(deg2rad(elev_ang_true + 10.3 
		/ (elev_ang_true + 5.11))));
	refrac_cor += 0.0019279;  // set R = 0.0 for h = 90.0 degree
	refrac_cor *= (pressure / 101.0) * (283.0 / (273.0 + temp));
	return refrac_cor * (1.0 / 60.0);
}

//#############################################################################
// Miscellaneous
//#############################################################################

// Namespace
#if defined(RTC_NAMESPACE_USE)
	RTC_NAMESPACE_USE
#endif

// Error function
void errorFunction(void* userPtr, enum RTCError error, const char* str) {
	printf("error %d: %s\n", error, str);
}

// Initialisation of device and registration of error handler
RTCDevice initializeDevice() {
	RTCDevice device = rtcNewDevice(NULL);
  	if (!device) {
    	printf("error %d: cannot create device\n", rtcGetDeviceError(NULL));
    }
  	rtcSetDeviceErrorFunction(device, errorFunction, NULL);
  	return device;
}

//#############################################################################
// Create scene from geometries
//#############################################################################

// Structures for triangle and quad
struct Triangle { int v0, v1, v2; };
struct Quad { int v0, v1, v2, v3; };
// -> above structures must contain 32-bit integers (-> Embree documentation).
//    Theoretically, these integers should be unsigned but the binary
//    representation until 2'147'483'647 is identical between signed/unsigned
//    integer.

// Initialise scene
RTCScene initializeScene(RTCDevice device, float* vert_grid,
	int dem_dim_0, int dem_dim_1, char* geom_type) {

	RTCScene scene = rtcNewScene(device);
  	rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);

  	int num_vert = (dem_dim_0 * dem_dim_1);
  	printf("DEM dimensions: (%d, %d) \n", dem_dim_0, dem_dim_1);
  	printf("Number of vertices: %d \n", num_vert);

	RTCGeometryType rtc_geom_type;
	if (strcmp(geom_type, "triangle") == 0) {
  		rtc_geom_type = RTC_GEOMETRY_TYPE_TRIANGLE;
  	} else if (strcmp(geom_type, "quad") == 0) {
  		rtc_geom_type = RTC_GEOMETRY_TYPE_QUAD;
  	} else { 	
  		rtc_geom_type = RTC_GEOMETRY_TYPE_GRID;
  	}  	

  	RTCGeometry geom = rtcNewGeometry(device, rtc_geom_type);  	
  	rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
  		RTC_FORMAT_FLOAT3, vert_grid, 0, 3*sizeof(float), num_vert);  	
	
	//-------------------------------------------------------------------------
	// Triangle
	//-------------------------------------------------------------------------
	if (strcmp(geom_type, "triangle") == 0) {
		cout << "Selected geometry type: triangle" << endl;
  		int num_tri = ((dem_dim_0 - 1) * (dem_dim_1 - 1)) * 2;
  		printf("Number of triangles: %d \n", num_tri);
  		Triangle* triangles = (Triangle*) rtcSetNewGeometryBuffer(geom,
  			RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle),
  			num_tri);
  		int n = 0;
  		for (int i = 0; i < (dem_dim_0 - 1); i++) {
  			for (int j = 0; j < (dem_dim_1 - 1); j++) {
  	  			triangles[n].v0 = (i * dem_dim_1) + j;
  	  			triangles[n].v1 = (i * dem_dim_1) + j + 1;
  	  			triangles[n].v2 = ((i + 1) * dem_dim_1) + j;
	  			n++;
  	  			triangles[n].v0 = (i * dem_dim_1) + j + 1;
  	  			triangles[n].v1 = ((i + 1) * dem_dim_1) + j + 1;
  	  			triangles[n].v2 = ((i + 1) * dem_dim_1) + j;
  	  			n++;
  			}
  		}
	//-------------------------------------------------------------------------
	// Quad
	//-------------------------------------------------------------------------
  	} else if (strcmp(geom_type, "quad") == 0) {
  		cout << "Selected geometry type: quad" << endl;
		int num_quad = ((dem_dim_0 - 1) * (dem_dim_1 - 1));
  		printf("Number of quads: %d \n", num_quad);							   
  		Quad* quads = (Quad*) rtcSetNewGeometryBuffer(geom,
  			RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, sizeof(Quad),
  			num_quad);
  		int n = 0;
  		for (int i = 0; i < (dem_dim_0 - 1); i++) {
  			for (int j = 0; j < (dem_dim_1 - 1); j++) {
  			//  identical to grid scene (-> otherwise reverse v0, v1, ...)
  	  		quads[n].v0 = (i * dem_dim_1) + j;
  	  		quads[n].v1 = (i * dem_dim_1) + j + 1;
  	  		quads[n].v2 = ((i + 1) * dem_dim_1) + j + 1;
  	  		quads[n].v3 = ((i + 1) * dem_dim_1) + j;
  	  		n++;
  		}
  	}    	
	//-------------------------------------------------------------------------
	// Grid
	//-------------------------------------------------------------------------  	
  	} else {
  		cout << "Selected geometry type: grid" << endl;
		RTCGrid* grid = (RTCGrid*)rtcSetNewGeometryBuffer(geom,
			RTC_BUFFER_TYPE_GRID, 0, RTC_FORMAT_GRID, sizeof(RTCGrid), 1);
    	grid[0].startVertexID = 0;
    	grid[0].stride        = dem_dim_1;
    	grid[0].width         = dem_dim_1;
    	grid[0].height        = dem_dim_0;
  	}
	//-------------------------------------------------------------------------

	auto start = std::chrono::high_resolution_clock::now();

	// Commit geometry
	rtcCommitGeometry(geom);

	rtcAttachGeometry(scene, geom);
	rtcReleaseGeometry(geom);

	//-------------------------------------------------------------------------

	// Commit scene
	rtcCommitScene(scene);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end - start;
	cout << "BVH build time: " << time.count() << " s" << endl;

	return scene;

}

//#############################################################################
// Initialise terrain
//#############################################################################

CppTerrain::CppTerrain() {
    
    device = initializeDevice();
    
}

CppTerrain::~CppTerrain() {

  	// Release resources allocated through Embree
  	rtcReleaseScene(scene);
  	rtcReleaseDevice(device);

}

void CppTerrain::initialise(float* vert_grid,
	int dem_dim_0, int dem_dim_1,
	int offset_0, int offset_1,
	float* vec_tilt,
	float* vec_norm,
	int dim_in_0, int dim_in_1,
	float* surf_enl_fac,
	float* elevation,
	unsigned char* mask,
	char* geom_type,
	float sw_dir_cor_fill,
	float ang_max,
	int refrac_cor) {

	dem_dim_0_cl = dem_dim_0;
	dem_dim_1_cl = dem_dim_1;
	vert_grid_cl = vert_grid;
	offset_0_cl = offset_0;
	offset_1_cl = offset_1;	
	vec_tilt_cl = vec_tilt;
	vec_norm_cl = vec_norm;
	dim_in_0_cl = dim_in_0;
	dim_in_1_cl = dim_in_1;
	surf_enl_fac_cl = surf_enl_fac;
	elevation_cl = elevation;
	mask_cl = mask;
	sw_dir_cor_fill_cl = sw_dir_cor_fill;
	ang_max_cl = ang_max;
	refrac_cor_cl = refrac_cor;
	
	// Parameters for reference atmosphere
	temperature_ref_cl = 283.15;  // reference temperature at sea level [K]
	pressure_ref_cl = 101.0;  // reference pressure at sea level [kPa]
	lapse_rate_cl = 0.0065;  // temperature lapse rate [K m-1]
	float g = 9.81;  // acceleration due to gravity at sea level [m s-2]
	float R_d = 287.0;  // gas constant for dry air [J K􏰅-1 kg􏰅-1]
	exp_cl = (g / (R_d * lapse_rate_cl));  // exponent for barometric formula 

	auto start_ini = std::chrono::high_resolution_clock::now();

	scene = initializeScene(device, vert_grid, dem_dim_0, dem_dim_1,
		geom_type);
	
	auto end_ini = std::chrono::high_resolution_clock::now();
  	std::chrono::duration<double> time = end_ini - start_ini;
  	cout << "Total initialisation time: " << time.count() << " s" << endl;
  	
  	int num_gc_tot = (dim_in_0 * dim_in_1);
  	int num_gc = 0;
  	for (size_t i = 0; i < (size_t)(dim_in_0 * dim_in_1); i++) {
  		if (mask[i] == 1) {
  			num_gc += 1;
  		}
  	}  	
  	printf("Considered grid cells (number): %d \n", num_gc);
  	cout << "Considered grid cells (fraction from total): " << ((float)num_gc 
  		/ (float)num_gc_tot * 100.0) << " %" << endl;
  		
  	if (refrac_cor_cl == 1) {
  		cout << "Account for atmospheric refraction" << endl;
  	}

}

//#############################################################################
// Compute shadow or correction factor for direct downward shortwave radiation
//#############################################################################

void CppTerrain::shadow(float* sun_position, unsigned char* shadow_buffer) {

	float ray_org_elev=0.05;

	tbb::parallel_for(tbb::blocked_range<size_t>(0,dim_in_0_cl),
		[&](tbb::blocked_range<size_t> r) {  // parallel

	//for (size_t i = 0; i < (size_t)dim_in_0_cl; i++) {  // serial
	for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel
  		for (size_t j = 0; j < (size_t)dim_in_1_cl; j++) {
  		
  			size_t ind_arr = lin_ind_2d(dim_in_1_cl, i, j);
  			if (mask_cl[ind_arr] == 1) {

    			// Get components of terrain surface / ellipsoid normal vectors
    			size_t ind_vec = lin_ind_2d(dim_in_1_cl, i, j) * 3;
  				float tilt_x = vec_tilt_cl[ind_vec];
  				float norm_x = vec_norm_cl[ind_vec];
  				ind_vec += 1;
  				float tilt_y = vec_tilt_cl[ind_vec];
  				float norm_y = vec_norm_cl[ind_vec];
  				ind_vec += 1;
  				float tilt_z = vec_tilt_cl[ind_vec];
  				float norm_z = vec_norm_cl[ind_vec];
  
  				// Ray origin
  				size_t ind_2d = lin_ind_2d(dem_dim_1_cl, i + offset_0_cl,
  					j + offset_1_cl);
  				float ray_org_x = (vert_grid_cl[ind_2d * 3 + 0] 
  					+ norm_x * ray_org_elev);
  				float ray_org_y = (vert_grid_cl[ind_2d * 3 + 1] 
  					+ norm_y * ray_org_elev);
  				float ray_org_z = (vert_grid_cl[ind_2d * 3 + 2] 
  					+ norm_z * ray_org_elev);

  				// Compute sun unit vector
  				float sun_x = (sun_position[0] - ray_org_x);
  				float sun_y = (sun_position[1] - ray_org_y);
  				float sun_z = (sun_position[2] - ray_org_z);
  				vec_unit(sun_x, sun_y, sun_z);

  				// Consider atmospheric refraction
  				float dot_prod_ns = (norm_x * sun_x + norm_y * sun_y
  					+ norm_z * sun_z);
  				if (refrac_cor_cl == 1) {
  					float elev_ang_true = 90.0 - rad2deg(acos(dot_prod_ns));
  					float temperature = temperature_ref_cl - (lapse_rate_cl 
  						* elevation_cl[ind_arr]);
  					float pressure = pressure_ref_cl 
  						* pow((temperature / temperature_ref_cl), exp_cl);
  					float refrac_cor = atmos_refrac(elev_ang_true,
  						K2degC(temperature), pressure);
  					float k_x, k_y, k_z;
  					cross_prod(sun_x, sun_y, sun_z, norm_x, norm_y, norm_z,
  						k_x, k_y, k_z);
  					vec_unit(k_x, k_y, k_z);
  					vec_rot(k_x, k_y, k_z, deg2rad(refrac_cor),
  						sun_x, sun_y, sun_z);
  					dot_prod_ns = (norm_x * sun_x + norm_y * sun_y
  						+ norm_z * sun_z);
  				}

  				// Check for self-shadowing
  				float dot_prod_ts = tilt_x * sun_x + tilt_y * sun_y
  					+ tilt_z * sun_z;
  				if (dot_prod_ts > 0.0) {

					// Intersect context
  					struct RTCIntersectContext context;
  					rtcInitIntersectContext(&context);

  					// Ray structure
  					struct RTCRay ray;
  					ray.org_x = ray_org_x;
  					ray.org_y = ray_org_y;
  					ray.org_z = ray_org_z;
  					ray.dir_x = sun_x;
  					ray.dir_y = sun_y;
  					ray.dir_z = sun_z;
  					ray.tnear = 0.0;
  					ray.tfar = std::numeric_limits<float>::infinity();

  					// Intersect ray with scene
  					rtcOccluded1(scene, &context, &ray);

					if (ray.tfar < 0.0) {
						shadow_buffer[ind_arr] = 2;
					} else {
						shadow_buffer[ind_arr] = 0;
					}
			
				} else {
			
					shadow_buffer[ind_arr] = 1;
			
				}
				
			} else {

				shadow_buffer[ind_arr] = 3;
				
			}

		}
	}
	
	}); // parallel

}

//-----------------------------------------------------------------------------

void CppTerrain::sw_dir_cor(float* sun_position, float* sw_dir_cor_buffer) {

	float ray_org_elev=0.05;
	float dot_prod_min = cos(deg2rad(ang_max_cl));

	tbb::parallel_for(tbb::blocked_range<size_t>(0,dim_in_0_cl),
		[&](tbb::blocked_range<size_t> r) {  // parallel

	//for (size_t i = 0; i < (size_t)dim_in_0_cl; i++) {  // serial
	for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel
  		for (size_t j = 0; j < (size_t)dim_in_1_cl; j++) {
  		
  			size_t ind_arr = lin_ind_2d(dim_in_1_cl, i, j);
  			if (mask_cl[ind_arr] == 1) {

    			// Get components of terrain surface / ellipsoid normal vectors
    			size_t ind_vec = lin_ind_2d(dim_in_1_cl, i, j) * 3;
  				float tilt_x = vec_tilt_cl[ind_vec];
  				float norm_x = vec_norm_cl[ind_vec];
  				ind_vec += 1;
  				float tilt_y = vec_tilt_cl[ind_vec];
  				float norm_y = vec_norm_cl[ind_vec];
  				ind_vec += 1;
  				float tilt_z = vec_tilt_cl[ind_vec];
  				float norm_z = vec_norm_cl[ind_vec];
  
  				// Ray origin
  				size_t ind_2d = lin_ind_2d(dem_dim_1_cl, i + offset_0_cl,
  					j + offset_1_cl);
  				float ray_org_x = (vert_grid_cl[ind_2d * 3 + 0] 
  					+ norm_x * ray_org_elev);
  				float ray_org_y = (vert_grid_cl[ind_2d * 3 + 1] 
  					+ norm_y * ray_org_elev);
  				float ray_org_z = (vert_grid_cl[ind_2d * 3 + 2] 
  					+ norm_z * ray_org_elev);

  				// Compute sun unit vector
  				float sun_x = (sun_position[0] - ray_org_x);
  				float sun_y = (sun_position[1] - ray_org_y);
  				float sun_z = (sun_position[2] - ray_org_z);
  				vec_unit(sun_x, sun_y, sun_z);
  				
  				// Consider atmospheric refraction
  				float dot_prod_ns = (norm_x * sun_x + norm_y * sun_y
  					+ norm_z * sun_z);
  				if (refrac_cor_cl == 1) {
  					float elev_ang_true = 90.0 - rad2deg(acos(dot_prod_ns));
  					float temperature = temperature_ref_cl - (lapse_rate_cl 
  						* elevation_cl[ind_arr]);
  					float pressure = pressure_ref_cl 
  						* pow((temperature / temperature_ref_cl), exp_cl);
  					float refrac_cor = atmos_refrac(elev_ang_true,
  						K2degC(temperature), pressure);
  					float k_x, k_y, k_z;
  					cross_prod(sun_x, sun_y, sun_z, norm_x, norm_y, norm_z,
  						k_x, k_y, k_z);
  					vec_unit(k_x, k_y, k_z);
  					vec_rot(k_x, k_y, k_z, deg2rad(refrac_cor),
  						sun_x, sun_y, sun_z);
  					dot_prod_ns = (norm_x * sun_x + norm_y * sun_y
  						+ norm_z * sun_z);
  				}
  			
  				// Check for self-shadowing
  				float dot_prod_ts = tilt_x * sun_x + tilt_y * sun_y 
  					+ tilt_z * sun_z;
  				if (dot_prod_ts > dot_prod_min) {
  			
					// Intersect context
  					struct RTCIntersectContext context;
  					rtcInitIntersectContext(&context);

  					// Ray structure
  					struct RTCRay ray;
  					ray.org_x = ray_org_x;
  					ray.org_y = ray_org_y;
  					ray.org_z = ray_org_z;
  					ray.dir_x = sun_x;
  					ray.dir_y = sun_y;
  					ray.dir_z = sun_z;
  					ray.tnear = 0.0;
  					ray.tfar = std::numeric_limits<float>::infinity();

  					// Intersect ray with scene
  					rtcOccluded1(scene, &context, &ray);

					if (ray.tfar < 0.0) {
						sw_dir_cor_buffer[ind_arr] = 0.0;
					} else {
						if (dot_prod_ns < dot_prod_min) {
							dot_prod_ns = dot_prod_min;
						}
						sw_dir_cor_buffer[ind_arr] = ((dot_prod_ts 
							/ dot_prod_ns) * surf_enl_fac_cl[ind_arr]);	
					}
			
				} else {
			
					sw_dir_cor_buffer[ind_arr] = 0.0;
			
				}
				
			} else {

				sw_dir_cor_buffer[ind_arr] = sw_dir_cor_fill_cl;
				
			}
	
		}
	}
	
	}); // parallel

}


