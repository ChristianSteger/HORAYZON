// Copyright (c) 2022 ETH Zurich, Christian R. Steger
// MIT License

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
#include <tbb/parallel_reduce.h>
#include <netcdf>  // NetCDF4
#include <vector>  // NetCDF4
// #include <netcdfcpp.h>  // NetCDF3
#include <sstream>
#include <iomanip>

using namespace std;
using namespace netCDF;  // NetCDF4
using namespace netCDF::exceptions;  // NetCDF4

//#############################################################################
// Auxiliary functions
//#############################################################################

// Compute linear index from subscripts (2D-array)
inline size_t lin_ind_2d(size_t dim_1, size_t ind_0, size_t ind_1) {
	return (ind_0 * dim_1 + ind_1);
}

// Compute linear index from subscripts (3D-array)
inline size_t lin_ind_3d(size_t dim_1, size_t dim_2, size_t ind_0, size_t ind_1,
	size_t ind_2) {
	return (ind_0 * (dim_1 * dim_2) + ind_1 * dim_2 + ind_2);
}

// Convert degree to radian
inline float deg2rad(float ang) {
	return ((ang / 180.0) * M_PI);
}

// Convert radian to degree
inline float rad2deg(float ang) {
	return ((ang / M_PI) * 180.0);
}

// Cross product
inline void cross_prod(float a_x, float a_y, float a_z, float b_x, float b_y,
	float b_z, float &c_x, float &c_y, float &c_z) {
	c_x = a_y * b_z - a_z * b_y;
    c_y = a_z * b_x - a_x * b_z;
    c_z = a_x * b_y - a_y * b_x;
}

// Matrix vector multiplication
inline void mat_vec_mult(float (&mat)[3][3], float (&vec)[3],
	float (&vec_res)[3]) {

	vec_res[0] = mat[0][0] * vec[0] + mat[0][1] * vec[1] + mat[0][2] * vec[2];
    vec_res[1] = mat[1][0] * vec[0] + mat[1][1] * vec[1] + mat[1][2] * vec[2];
    vec_res[2] = mat[2][0] * vec[0] + mat[2][1] * vec[1] + mat[2][2] * vec[2];

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
	int dem_dim_0, int dem_dim_1, char* geom_type,
	float* vert_simp, int num_vert_simp, int* tri_ind_simp, int num_tri_simp) {

	RTCScene scene = rtcNewScene(device);
  	rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);
  	//rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST | RTC_SCENE_FLAG_COMPACT);
  	//rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST | RTC_SCENE_FLAG_DYNAMIC);
  	//rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_LOW);
  	//rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_HIGH);

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
	// Add triangles for outer simplified domain
	//-------------------------------------------------------------------------

	if (num_vert_simp >= 3) {
	
		cout << "Add triangles for outer simplified domain" << endl;
		printf("- number of verties: %d \n", num_vert_simp);
		printf("- number of triangles: %d \n", num_tri_simp);

		RTCGeometry geom_add = rtcNewGeometry(device,
			RTC_GEOMETRY_TYPE_TRIANGLE);
  		rtcSetSharedGeometryBuffer(geom_add, RTC_BUFFER_TYPE_VERTEX, 0,
  			RTC_FORMAT_FLOAT3, vert_simp, 0, 3*sizeof(float), num_vert_simp);
  		rtcSetSharedGeometryBuffer(geom_add, RTC_BUFFER_TYPE_INDEX, 0,
  			RTC_FORMAT_UINT3, tri_ind_simp, 0, 3*sizeof(int), num_tri_simp);

		// Commit geometry
		rtcCommitGeometry(geom_add);

		rtcAttachGeometry(scene, geom_add);
		rtcReleaseGeometry(geom_add);
	
	}

	//-------------------------------------------------------------------------

	// Commit scene
	rtcCommitScene(scene);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end - start;
	cout << "BVH build time: " << time.count() << " s" << endl;

	return scene;

}

//#############################################################################
// Ray casting
//#############################################################################

//-----------------------------------------------------------------------------
// Cast single ray (occluded; higher performance)
//-----------------------------------------------------------------------------

bool castRay_occluded1(RTCScene scene, float ox, float oy, float oz, float dx,
	float dy, float dz, float dist_search) {
  
	// Intersect context
  	struct RTCIntersectContext context;
  	rtcInitIntersectContext(&context);

  	// Ray structure
  	struct RTCRay ray;
  	ray.org_x = ox;
  	ray.org_y = oy;
  	ray.org_z = oz;
  	ray.dir_x = dx;
  	ray.dir_y = dy;
  	ray.dir_z = dz;
  	ray.tnear = 0.0;
  	//ray.tfar = std::numeric_limits<float>::infinity();
  	ray.tfar = dist_search;
  	//ray.mask = -1;
  	//ray.flags = 0;

  	// Intersect ray with scene
  	rtcOccluded1(scene, &context, &ray);
  
  	return (ray.tfar < 0.0);

}

//-----------------------------------------------------------------------------
// Cast single ray (intersect1; lower performance)
//-----------------------------------------------------------------------------

bool castRay_intersect1(RTCScene scene, float ox, float oy, float oz, float dx,
	float dy, float dz, float dist_search, float &dist) {
  
	// Intersect context
	struct RTCIntersectContext context;
	rtcInitIntersectContext(&context);

  	// Ray hit structure
  	struct RTCRayHit rayhit;
  	rayhit.ray.org_x = ox;
  	rayhit.ray.org_y = oy;
  	rayhit.ray.org_z = oz;
  	rayhit.ray.dir_x = dx;
  	rayhit.ray.dir_y = dy;
  	rayhit.ray.dir_z = dz;
  	rayhit.ray.tnear = 0.0;
  	//rayhit.ray.tfar = std::numeric_limits<float>::infinity();
  	rayhit.ray.tfar = dist_search;
  	//rayhit.ray.mask = -1;
  	//rayhit.ray.flags = 0;
  	rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
  	rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

  	// Intersect ray with scene
  	rtcIntersect1(scene, &context, &rayhit);
  	dist = rayhit.ray.tfar;

  	return (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);

}

//#############################################################################
// Horizon detection algorithms (horizon elevation angle)
//#############################################################################

//-----------------------------------------------------------------------------
// Discrete sampling
//-----------------------------------------------------------------------------

void ray_discrete_sampling(float ray_org_x, float ray_org_y, float ray_org_z,
	size_t azim_num, float hori_acc, float dist_search,
	float elev_ang_low_lim, float elev_ang_up_lim, int elev_num,
	RTCScene scene, size_t &num_rays, float* hori_buffer,
	float* azim_sin, float* azim_cos, float* elev_ang,
	float* elev_cos, float* elev_sin, float (&rot_inv)[3][3]) {

  	for (size_t k = 0; k < azim_num; k++) {
		
		int ind_elev = 0;
		int ind_elev_prev = 0;
  		bool hit = true;
  		while (hit) {
  		
			ind_elev_prev = ind_elev;
			ind_elev = min(ind_elev + 10, elev_num - 1);
			float ray[3] = {elev_cos[ind_elev] * azim_sin[k],
							elev_cos[ind_elev] * azim_cos[k],
							elev_sin[ind_elev]};
			float ray_rot[3];
			mat_vec_mult(rot_inv, ray, ray_rot);
  			hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
  				ray_org_z, ray_rot[0], ray_rot[1], ray_rot[2],
  				dist_search);
  			num_rays += 1;
  
  		}
  		hori_buffer[k] = (elev_ang[ind_elev_prev] + elev_ang[ind_elev]) / 2.0;

  	}

}

//-----------------------------------------------------------------------------
// Binary search
//-----------------------------------------------------------------------------

void ray_binary_search(float ray_org_x, float ray_org_y, float ray_org_z,
	size_t azim_num, float hori_acc, float dist_search,
	float elev_ang_low_lim, float elev_ang_up_lim, int elev_num,
	RTCScene scene, size_t &num_rays, float* hori_buffer,
	float* azim_sin, float* azim_cos, float* elev_ang,
	float* elev_cos, float* elev_sin, float (&rot_inv)[3][3]) {

  	for (size_t k = 0; k < azim_num; k++) {
  	
  		float lim_up = elev_ang_up_lim;
  		float lim_low = elev_ang_low_lim;
  		float elev_samp = (lim_up + lim_low) / 2.0;
  		int ind_elev = ((int)roundf((elev_samp - elev_ang_low_lim)
  			/ (hori_acc / 5.0)));
  		
  		while (max(lim_up - elev_ang[ind_elev],
  			elev_ang[ind_elev] - lim_low) > hori_acc) {

			float ray[3] = {elev_cos[ind_elev] * azim_sin[k],
							elev_cos[ind_elev] * azim_cos[k],
							elev_sin[ind_elev]};
			float ray_rot[3];
			mat_vec_mult(rot_inv, ray, ray_rot);
  			bool hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
  				ray_org_z, ray_rot[0], ray_rot[1], ray_rot[2],
  				dist_search);
  			num_rays += 1;
  			
  			if (hit) {
  				lim_low = elev_ang[ind_elev];
  			} else {
  				lim_up = elev_ang[ind_elev];
  			}
  			elev_samp = (lim_up + lim_low) / 2.0;
  			ind_elev = ((int)roundf((elev_samp - elev_ang_low_lim)
  				/ (hori_acc / 5.0)));
  			
  		}
  		hori_buffer[k] = elev_samp;

  	}

}

//-----------------------------------------------------------------------------
// Guess horizon from previous azimuth direction
//-----------------------------------------------------------------------------

void ray_guess_const(float ray_org_x, float ray_org_y, float ray_org_z,
	size_t azim_num, float hori_acc, float dist_search,
	float elev_ang_low_lim, float elev_ang_up_lim, int elev_num,
	RTCScene scene, size_t &num_rays, float* hori_buffer,
	float* azim_sin, float* azim_cos, float* elev_ang,
	float* elev_cos, float* elev_sin, float (&rot_inv)[3][3]) {

	// ------------------------------------------------------------------------
  	// First azimuth direction (binary search)
  	// ------------------------------------------------------------------------

  	float lim_up = elev_ang_up_lim;
  	float lim_low = elev_ang_low_lim;
  	float elev_samp = (lim_up + lim_low) / 2.0;
  	int ind_elev = ((int)roundf((elev_samp - elev_ang_low_lim)
  		/ (hori_acc / 5.0)));
  		
  	while (max(lim_up - elev_ang[ind_elev],
  		elev_ang[ind_elev] - lim_low) > hori_acc) {

		float ray[3] = {elev_cos[ind_elev] * azim_sin[0],
						elev_cos[ind_elev] * azim_cos[0],
						elev_sin[ind_elev]};
		float ray_rot[3];
		mat_vec_mult(rot_inv, ray, ray_rot);
  		bool hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
  			ray_org_z, ray_rot[0], ray_rot[1], ray_rot[2],
  			dist_search);
  		num_rays += 1;
  			
  		if (hit) {
  			lim_low = elev_ang[ind_elev];
  		} else {
  			lim_up = elev_ang[ind_elev];
  		}
  		elev_samp = (lim_up + lim_low) / 2.0;
  		ind_elev = ((int)roundf((elev_samp - elev_ang_low_lim)
  			/ (hori_acc / 5.0)));
  			
  	}

  	hori_buffer[0] = elev_samp;
  	int ind_elev_prev_azim = ind_elev;

	// ------------------------------------------------------------------------
	// Remaining azimuth directions (guess horizon from previous
	// azimuth direction)
	// ------------------------------------------------------------------------
		
	for (size_t k = 1; k < azim_num; k++) {
	
		// Move upwards
		ind_elev = max(ind_elev_prev_azim - 5, 0);
		int ind_elev_prev = 0;
		bool hit = true;
		int count = 0;
		while (hit) {
			
			ind_elev_prev = ind_elev;
			ind_elev = min(ind_elev + 10, elev_num - 1);
			float ray[3] = {elev_cos[ind_elev] * azim_sin[k],
							elev_cos[ind_elev] * azim_cos[k],
							elev_sin[ind_elev]};
			float ray_rot[3];
			mat_vec_mult(rot_inv, ray, ray_rot);
  			hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
  				ray_org_z, ray_rot[0], ray_rot[1], ray_rot[2],
  				dist_search);
  			num_rays += 1;
  			count += 1;
		
		}
		
		if (count > 1) {

  			elev_samp = (elev_ang[ind_elev_prev] + elev_ang[ind_elev]) / 2.0;
  			ind_elev = ((int)roundf((elev_samp - elev_ang_low_lim)
  				/ (hori_acc / 5.0)));
  			hori_buffer[k] = elev_ang[ind_elev];
  			ind_elev_prev_azim = ind_elev;
  			continue;
		
		}
		
		// Move downwards
		ind_elev = min(ind_elev_prev_azim + 5, elev_num - 1);
		hit = false;
		while (!hit) {
			
			ind_elev_prev = ind_elev;
			ind_elev = max(ind_elev - 10, 0);
			float ray[3] = {elev_cos[ind_elev] * azim_sin[k],
							elev_cos[ind_elev] * azim_cos[k],
							elev_sin[ind_elev]};
			float ray_rot[3];
			mat_vec_mult(rot_inv, ray, ray_rot);
  			hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
  				ray_org_z, ray_rot[0], ray_rot[1], ray_rot[2],
  				dist_search);
  			num_rays += 1;		
		
		}

  		elev_samp = (elev_ang[ind_elev_prev] + elev_ang[ind_elev]) / 2.0;
  		ind_elev = ((int)roundf((elev_samp - elev_ang_low_lim)
  			/ (hori_acc / 5.0)));
  		hori_buffer[k] = elev_ang[ind_elev];
  		ind_elev_prev_azim = ind_elev;
			
	}

}

//-----------------------------------------------------------------------------
// Declare function pointer and assign function
//-----------------------------------------------------------------------------

void (*function_pointer)(float ray_org_x, float ray_org_y, float ray_org_z,
	size_t azim_num, float hori_acc, float dist_search,
	float elev_ang_low_lim, float elev_ang_up_lim, int elev_num,
	RTCScene scene, size_t &num_rays, float* hori_buffer,
	float* azim_sin, float* azim_cos, float* elev_ang,
	float* elev_cos, float* elev_sin, float (&rot_inv)[3][3]);

//#############################################################################
// Horizon detection algorithms (horizon elevation angle and distance)
//#############################################################################

//-----------------------------------------------------------------------------
// Discrete sampling
//-----------------------------------------------------------------------------

void ray_discrete_sampling_hori_dist(float ray_org_x, float ray_org_y,
	float ray_org_z, size_t azim_num, float hori_acc, float dist_search,
	float elev_ang_low_lim, float elev_ang_up_lim, int elev_num,
	RTCScene scene, size_t &num_rays, float* hori_buffer, float* dist_buffer,
	float* azim_sin, float* azim_cos, float* elev_ang,
	float* elev_cos, float* elev_sin, float (&rot_inv)[3][3]) {

	float dist = 0.0;
	float dist_hit = 0.0;

  	for (size_t k = 0; k < azim_num; k++) {
		
		int ind_elev = 0;
		int ind_elev_prev = 0;
  		bool hit = true;
  		while (hit) {
  		
			ind_elev_prev = ind_elev;
			ind_elev = min(ind_elev + 10, elev_num - 1);
			float ray[3] = {elev_cos[ind_elev] * azim_sin[k],
							elev_cos[ind_elev] * azim_cos[k],
							elev_sin[ind_elev]};
			float ray_rot[3];
			mat_vec_mult(rot_inv, ray, ray_rot);
  			hit = castRay_intersect1(scene, ray_org_x, ray_org_y,
  				ray_org_z, ray_rot[0], ray_rot[1], ray_rot[2],
  				dist_search, dist);
  			if (hit) {
  				dist_hit = dist;
  			}
  			num_rays += 1;
  
  		}
  		hori_buffer[k] = (elev_ang[ind_elev_prev] + elev_ang[ind_elev]) / 2.0;
  		dist_buffer[k] = dist_hit;

  	}

}

//-----------------------------------------------------------------------------
// Binary search
//-----------------------------------------------------------------------------

void ray_binary_search_hori_dist(float ray_org_x, float ray_org_y,
	float ray_org_z, size_t azim_num, float hori_acc, float dist_search,
	float elev_ang_low_lim, float elev_ang_up_lim, int elev_num,
	RTCScene scene, size_t &num_rays, float* hori_buffer, float* dist_buffer,
	float* azim_sin, float* azim_cos, float* elev_ang,
	float* elev_cos, float* elev_sin, float (&rot_inv)[3][3]) {

	float dist = 0.0;
	float dist_hit = 0.0;

  	for (size_t k = 0; k < azim_num; k++) {
  	
  		float lim_up = elev_ang_up_lim;
  		float lim_low = elev_ang_low_lim;
  		float elev_samp = (lim_up + lim_low) / 2.0;
  		int ind_elev = ((int)roundf((elev_samp - elev_ang_low_lim)
  			/ (hori_acc / 5.0)));
  		
  		while (max(lim_up - elev_ang[ind_elev],
  			elev_ang[ind_elev] - lim_low) > hori_acc) {

			float ray[3] = {elev_cos[ind_elev] * azim_sin[k],
							elev_cos[ind_elev] * azim_cos[k],
							elev_sin[ind_elev]};
			float ray_rot[3];
			mat_vec_mult(rot_inv, ray, ray_rot);
  			bool hit = castRay_intersect1(scene, ray_org_x, ray_org_y,
  				ray_org_z, ray_rot[0], ray_rot[1], ray_rot[2],
  				dist_search, dist);
  			if (hit) {
  				dist_hit = dist;
  			}
  			num_rays += 1;
  			
  			if (hit) {
  				lim_low = elev_ang[ind_elev];
  			} else {
  				lim_up = elev_ang[ind_elev];
  			}
  			elev_samp = (lim_up + lim_low) / 2.0;
  			ind_elev = ((int)roundf((elev_samp - elev_ang_low_lim)
  				/ (hori_acc / 5.0)));
  			
  		}
  		hori_buffer[k] = elev_samp;
  		dist_buffer[k] = dist_hit;

  	}

}

//-----------------------------------------------------------------------------
// Declare function pointer and assign function
//-----------------------------------------------------------------------------

void (*function_pointer_hori_dist)(float ray_org_x, float ray_org_y, float
	ray_org_z, size_t azim_num, float hori_acc, float dist_search,
	float elev_ang_low_lim, float elev_ang_up_lim, int elev_num,
	RTCScene scene, size_t &num_rays, float* hori_buffer, float* dist_buffer,
	float* azim_sin, float* azim_cos, float* elev_ang,
	float* elev_cos, float* elev_sin, float (&rot_inv)[3][3]);

//#############################################################################
// Compute horizon for gridded domain
//#############################################################################

//-----------------------------------------------------------------------------
// Output writing (NetCDF4 interface)
//-----------------------------------------------------------------------------

void output_netcdf_gridded(float* hori_buffer, int azim_num,
	size_t in_dim_len_0, size_t in_dim_len_1, char* file_out,
	float* x_axis_val, float* y_axis_val, char* x_axis_name, char* y_axis_name,
	char* units) {

  	// Compute azimuth angles
  	float *azim_ang = new float[azim_num];
    for (int i = 0; i < azim_num; i++) {
    	azim_ang[i] = ((2 * M_PI) / azim_num * i);
    }
  	
 	int n_azim = azim_num;
	int n_y = in_dim_len_0;
	int n_x = in_dim_len_1;
  	
  	try { 
  	
  		NcFile dataFile(file_out, NcFile::replace);
  		
		NcDim dim_x = dataFile.addDim(x_axis_name, n_x);
		NcDim dim_y = dataFile.addDim(y_axis_name, n_y); 
		NcDim dim_azim = dataFile.addDim("azim", n_azim);
		
		vector<NcDim> dims_azim;
		dims_azim.push_back(dim_azim);
		NcVar data_azim = dataFile.addVar("azim", ncFloat, dims_azim);
		data_azim.putVar(azim_ang);
		data_azim.putAtt("units", "radian");

		vector<NcDim> dims_x;
		dims_x.push_back(dim_x);
		NcVar data_x = dataFile.addVar(x_axis_name, ncFloat, dims_x);
		data_x.putVar(x_axis_val);
		data_x.putAtt("units", units);

		vector<NcDim> dims_y;
		dims_y.push_back(dim_y);
		NcVar data_y = dataFile.addVar(y_axis_name, ncFloat, dims_y);
		data_y.putVar(y_axis_val);
		data_y.putAtt("units", units);

		vector<NcDim> dims_data;
		dims_data.push_back(dim_y);
		dims_data.push_back(dim_x);
		dims_data.push_back(dim_azim);
		NcVar data = dataFile.addVar("horizon", ncFloat, dims_data);
		data.putVar(hori_buffer);
		data.putAtt("units", "radian");

    }
	catch(NcException& e)
    	{e.what();
      	cout << "Could not write to NetCDF file" << endl;
    }
     		
  	delete[] azim_ang;

}

//-----------------------------------------------------------------------------
// Output writing (NetCDF3 interface; legacy; outdated)
//-----------------------------------------------------------------------------

// void output_netcdf_gridded(float* hori_buffer, int azim_num,
// 	size_t in_dim_len_0, size_t in_dim_len_1, char* file_out,
// 	float* x_axis_val, float* y_axis_val, char* x_axis_name, char* y_axis_name,
// 	char* units) {
// 
//   	// Compute azimuth angles
//   	float *azim_ang = new float[azim_num];
//     for (int i = 0; i < azim_num; i++) {
//     	azim_ang[i] = ((2 * M_PI) / azim_num * i);
//     }
//   	
//  	int n_azim = azim_num;
// 	int n_y = in_dim_len_0;
// 	int n_x = in_dim_len_1;
//   	
//     NcFile dataFile(file_out, NcFile::Replace);
//     if (!dataFile.is_valid()) {
// 		cout << "Could not open NetCDF file" << endl;
//     } else {
// 
//    		NcDim* dim_x = dataFile.add_dim(x_axis_name, n_x);
//     	NcDim* dim_y = dataFile.add_dim(y_axis_name, n_y);
//     	NcDim* dim_azim = dataFile.add_dim("azim", n_azim);
// 
//     	NcVar *data_azim = dataFile.add_var("azim", ncFloat, dim_azim);
//     	data_azim->put(&azim_ang[0], n_azim);
//     	data_azim->add_att("units", "radian");
//     	NcVar *data_x = dataFile.add_var(x_axis_name, ncFloat, dim_x);
//     	data_x->put(&x_axis_val[0], n_x);
//     	data_x->add_att("units", units);
//     	NcVar *data_y = dataFile.add_var(y_axis_name, ncFloat, dim_y);
//     	data_y->put(&y_axis_val[0], n_y);
//     	data_y->add_att("units", units);
// 
//     	NcVar *data = dataFile.add_var("horizon", ncFloat, dim_azim,
//     		dim_y, dim_x);
//     	data->put(&hori_buffer[0], n_azim, n_y, n_x);
//     	data->add_att("units", "radian");
// 
//     }
//      		
//   	delete[] azim_ang;
// 
// }

//-----------------------------------------------------------------------------
// Main function
//-----------------------------------------------------------------------------

void horizon_gridded_comp(float* vert_grid,
	int dem_dim_0, int dem_dim_1,
	float* vec_norm, float* vec_north,
	int offset_0, int offset_1,
	float* hori_buffer,
	int dim_in_0, int dim_in_1,
	int azim_num, float dist_search,
	float hori_acc, char* ray_algorithm, char* geom_type,
	float* vert_simp, int num_vert_simp,
	int* tri_ind_simp, int num_tri_simp,
    char* file_out,
    float* x_axis_val, float* y_axis_val,
    char* x_axis_name, char* y_axis_name, char* units,
    float hori_buffer_size_max,
    float elev_ang_low_lim,
    uint8_t* mask, float hori_fill,
    float ray_org_elev) {

	cout << "--------------------------------------------------------" << endl;
	cout << "Horizon computation with Intel Embree" << endl;
	cout << "--------------------------------------------------------" << endl;

	// Hard-coded settings
  	float elev_ang_up_lim = 89.98;  // upper limit for elevation angle [degree]
  
  	// Initialization
  	auto start_ini = std::chrono::high_resolution_clock::now();

  	RTCDevice device = initializeDevice();
  	RTCScene scene = initializeScene(device, vert_grid, dem_dim_0, dem_dim_1,
  		geom_type, vert_simp, num_vert_simp, tri_ind_simp, num_tri_simp);

  	// Query properties of device
  	// bool cullingEnabled = rtcGetDeviceProperty(device,
  	//	RTC_DEVICE_PROPERTY_BACKFACE_CULLING_ENABLED);
  	// cout << "Backface culling enabled: " << cullingEnabled << endl;

  	auto end_ini = std::chrono::high_resolution_clock::now();
  	std::chrono::duration<double> time = end_ini - start_ini;
  	cout << "Total initialisation time: " << time.count() << " s" << endl;

  	// Unit conversions
  	hori_acc = deg2rad(hori_acc);
  	elev_ang_low_lim = deg2rad(elev_ang_low_lim);
  	elev_ang_up_lim = deg2rad(elev_ang_up_lim);
  	dist_search *= 1000.0;  // [kilometre] -> [metre]

	// Select algorithm for horizon detection
  	cout << "Horizon detection algorithm: ";
  	if (strcmp(ray_algorithm, "discrete_sampling") == 0) {
    	cout << "discrete_sampling" << endl;		
  		function_pointer = ray_discrete_sampling;
  	} else if (strcmp(ray_algorithm, "binary_search") == 0) {
    	cout << "binary search" << endl;
    	function_pointer = ray_binary_search;
  	} else if (strcmp(ray_algorithm, "guess_constant") == 0) {
    	cout << "guess horizon from previous azimuth direction" << endl;
    	function_pointer = ray_guess_const;
	}

  	int num_gc_tot = (dim_in_0 * dim_in_1);
  	int num_gc = 0;
  	for (size_t i = 0; i < (size_t)(dim_in_0 * dim_in_1); i++) {
  		if (mask[i] == 1) {
  			num_gc += 1;
  		}
  	}
  	printf("Number of grid cells for which horizon is computed: %d \n",
  		num_gc);
  	cout << "Fraction of total number of grid cells: " << ((float)num_gc 
  		/ (float)num_gc_tot * 100.0) << " %" << endl;

	float hori_buffer_size = (((float)dim_in_0 * (float)dim_in_1
		* (float)azim_num * 4.0) / pow(10.0, 9.0));
	cout << "Total memory required for horizon output: " 
		<< hori_buffer_size << " GB" << endl;

	size_t num_rays = 0;
  	std::chrono::duration<double> time_ray = std::chrono::seconds(0);
  	std::chrono::duration<double> time_out = std::chrono::seconds(0);
  	
    // ------------------------------------------------------------------------
  	// Allocate and initialise arrays with evaluated trigonometric functions
    // ------------------------------------------------------------------------ 
    
    // Azimuth angles (allocate on stack)
    float azim_sin[azim_num];
    float azim_cos[azim_num];
    float ang;
    for (int i = 0; i < azim_num; i++) {
    	ang = ((2 * M_PI) / azim_num * i);
    	azim_sin[i] = sin(ang);
    	azim_cos[i] = cos(ang);
    }
    
    // Elevation angles (allocate on stack)
    int elev_num = ((int)ceil((elev_ang_up_lim - elev_ang_low_lim)
    	/ (hori_acc / 5.0)) + 1);
    float elev_ang[elev_num];
    float elev_sin[elev_num];
    float elev_cos[elev_num];	
    for (int i = 0; i < elev_num; i++) {
    	ang = elev_ang_up_lim - (hori_acc / 5.0) * i;
    	elev_ang[elev_num - i - 1] = ang;
    	elev_sin[elev_num - i - 1] = sin(ang);
    	elev_cos[elev_num - i - 1] = cos(ang);
    }
  	
    // ------------------------------------------------------------------------
  	// Compute and save horizon in one iteration
    // ------------------------------------------------------------------------

    if (hori_buffer_size <= hori_buffer_size_max) {
    
    	cout << "Compute and save horizon in one iteration" << endl;

    	// --------------------------------------------------------------------
  		// Perform ray tracing
    	// --------------------------------------------------------------------

  		auto start_ray = std::chrono::high_resolution_clock::now();
    
		num_rays += tbb::parallel_reduce(
			tbb::blocked_range<size_t>(0,dim_in_0), 0.0,
			[&](tbb::blocked_range<size_t> r, size_t num_rays) {  // parallel

		//for (size_t i = 0; i < dim_in_0; i++) {  // serial
		for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel
  			for (size_t j = 0; j < (size_t)dim_in_1; j++) {

  				size_t ind_arr = lin_ind_2d(dim_in_1, i, j);
  				size_t ind_hori = lin_ind_3d(dim_in_1, azim_num, i, j, 0);
  				if (mask[ind_arr] == 1) {
    	
    				// Get vector components
    				size_t ind_vec = lin_ind_2d(dim_in_1, i, j) * 3;
  					float norm_x = vec_norm[ind_vec];
  					float north_x = vec_north[ind_vec];
  					ind_vec += 1;
  					float norm_y = vec_norm[ind_vec];
  					float north_y = vec_north[ind_vec];
  					ind_vec += 1;
  					float norm_z = vec_norm[ind_vec];
  					float north_z = vec_north[ind_vec];
  		
  					// Ray origin
  					size_t ind_2d = lin_ind_2d(dem_dim_1, i + offset_0,
  						j + offset_1);
  					float ray_org_x = vert_grid[ind_2d * 3 + 0] 
  						+ norm_x * ray_org_elev;
  					float ray_org_y = vert_grid[ind_2d * 3 + 1] 
  						+ norm_y * ray_org_elev;
  					float ray_org_z = vert_grid[ind_2d * 3 + 2] 
  						+ norm_z * ray_org_elev;
  				
  					// Compute inverse of rotation matrix
  					float east_x, east_y, east_z;
					cross_prod(north_x, north_y, north_z,
							   norm_x, norm_y, norm_z,
							   east_x, east_y, east_z);		
					float rot_inv[3][3] = {{east_x, north_x, norm_x},
									   	   {east_y, north_y, norm_y},
									   	   {east_z, north_z, norm_z}};
  				
  					// Perform ray tracing
  					function_pointer(ray_org_x, ray_org_y, ray_org_z,
  				 		azim_num, hori_acc, dist_search,
  				 		elev_ang_low_lim, elev_ang_up_lim, elev_num,
  				 		scene, num_rays, &hori_buffer[ind_hori],
  				 		azim_sin, azim_cos, elev_ang,
  				 		elev_cos, elev_sin, rot_inv);
  				 	
  				 } else {
  					for (int k = 0; k < azim_num; k++) {
  						hori_buffer[ind_hori] = hori_fill;  // [radian]
  						ind_hori += 1;
  					}
  				 }
    	
  			}
  		}
  	
  		return num_rays;  // parallel
  		}, std::plus<size_t>());  // parallel
    
  		auto end_ray = std::chrono::high_resolution_clock::now();
  		time_ray += (end_ray - start_ray);
  		
    	// --------------------------------------------------------------------

  		// Save horizon to NetCDF file
    	auto start_out = std::chrono::high_resolution_clock::now();
    	output_netcdf_gridded(hori_buffer, azim_num,
    		dim_in_0, dim_in_1, file_out, x_axis_val, y_axis_val,
    		x_axis_name, y_axis_name, units);
  		auto end_out = std::chrono::high_resolution_clock::now();
  		time_out += (end_out - start_out);

    // ------------------------------------------------------------------------
  	// Compute and save horizon in multiple iterations
    // ------------------------------------------------------------------------
  	
  	} else {
  	
  		cout << "Compute and save horizon in multiple iterations" << endl;
  		
  		cout << "Block indices: ";
  		int num_iter = ceil(hori_buffer_size / hori_buffer_size_max);
  		int block_ind[num_iter + 2];
  		if ((dim_in_1 % num_iter) == 0) {
  			for (int i = 0; i < (num_iter + 1); i++) {
  				block_ind[i] = (int)(dim_in_1 / num_iter) * i;
  				cout << block_ind[i] << ", ";
  			}
  		} else {
  			for (int i = 0; i < (num_iter + 1); i++) {
  				block_ind[i] = (int)(dim_in_1 / num_iter) * i;
  				cout << block_ind[i] << ", ";
  			}
  			block_ind[num_iter + 1] = dim_in_1;  		
  			cout << block_ind[num_iter + 1] << ", ";
  			num_iter += 1;
  		}
  		cout << endl;
  		cout << "Number of iterations: " << num_iter << endl;
  		
  		int len_x;
  		for (int m = 0; m < num_iter; m++) {
  		
    		// ----------------------------------------------------------------
  			// Perform ray tracing
    		// ----------------------------------------------------------------
    		
    		auto start_ray = std::chrono::high_resolution_clock::now();
    		
    		len_x = block_ind[m + 1] - block_ind[m];
  	
			num_rays += tbb::parallel_reduce(
				tbb::blocked_range<size_t>(0,dim_in_0), 0.0,
				[&](tbb::blocked_range<size_t> r, size_t num_rays) {

			for (size_t i=r.begin(); i<r.end(); ++i) {
  				for (size_t j = block_ind[m]; j < (size_t)block_ind[m + 1];
  					j++) {

  					size_t ind_arr = lin_ind_2d(dim_in_1, i, j);
  					size_t ind_hori = lin_ind_3d(len_x, azim_num, i,
  						j - block_ind[m], 0);
  					if (mask[ind_arr] == 1) {
    	
    					// Get vector components
    					size_t ind_vec = lin_ind_2d(dim_in_1, i, j) * 3;
  						float norm_x = vec_norm[ind_vec];
  						float north_x = vec_north[ind_vec];
  						ind_vec += 1;
  						float norm_y = vec_norm[ind_vec];
  						float north_y = vec_north[ind_vec];
  						ind_vec += 1;
  						float norm_z = vec_norm[ind_vec];
  						float north_z = vec_north[ind_vec];
  		
  						// Ray origin
  						size_t ind_2d = lin_ind_2d(dem_dim_1, i + offset_0,
  							j + offset_1);
  						float ray_org_x = (vert_grid[ind_2d * 3 + 0]
  							+ norm_x * ray_org_elev);
  						float ray_org_y = (vert_grid[ind_2d * 3 + 1]
  							+ norm_y * ray_org_elev);
  						float ray_org_z = (vert_grid[ind_2d * 3 + 2]
  							+ norm_z * ray_org_elev);
  						
  						// Compute inverse of rotation matrix
  						float east_x, east_y, east_z;
						cross_prod(north_x, north_y, north_z,
								   norm_x, norm_y, norm_z,
								   east_x, east_y, east_z);				
						float rot_inv[3][3] = {{east_x, north_x, norm_x},
											   {east_y, north_y, norm_y},
											   {east_z, north_z, norm_z}};
  			
  						// Perform ray tracing
  						function_pointer(ray_org_x, ray_org_y, ray_org_z,
  				 			azim_num, hori_acc, dist_search,
  				 			elev_ang_low_lim, elev_ang_up_lim, elev_num,
  				 			scene, num_rays, &hori_buffer[ind_hori],
  				 			azim_sin, azim_cos, elev_ang,
  				 			elev_cos, elev_sin, rot_inv);

	   				 	} else {
  						for (int k = 0; k < azim_num; k++) {
  							hori_buffer[ind_hori] = hori_fill;  // [radian]
  							ind_hori += 1;
  						}
  				 	}
    	
  				}
  			}
  	
  			return num_rays;
  			}, std::plus<size_t>());

  			auto end_ray = std::chrono::high_resolution_clock::now();
  			time_ray += (end_ray - start_ray);

    		// ----------------------------------------------------------------
    		
    		auto start_out = std::chrono::high_resolution_clock::now();
    		
    		// Define output name for current iteration
  			std::string file_out_str(file_out);
  			int str_len = file_out_str.length();
    		std::stringstream ss;
			ss << std::setw(2) << std::setfill('0') << (m + 1);
  			std::string file_out_iter = (file_out_str.substr(0, str_len - 3)
  				+ "_p"+ ss.str() + ".nc");
  			char* file_out_iter_c = new char[file_out_iter.length()];
  			strcpy(file_out_iter_c, file_out_iter.c_str() );
  	  		
  	  		// Save horizon to NetCDF file
    		output_netcdf_gridded(hori_buffer, azim_num, dim_in_0, len_x,
    			file_out_iter_c, &x_axis_val[block_ind[m]], y_axis_val,
    			x_axis_name, y_axis_name, units);
    		
    		delete[] file_out_iter_c;

  			auto end_out = std::chrono::high_resolution_clock::now();
  			time_out += (end_out - start_out);

  			cout << "Iteration " << (m + 1) << " completed" << endl;
  	
  		}

  	}
  	
    // ------------------------------------------------------------------------
    
    cout << "Ray tracing time: " << time_ray.count() << " s" << endl;
  	cout << "Writing to NetCDF file: " << time_out.count() << " s" << endl;
    
  	// Print number of rays needed for location and azimuth direction
  	cout << "Number of rays shot: " << num_rays << endl;	
  	float ratio = (float)num_rays / (float)(num_gc * azim_num);
  	printf("Average number of rays per location and azimuth: %.2f \n", ratio);

  	// Release resources allocated through Embree
  	rtcReleaseScene(scene);
  	rtcReleaseDevice(device);

  	auto end_tot = std::chrono::high_resolution_clock::now();
  	time = end_tot - start_ini;
  	cout << "Total run time: " << time.count() << " s" << endl;

	cout << "--------------------------------------------------------" << endl;
 
}

//#############################################################################
// Compute horizon for individual grid cells
//#############################################################################

//-----------------------------------------------------------------------------
// Output writing (NetCDF4 interface)
//-----------------------------------------------------------------------------

void output_netcdf_gridcells(float* hori_buffer, float* dist_buffer,
	int azim_num, int num_gc, char* file_out, int* indices,
	int hori_dist_out) {

  	// Compute azimuth angles
  	float *azim_ang = new float[azim_num];
    for (int i = 0; i < azim_num; i++) {
    	azim_ang[i] = ((2 * M_PI) / azim_num * i);
    }
    
    // Reshape indices data
    int *indices_0 = new int[num_gc];
    int *indices_1 = new int[num_gc];
    int count = 0;
    for (int i = 0; i < num_gc; i++) {
    	indices_0[i] = indices[count];
    	count += 1;
    	indices_1[i] = indices[count];
    	count += 1;
    }

  	try {
  	
  		NcFile dataFile(file_out, NcFile::replace);
  		
		NcDim dim_gc = dataFile.addDim("grid_cells", num_gc);
		NcDim dim_azim = dataFile.addDim("azim", azim_num);
		
		vector<NcDim> dims_azim;
		dims_azim.push_back(dim_azim);
		NcVar data_azim = dataFile.addVar("azim", ncFloat, dims_azim);
		data_azim.putVar(azim_ang);
		data_azim.putAtt("units", "radian");

		vector<NcDim> dims_ind_0;
		dims_ind_0.push_back(dim_gc);
		NcVar data_ind_0 = dataFile.addVar("indices_0", ncInt, dims_ind_0);
		data_ind_0.putVar(indices_0);
		
		vector<NcDim> dims_ind_1;
		dims_ind_1.push_back(dim_gc);
		NcVar data_ind_1 = dataFile.addVar("indices_1", ncInt, dims_ind_1);
		data_ind_1.putVar(indices_1);

		vector<NcDim> dims_data;
		dims_data.push_back(dim_gc);
		dims_data.push_back(dim_azim);
		NcVar data = dataFile.addVar("horizon", ncFloat, dims_data);
		data.putVar(hori_buffer);
		data.putAtt("units", "radian");
		
		if (hori_dist_out == 1) {
			NcVar data_dist = dataFile.addVar("horizon_distance", ncFloat,
			dims_data);
			data_dist.putVar(dist_buffer);
			data_dist.putAtt("units", "metre");
		}

    }
	catch(NcException& e)
    	{e.what();
      	cout << "Could not write to NetCDF file" << endl;
    }
     		
  	delete[] azim_ang;

}

//-----------------------------------------------------------------------------
// Main function
//-----------------------------------------------------------------------------

void horizon_gridcells_comp(float* vert_grid,
	int dem_dim_0, int dem_dim_1,
	int* indices,
	float* vec_norm, float* vec_north,
	int offset_0, int offset_1,
	float* hori_buffer,
	int num_gc,
	int azim_num, float dist_search,
	float hori_acc, char* ray_algorithm, char* geom_type,
    char* file_out,
    float elev_ang_low_lim,
    float ray_org_elev,
    int hori_dist_out) {

	cout << "--------------------------------------------------------" << endl;
	cout << "Horizon computation with Intel Embree" << endl;
	cout << "--------------------------------------------------------" << endl;

	// Hard-coded settings
  	float elev_ang_up_lim = 89.98;  // upper limit for elevation angle [degree]
  	
  	// Dummy variables for simplified triangle mesh
  	float vert_simp[4] = {0.0, 0.0, 0.0, 0.0};
    int num_vert_simp = 1;
    int tri_ind_simp[4] = {0, 0, 0, 0};
    int num_tri_simp = 1;
  
  	// Initialization
  	auto start_ini = std::chrono::high_resolution_clock::now();

  	RTCDevice device = initializeDevice();
  	RTCScene scene = initializeScene(device, vert_grid, dem_dim_0, dem_dim_1,
  		geom_type, vert_simp, num_vert_simp, tri_ind_simp, num_tri_simp);

  	auto end_ini = std::chrono::high_resolution_clock::now();
  	std::chrono::duration<double> time = end_ini - start_ini;
  	cout << "Total initialisation time: " << time.count() << " s" << endl;

  	// Unit conversions
  	hori_acc = deg2rad(hori_acc);
  	elev_ang_low_lim = deg2rad(elev_ang_low_lim);
  	elev_ang_up_lim = deg2rad(elev_ang_up_lim);
  	dist_search *= 1000.0;  // [kilometre] -> [metre]

  	printf("Number of grid cells for which horizon is computed: %d \n",
  		num_gc);

	size_t num_rays = 0;
  	std::chrono::duration<double> time_ray = std::chrono::seconds(0);
  	std::chrono::duration<double> time_out = std::chrono::seconds(0);
  	
    // ------------------------------------------------------------------------
  	// Allocate and initialise arrays with evaluated trigonometric functions
    // ------------------------------------------------------------------------ 
    
    // Azimuth angles (allocate on stack)
    float azim_sin[azim_num];
    float azim_cos[azim_num];
    float ang;
    for (int i = 0; i < azim_num; i++) {
    	ang = ((2 * M_PI) / azim_num * i);
    	azim_sin[i] = sin(ang);
    	azim_cos[i] = cos(ang);
    }
    
    // Elevation angles (allocate on stack)
    int elev_num = ((int)ceil((elev_ang_up_lim - elev_ang_low_lim)
    	/ (hori_acc / 5.0)) + 1);
    float elev_ang[elev_num];
    float elev_sin[elev_num];
    float elev_cos[elev_num];	
    for (int i = 0; i < elev_num; i++) {
    	ang = elev_ang_up_lim - (hori_acc / 5.0) * i;
    	elev_ang[elev_num - i - 1] = ang;
    	elev_sin[elev_num - i - 1] = sin(ang);
    	elev_cos[elev_num - i - 1] = cos(ang);
    }
  	
    // ------------------------------------------------------------------------
  	// Perform ray tracing
    // ------------------------------------------------------------------------
    
    // Dynamically allocate buffer for horizon distance values
    int dist_buffer_len;
    if (hori_dist_out == 0) {
    	dist_buffer_len = 1;
    } else {
    	dist_buffer_len = num_gc * azim_num;
    }
  	float *dist_buffer = new float[dist_buffer_len];

    if (hori_dist_out == 0) {  // (horizon elevation angle)

		// Select algorithm for horizon detection
  		cout << "Horizon detection algorithm: ";
  		if (strcmp(ray_algorithm, "discrete_sampling") == 0) {
    		cout << "discrete_sampling" << endl;		
  			function_pointer = ray_discrete_sampling;
  		} else if (strcmp(ray_algorithm, "binary_search") == 0) {
    		cout << "binary search" << endl;
    		function_pointer = ray_binary_search;
  		} else if (strcmp(ray_algorithm, "guess_constant") == 0) {
    		cout << "guess horizon from previous azimuth direction" << endl;
    		function_pointer = ray_guess_const;
		}

   		auto start_ray = std::chrono::high_resolution_clock::now();
     
		num_rays += tbb::parallel_reduce(
			tbb::blocked_range<size_t>(0,num_gc), 0.0,
			[&](tbb::blocked_range<size_t> r, size_t num_rays) {  // parallel

		//for (size_t i = 0; i < num_gc; i++) {  // serial
		for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel

 			// Get vector components
 			size_t ind_vec = i * 3;
  			float norm_x = vec_norm[ind_vec];
  			float north_x = vec_north[ind_vec];
  			ind_vec += 1;
  			float norm_y = vec_norm[ind_vec];
  			float north_y = vec_north[ind_vec];
  			ind_vec += 1;
  			float norm_z = vec_norm[ind_vec];
  			float north_z = vec_north[ind_vec];

  			// Ray origin
  			size_t ind_2d = lin_ind_2d(dem_dim_1, indices[i * 2] + offset_0,
  				indices[i * 2 + 1] + offset_1);
  			float ray_org_x = vert_grid[ind_2d * 3 + 0] 
  				+ norm_x * ray_org_elev;
  			float ray_org_y = vert_grid[ind_2d * 3 + 1] 
  				+ norm_y * ray_org_elev;
  			float ray_org_z = vert_grid[ind_2d * 3 + 2] 
  				+ norm_z * ray_org_elev;
				
  			// Compute inverse of rotation matrix
  			float east_x, east_y, east_z;
			cross_prod(north_x, north_y, north_z,
				   	   norm_x, norm_y, norm_z,
				   	    east_x, east_y, east_z);		
			float rot_inv[3][3] = {{east_x, north_x, norm_x},
							   	   {east_y, north_y, norm_y},
							   	   {east_z, north_z, norm_z}};
   				
  			// Perform ray tracing
  			size_t ind_out = i * azim_num;
  			function_pointer(ray_org_x, ray_org_y, ray_org_z,
  				azim_num, hori_acc, dist_search,
  				elev_ang_low_lim, elev_ang_up_lim, elev_num,
  				scene, num_rays, &hori_buffer[ind_out],
  				azim_sin, azim_cos, elev_ang,
  				elev_cos, elev_sin, rot_inv);

  		}

  		return num_rays;  // parallel
  		}, std::plus<size_t>());  // parallel
    
  		auto end_ray = std::chrono::high_resolution_clock::now();
  		time_ray += (end_ray - start_ray);
  	
  	} else { // (horizon elevation angle and distance)

		// Select algorithm for horizon detection
  		cout << "Horizon detection algorithm: ";
  		if (strcmp(ray_algorithm, "discrete_sampling") == 0) {
    		cout << "discrete_sampling" << endl;		
  			function_pointer_hori_dist = ray_discrete_sampling_hori_dist;
  		} else if (strcmp(ray_algorithm, "binary_search") == 0) {
    		cout << "binary search" << endl;
    		function_pointer_hori_dist = ray_binary_search_hori_dist;
		}

   		auto start_ray = std::chrono::high_resolution_clock::now();
     
		num_rays += tbb::parallel_reduce(
			tbb::blocked_range<size_t>(0,num_gc), 0.0,
			[&](tbb::blocked_range<size_t> r, size_t num_rays) {  // parallel

		//for (size_t i = 0; i < num_gc; i++) {  // serial
		for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel

 			// Get vector components
 			size_t ind_vec = i * 3;
  			float norm_x = vec_norm[ind_vec];
  			float north_x = vec_north[ind_vec];
  			ind_vec += 1;
  			float norm_y = vec_norm[ind_vec];
  			float north_y = vec_north[ind_vec];
  			ind_vec += 1;
  			float norm_z = vec_norm[ind_vec];
  			float north_z = vec_north[ind_vec];

  			// Ray origin
  			size_t ind_2d = lin_ind_2d(dem_dim_1, indices[i * 2] + offset_0,
  				indices[i * 2 + 1] + offset_1);
  			float ray_org_x = vert_grid[ind_2d * 3 + 0] 
  				+ norm_x * ray_org_elev;
  			float ray_org_y = vert_grid[ind_2d * 3 + 1] 
  				+ norm_y * ray_org_elev;
  			float ray_org_z = vert_grid[ind_2d * 3 + 2] 
  				+ norm_z * ray_org_elev;
				
  			// Compute inverse of rotation matrix
  			float east_x, east_y, east_z;
			cross_prod(north_x, north_y, north_z,
				   	   norm_x, norm_y, norm_z,
				   	    east_x, east_y, east_z);		
			float rot_inv[3][3] = {{east_x, north_x, norm_x},
							   	   {east_y, north_y, norm_y},
							   	   {east_z, north_z, norm_z}};
   				
  			// Perform ray tracing
  			size_t ind_out = i * azim_num;
  			function_pointer_hori_dist(ray_org_x, ray_org_y, ray_org_z,
  				azim_num, hori_acc, dist_search,
  				elev_ang_low_lim, elev_ang_up_lim, elev_num,
  				scene, num_rays, &hori_buffer[ind_out], &dist_buffer[ind_out],
  				azim_sin, azim_cos, elev_ang,
  				elev_cos, elev_sin, rot_inv);

  		}

  		return num_rays;  // parallel
  		}, std::plus<size_t>());  // parallel
    
  		auto end_ray = std::chrono::high_resolution_clock::now();
  		time_ray += (end_ray - start_ray);
  	
  	}

    // ------------------------------------------------------------------------

  	// Save horizon to NetCDF file
     auto start_out = std::chrono::high_resolution_clock::now();
     output_netcdf_gridcells(hori_buffer, dist_buffer,
     	azim_num, num_gc, file_out, indices,
     	hori_dist_out);
   	 auto end_out = std::chrono::high_resolution_clock::now();
   	 time_out += (end_out - start_out);

    // ------------------------------------------------------------------------
    
    cout << "Ray tracing time: " << time_ray.count() << " s" << endl;
  	cout << "Writing to NetCDF file: " << time_out.count() << " s" << endl;
    
  	// Print number of rays needed for location and azimuth direction
  	cout << "Number of rays shot: " << num_rays << endl;	
  	float ratio = (float)num_rays / (float)(num_gc * azim_num);
  	printf("Average number of rays per location and azimuth: %.2f \n", ratio);

  	// Release resources allocated through Embree
  	rtcReleaseScene(scene);
  	rtcReleaseDevice(device);

  	auto end_tot = std::chrono::high_resolution_clock::now();
  	time = end_tot - start_ini;
  	cout << "Total run time: " << time.count() << " s" << endl;

	cout << "--------------------------------------------------------" << endl;
 
}

//#############################################################################
// Compute horizon for arbitrary locations
//#############################################################################

//-----------------------------------------------------------------------------
// Output writing (NetCDF4 interface)
//-----------------------------------------------------------------------------

void output_netcdf_locations(float* hori_buffer, float* dist_buffer,
	int azim_num, int num_loc, char* file_out,
	float* x_axis_val, float* y_axis_val,
	char* x_axis_name, char* y_axis_name, char* units,
	int hori_dist_out) {

  	// Compute azimuth angles
  	float *azim_ang = new float[azim_num];
    for (int i = 0; i < azim_num; i++) {
    	azim_ang[i] = ((2 * M_PI) / azim_num * i);
    }
  	
  	try {
  	
  		NcFile dataFile(file_out, NcFile::replace);
  		
		NcDim dim_loc = dataFile.addDim("locations", num_loc);
		NcDim dim_azim = dataFile.addDim("azim", azim_num);
		
		vector<NcDim> dims_azim;
		dims_azim.push_back(dim_azim);
		NcVar data_azim = dataFile.addVar("azim", ncFloat, dims_azim);
		data_azim.putVar(azim_ang);
		data_azim.putAtt("units", "radian");

		vector<NcDim> dims_x;
		dims_x.push_back(dim_loc);
		NcVar data_x = dataFile.addVar(x_axis_name, ncFloat, dims_x);
		data_x.putVar(x_axis_val);
		data_x.putAtt("units", units);
		
		vector<NcDim> dims_y;
		dims_y.push_back(dim_loc);
		NcVar data_y = dataFile.addVar(y_axis_name, ncFloat, dims_y);
		data_y.putVar(y_axis_val);
		data_y.putAtt("units", units);

		vector<NcDim> dims_data;
		dims_data.push_back(dim_loc);
		dims_data.push_back(dim_azim);
		NcVar data = dataFile.addVar("horizon", ncFloat, dims_data);
		data.putVar(hori_buffer);
		data.putAtt("units", "radian");
		
		if (hori_dist_out == 1) {
			NcVar data_dist = dataFile.addVar("horizon_distance", ncFloat,
			dims_data);
			data_dist.putVar(dist_buffer);
			data_dist.putAtt("units", "metre");
		}

    }
	catch(NcException& e)
    	{e.what();
      	cout << "Could not write to NetCDF file" << endl;
    }
     		
  	delete[] azim_ang;

}

//-----------------------------------------------------------------------------
// Main function
//-----------------------------------------------------------------------------

void horizon_locations_comp(float* vert_grid,
	int dem_dim_0, int dem_dim_1,
	float* coords,
	float* vec_norm, float* vec_north,
	float* hori_buffer,
	int num_loc,
	int azim_num, float dist_search,
	float hori_acc, char* ray_algorithm, char* geom_type,
    char* file_out,
    float* x_axis_val, float* y_axis_val,
    char* x_axis_name, char* y_axis_name, char* units,
    float elev_ang_low_lim,
    float* ray_org_elev,
    int hori_dist_out) {

	cout << "--------------------------------------------------------" << endl;
	cout << "Horizon computation with Intel Embree" << endl;
	cout << "--------------------------------------------------------" << endl;

	// Hard-coded settings
  	float elev_ang_up_lim = 89.98;  // upper limit for elevation angle [degree]

  	// Dummy variables for simplified triangle mesh
  	float vert_simp[4] = {0.0, 0.0, 0.0, 0.0};
    int num_vert_simp = 1;
    int tri_ind_simp[4] = {0, 0, 0, 0};
    int num_tri_simp = 1;

  	// Initialization
  	auto start_ini = std::chrono::high_resolution_clock::now();

  	RTCDevice device = initializeDevice();
  	RTCScene scene = initializeScene(device, vert_grid, dem_dim_0, dem_dim_1,
  		geom_type, vert_simp, num_vert_simp, tri_ind_simp, num_tri_simp);

  	auto end_ini = std::chrono::high_resolution_clock::now();
  	std::chrono::duration<double> time = end_ini - start_ini;
  	cout << "Total initialisation time: " << time.count() << " s" << endl;

  	// Unit conversions
  	hori_acc = deg2rad(hori_acc);
  	elev_ang_low_lim = deg2rad(elev_ang_low_lim);
  	elev_ang_up_lim = deg2rad(elev_ang_up_lim);
  	dist_search *= 1000.0;  // [kilometre] -> [metre]

  	printf("Number of locations for which horizon is computed: %d \n",
  		num_loc);

	size_t num_rays = 0;
  	std::chrono::duration<double> time_ray = std::chrono::seconds(0);
  	std::chrono::duration<double> time_out = std::chrono::seconds(0);
  	
    // ------------------------------------------------------------------------
  	// Allocate and initialise arrays with evaluated trigonometric functions
    // ------------------------------------------------------------------------ 
    
    // Azimuth angles (allocate on stack)
    float azim_sin[azim_num];
    float azim_cos[azim_num];
    float ang;
    for (int i = 0; i < azim_num; i++) {
    	ang = ((2 * M_PI) / azim_num * i);
    	azim_sin[i] = sin(ang);
    	azim_cos[i] = cos(ang);
    }
    
    // Elevation angles (allocate on stack)
    int elev_num = ((int)ceil((elev_ang_up_lim - elev_ang_low_lim)
    	/ (hori_acc / 5.0)) + 1);
    float elev_ang[elev_num];
    float elev_sin[elev_num];
    float elev_cos[elev_num];	
    for (int i = 0; i < elev_num; i++) {
    	ang = elev_ang_up_lim - (hori_acc / 5.0) * i;
    	elev_ang[elev_num - i - 1] = ang;
    	elev_sin[elev_num - i - 1] = sin(ang);
    	elev_cos[elev_num - i - 1] = cos(ang);
    }
  	
    // ------------------------------------------------------------------------
  	// Perform ray tracing
    // ------------------------------------------------------------------------
    
    // Dynamically allocate buffer for horizon distance values
    int dist_buffer_len;
    if (hori_dist_out == 0) {
    	dist_buffer_len = 1;
    } else {
    	dist_buffer_len = num_loc * azim_num;
    }
  	float *dist_buffer = new float[dist_buffer_len];
  	for (int i=0; i<dist_buffer_len; ++i) dist_buffer[i] = NAN;

    if (hori_dist_out == 0) {  // (horizon elevation angle)

		// Select algorithm for horizon detection
  		cout << "Horizon detection algorithm: ";
  		if (strcmp(ray_algorithm, "discrete_sampling") == 0) {
    		cout << "discrete_sampling" << endl;		
  			function_pointer = ray_discrete_sampling;
  		} else if (strcmp(ray_algorithm, "binary_search") == 0) {
    		cout << "binary search" << endl;
    		function_pointer = ray_binary_search;
  		} else if (strcmp(ray_algorithm, "guess_constant") == 0) {
    		cout << "guess horizon from previous azimuth direction" << endl;
    		function_pointer = ray_guess_const;
		}

   		auto start_ray = std::chrono::high_resolution_clock::now();
     
		num_rays += tbb::parallel_reduce(
			tbb::blocked_range<size_t>(0,num_loc), 0.0,
			[&](tbb::blocked_range<size_t> r, size_t num_rays) {  // parallel

		//for (size_t i = 0; i < num_loc; i++) {  // serial
		for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel

 			// Get vector components
 			size_t ind_vec = i * 3;
  			float norm_x = vec_norm[ind_vec];
  			float north_x = vec_north[ind_vec];
  			ind_vec += 1;
  			float norm_y = vec_norm[ind_vec];
  			float north_y = vec_north[ind_vec];
  			ind_vec += 1;
  			float norm_z = vec_norm[ind_vec];
  			float north_z = vec_north[ind_vec];

  			// Compute horizon in case location normal intersects with
  			// triangle mesh
  			float ini_x = coords[i * 3];
  			float ini_y = coords[i * 3 + 1];
  			float ini_z = coords[i * 3 + 2];
  			float dist = 0.0;
  			bool hit = false;
  			hit = castRay_intersect1(scene, ini_x, ini_y, ini_z,
  				norm_x, norm_y, norm_z, 100000.0, dist);
  			if (!hit) {  // check downwards if no intersection is found
   				hit = castRay_intersect1(scene, ini_x, ini_y, ini_z,
   					-norm_x, -norm_y, -norm_z, 100000.0, dist);
  				dist *= -1.0;
  			}
  			if (hit) {
  			
  				// Ray origin
  				float ray_org_x = ini_x + norm_x * (dist + ray_org_elev[i]);
  				float ray_org_y = ini_y + norm_y * (dist + ray_org_elev[i]);
  				float ray_org_z = ini_z + norm_z * (dist + ray_org_elev[i]);
				
  				// Compute inverse of rotation matrix
  				float east_x, east_y, east_z;
				cross_prod(north_x, north_y, north_z,
						   norm_x, norm_y, norm_z,
						   east_x, east_y, east_z);		
				float rot_inv[3][3] = {{east_x, north_x, norm_x},
									   {east_y, north_y, norm_y},
									   {east_z, north_z, norm_z}};
   				
  				// Perform ray tracing
  				size_t ind_out = i * azim_num;
  				function_pointer(ray_org_x, ray_org_y, ray_org_z,
  					azim_num, hori_acc, dist_search,
  					elev_ang_low_lim, elev_ang_up_lim, elev_num,
  					scene, num_rays, &hori_buffer[ind_out],
  					azim_sin, azim_cos, elev_ang,
  					elev_cos, elev_sin, rot_inv);
  				
  			}

  		}

  		return num_rays;  // parallel
  		}, std::plus<size_t>());  // parallel
    
  		auto end_ray = std::chrono::high_resolution_clock::now();
  		time_ray += (end_ray - start_ray);
  	
  	} else { // (horizon elevation angle and distance)

		// Select algorithm for horizon detection
  		cout << "Horizon detection algorithm: ";
  		if (strcmp(ray_algorithm, "discrete_sampling") == 0) {
    		cout << "discrete_sampling" << endl;		
  			function_pointer_hori_dist = ray_discrete_sampling_hori_dist;
  		} else if (strcmp(ray_algorithm, "binary_search") == 0) {
    		cout << "binary search" << endl;
    		function_pointer_hori_dist = ray_binary_search_hori_dist;
		}

   		auto start_ray = std::chrono::high_resolution_clock::now();
     
		num_rays += tbb::parallel_reduce(
			tbb::blocked_range<size_t>(0,num_loc), 0.0,
			[&](tbb::blocked_range<size_t> r, size_t num_rays) {  // parallel

		//for (size_t i = 0; i < num_loc; i++) {  // serial
		for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel

 			// Get vector components
 			size_t ind_vec = i * 3;
  			float norm_x = vec_norm[ind_vec];
  			float north_x = vec_north[ind_vec];
  			ind_vec += 1;
  			float norm_y = vec_norm[ind_vec];
  			float north_y = vec_north[ind_vec];
  			ind_vec += 1;
  			float norm_z = vec_norm[ind_vec];
  			float north_z = vec_north[ind_vec];

  			// Compute horizon in case location normal intersects with
  			// triangle mesh
  			float ini_x = coords[i * 3];
  			float ini_y = coords[i * 3 + 1];
  			float ini_z = coords[i * 3 + 2];
  			float dist = 0.0;
  			bool hit = false;
  			hit = castRay_intersect1(scene, ini_x, ini_y, ini_z,
  				norm_x, norm_y, norm_z, 100000.0, dist);
  			if (!hit) {  // check downwards if no intersection is found
   				hit = castRay_intersect1(scene, ini_x, ini_y, ini_z,
   					-norm_x, -norm_y, -norm_z, 100000.0, dist);
  				dist *= -1.0;
  			}
  			if (hit) {

  				// Ray origin
  				float ray_org_x = ini_x + norm_x * (dist + ray_org_elev[i]);
  				float ray_org_y = ini_y + norm_y * (dist + ray_org_elev[i]);
  				float ray_org_z = ini_z + norm_z * (dist + ray_org_elev[i]);
				
  				// Compute inverse of rotation matrix
  				float east_x, east_y, east_z;
				cross_prod(north_x, north_y, north_z,
						   norm_x, norm_y, norm_z,
						   east_x, east_y, east_z);		
				float rot_inv[3][3] = {{east_x, north_x, norm_x},
									   {east_y, north_y, norm_y},
									   {east_z, north_z, norm_z}};
   				
  				// Perform ray tracing
  				size_t ind_out = i * azim_num;
  				function_pointer_hori_dist(ray_org_x, ray_org_y, ray_org_z,
  					azim_num, hori_acc, dist_search,
  					elev_ang_low_lim, elev_ang_up_lim, elev_num,
  					scene, num_rays, &hori_buffer[ind_out],
  					&dist_buffer[ind_out],
  					azim_sin, azim_cos, elev_ang,
  					elev_cos, elev_sin, rot_inv);
  					
  			}

  		}

  		return num_rays;  // parallel
  		}, std::plus<size_t>());  // parallel
    
  		auto end_ray = std::chrono::high_resolution_clock::now();
  		time_ray += (end_ray - start_ray);
  	
  	}

    // ------------------------------------------------------------------------

  	// Save horizon to NetCDF file
     auto start_out = std::chrono::high_resolution_clock::now();
     output_netcdf_locations(hori_buffer, dist_buffer,
     	azim_num, num_loc, file_out,
     	x_axis_val, y_axis_val,
     	x_axis_name, y_axis_name, units,
     	hori_dist_out);
   	 auto end_out = std::chrono::high_resolution_clock::now();
   	 time_out += (end_out - start_out);

    // ------------------------------------------------------------------------
    
    cout << "Ray tracing time: " << time_ray.count() << " s" << endl;
  	cout << "Writing to NetCDF file: " << time_out.count() << " s" << endl;
    
  	// Print number of rays needed for location and azimuth direction
  	cout << "Number of rays shot: " << num_rays << endl;	
  	float ratio = (float)num_rays / (float)(num_loc * azim_num);
  	printf("Average number of rays per location and azimuth: %.2f \n", ratio);

  	// Release resources allocated through Embree
  	rtcReleaseScene(scene);
  	rtcReleaseDevice(device);

  	auto end_tot = std::chrono::high_resolution_clock::now();
  	time = end_tot - start_ini;
  	cout << "Total run time: " << time.count() << " s" << endl;

	cout << "--------------------------------------------------------" << endl;
 
}
