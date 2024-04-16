// Copyright (c) 2022 ETH Zurich, Christian R. Steger
// MIT License

#ifndef TESTLIB_H
#define TESTLIB_H

// Compute horizon for gridded domain
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
    float elev_ang_low_lim,
    uint8_t* mask, float hori_fill,
    float ray_org_elev);

// Compute horizon for arbitrary locations
void horizon_locations_comp(float* vert_grid, 
	int dem_dim_0, int dem_dim_1,
	float* coords,
	float* vec_norm, float* vec_north,
	float* hori_buffer,
	float* hori_dist_buffer,
	int num_loc,
	int azim_num, float dist_search,
	float hori_acc, char* ray_algorithm, char* geom_type,
    float elev_ang_low_lim,
    float* ray_org_elev,
    int hori_dist_out);

#endif

