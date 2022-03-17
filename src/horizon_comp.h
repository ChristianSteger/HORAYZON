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
    char* file_out,
    float* x_axis_val, float* y_axis_val,
    char* x_axis_name, char* y_axis_name, char* units,
    float hori_buffer_size_max,
    float elev_ang_low_lim,
    uint8_t* mask, float hori_fill,
    float ray_org_elev);

// Compute horizon for individual grid cells
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
    int hori_dist_out);

// Compute horizon for arbitrary locations
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
    float ray_org_elev,
    int hori_dist_out);

#endif

