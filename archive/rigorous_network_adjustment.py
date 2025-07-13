#!/usr/bin/env python3
"""
Rigorous Geodetic Network Adjustment for OSS Colonial Virginia Patent Polygons

This implements a proper constrained least squares network adjustment where:
1. Individual parcels are nodes in the network
2. Seed parcels provide position constraints
3. Adjacent parcels must maintain boundary coherence
4. Non-seed parcels are free to adjust within constraints

Based on geodetic adjustment theory from:
- Ghilani & Wolf: "Adjustment Computations" 
- Caspary: "Concepts of Network and Deformation Analysis"
- Cross & Pribyl: "Adjustment of Geodetic Networks"
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from shapely.geometry import Point
from rtree import index
import time
import warnings
warnings.filterwarnings('ignore')

class RigorousNetworkAdjustment:
    """
    Rigorous geodetic network adjustment implementation
    """
    
    def __init__(self, min_boundary_length=50, weight_boundary=1.0, weight_seed=0.1):
        self.min_boundary_length = min_boundary_length  # meters
        self.weight_boundary = weight_boundary  # boundary constraint weight (m)
        self.weight_seed = weight_seed  # seed constraint weight (m)
        self.adjacency_pairs = []
        self.seed_parcels = set()
        self.parcel_centroids = {}
        
    def load_data(self):
        """Load GeoJSON and seed parcel data"""
        print("Loading data...")
        
        # Load the main GeoJSON file
        gdf = gpd.read_file('data/raw/CentralVAPatents_PLY-shp/centralva.geojson')
        print(f"Loaded {len(gdf)} parcels from GeoJSON")
        
        # Load seed parcels
        seeds_df = pd.read_csv('data/processed/manual_anchor_worksheet.csv')
        
        # Filter to valid seed parcels
        valid_seeds = seeds_df[
            (seeds_df['exclude_reason'].isna()) | 
            (seeds_df['exclude_reason'] == '')
        ].copy()
        valid_seeds = valid_seeds.dropna(subset=['centroid_lat', 'centroid_lon'])
        
        print(f"Found {len(valid_seeds)} valid seed parcels")
        
        return gdf, valid_seeds
    
    def reproject_data(self, gdf):
        """Reproject to EPSG:3857 (Web Mercator) for metric coordinates"""
        print("Reprojecting to EPSG:3857...")
        
        if gdf.crs is None:
            gdf = gdf.set_crs('EPSG:4326')
        
        gdf_proj = gdf.to_crs('EPSG:3857')
        return gdf_proj
    
    def build_adjacency_network(self, gdf):
        """Build network of adjacent parcels with shared boundaries"""
        print("Building adjacency network...")
        
        # Create spatial index
        spatial_idx = index.Index()
        for idx, geom in enumerate(gdf.geometry):
            if geom and geom.is_valid:
                spatial_idx.insert(idx, geom.bounds)
        
        # Store parcel centroids
        for idx, geom in enumerate(gdf.geometry):
            if geom and geom.is_valid:
                self.parcel_centroids[idx] = geom.centroid
        
        print(f"Processing {len(gdf)} parcels for adjacency...")
        
        for i, geom_i in enumerate(gdf.geometry):
            if i % 1000 == 0:
                print(f"Processing parcel {i}/{len(gdf)} ({100*i/len(gdf):.1f}%)")
                
            if not (geom_i and geom_i.is_valid):
                continue
            
            # Get potential neighbors
            candidates = list(spatial_idx.intersection(geom_i.bounds))
            
            for j in candidates:
                if i >= j:  # Avoid duplicates
                    continue
                    
                geom_j = gdf.geometry.iloc[j]
                if not (geom_j and geom_j.is_valid):
                    continue
                
                try:
                    intersection = geom_i.intersection(geom_j)
                    
                    # Check if they share a significant boundary
                    if hasattr(intersection, 'length') and intersection.length > self.min_boundary_length:
                        # Get boundary midpoint
                        if intersection.geom_type == 'LineString':
                            coords = list(intersection.coords)
                            if len(coords) >= 2:
                                mid_x = (coords[0][0] + coords[-1][0]) / 2
                                mid_y = (coords[0][1] + coords[-1][1]) / 2
                            else:
                                continue
                        elif intersection.geom_type == 'MultiLineString':
                            longest_line = max(intersection.geoms, key=lambda x: x.length)
                            coords = list(longest_line.coords)
                            if len(coords) >= 2:
                                mid_x = (coords[0][0] + coords[-1][0]) / 2
                                mid_y = (coords[0][1] + coords[-1][1]) / 2
                            else:
                                continue
                        else:
                            continue
                        
                        self.adjacency_pairs.append({
                            'parcel_i': i,
                            'parcel_j': j,
                            'boundary_mid_x': mid_x,
                            'boundary_mid_y': mid_y,
                            'shared_length': intersection.length
                        })
                        
                except Exception:
                    continue
        
        print(f"Found {len(self.adjacency_pairs)} adjacency relationships")
    
    def identify_seed_parcels(self, gdf, seeds_df):
        """Identify which parcels are seed parcels with known positions"""
        print("Identifying seed parcels...")
        
        # Find ID field
        id_field = None
        for field in ['OBJECTID', 'oss_id', 'grant_id', 'id']:
            if field in gdf.columns:
                id_field = field
                break
        
        if not id_field:
            print("Warning: No suitable ID field found")
            return
        
        seed_ids = set(seeds_df['oss_id'].dropna())
        print(f"Seed IDs to match: {len(seed_ids)}")
        
        for idx, row in gdf.iterrows():
            if row.get(id_field) in seed_ids:
                self.seed_parcels.add(idx)
        
        print(f"Matched {len(self.seed_parcels)} seed parcels")
        print(f"Sample seed parcel indices: {sorted(list(self.seed_parcels))[:10]}")
    
    def setup_adjustment_system(self, gdf, seeds_df):
        """Set up the constrained least squares system"""
        print("Setting up adjustment system...")
        
        # Each parcel has 2 unknowns: dx, dy (translation from initial position)
        n_parcels = len(gdf)
        n_params = 2 * n_parcels
        
        # But seed parcels are constrained (dx=0, dy=0)
        free_params = []
        param_map = {}
        
        param_idx = 0
        for i in range(n_parcels):
            if i not in self.seed_parcels:
                param_map[i] = slice(param_idx, param_idx + 2)
                free_params.extend([2*i, 2*i + 1])
                param_idx += 2
        
        n_free_params = param_idx
        print(f"Free parameters: {n_free_params} (for {len(param_map)} non-seed parcels)")
        print(f"Constrained parameters: {2 * len(self.seed_parcels)} (for {len(self.seed_parcels)} seed parcels)")
        
        # Create observation equations
        observations = []
        
        # 1. Boundary coherence constraints
        for adj in self.adjacency_pairs:
            i, j = adj['parcel_i'], adj['parcel_j']
            
            # Weight by shared boundary length
            weight = min(adj['shared_length'] / 100.0, 10.0)  # max weight = 10
            
            observations.append({
                'type': 'boundary',
                'parcel_i': i,
                'parcel_j': j,
                'target_x': 0,  # difference should be 0
                'target_y': 0,
                'weight': weight
            })
        
        # 2. Seed position constraints (if we want to allow small adjustments)
        # For rigorous adjustment, we could allow seeds to move slightly within uncertainty
        
        print(f"Created {len(observations)} constraint equations")
        
        return param_map, observations, n_free_params
    
    def residual_function(self, params, param_map, observations, gdf):
        """Compute residuals for least squares optimization"""
        residuals = []
        
        for obs in observations:
            if obs['type'] == 'boundary':
                i, j = obs['parcel_i'], obs['parcel_j']
                
                # Get adjustments for parcel i
                if i in self.seed_parcels:
                    dx_i, dy_i = 0, 0
                elif i in param_map:
                    p_slice = param_map[i]
                    dx_i, dy_i = params[p_slice]
                else:
                    dx_i, dy_i = 0, 0
                
                # Get adjustments for parcel j
                if j in self.seed_parcels:
                    dx_j, dy_j = 0, 0
                elif j in param_map:
                    p_slice = param_map[j]
                    dx_j, dy_j = params[p_slice]
                else:
                    dx_j, dy_j = 0, 0
                
                # Current positions
                centroid_i = self.parcel_centroids[i]
                centroid_j = self.parcel_centroids[j]
                
                # Adjusted positions
                x_i_adj = centroid_i.x + dx_i
                y_i_adj = centroid_i.y + dy_i
                x_j_adj = centroid_j.x + dx_j
                y_j_adj = centroid_j.y + dy_j
                
                # The constraint is that both parcels should be at the boundary midpoint
                # So the difference between their adjusted positions and the boundary should be equal
                boundary_x = obs.get('boundary_mid_x', (centroid_i.x + centroid_j.x) / 2)
                boundary_y = obs.get('boundary_mid_y', (centroid_i.y + centroid_j.y) / 2)
                
                # Residuals: difference in how far each parcel is from boundary
                residual_x = (x_i_adj - boundary_x) - (x_j_adj - boundary_x)
                residual_y = (y_i_adj - boundary_y) - (y_j_adj - boundary_y)
                
                residuals.extend([
                    residual_x * obs['weight'],
                    residual_y * obs['weight']
                ])
        
        return np.array(residuals)
    
    def solve_adjustment(self, param_map, observations, gdf, n_free_params):
        """Solve the rigorous network adjustment"""
        print(f"Solving network adjustment: {n_free_params} parameters, {len(observations)} constraints")
        
        if n_free_params == 0:
            print("No free parameters - all parcels are constrained")
            return np.array([])
        
        # Initial parameter values (no adjustments)
        x0 = np.zeros(n_free_params)
        
        print("Starting least squares optimization...")
        
        # Solve using robust optimization
        result = least_squares(
            self.residual_function,
            x0,
            args=(param_map, observations, gdf),
            verbose=2,
            ftol=1e-8,
            xtol=1e-8,
            max_nfev=1000
        )
        
        if result.success:
            print("Network adjustment converged successfully")
            print(f"Final cost: {result.cost:.2e}")
            print(f"Optimality: {result.optimality:.2e}")
        else:
            print(f"Warning: Network adjustment convergence issues: {result.message}")
        
        return result.x
    
    def compute_parcel_shifts(self, gdf, param_map, solution):
        """Compute shift magnitude for each parcel"""
        print("Computing parcel shifts...")
        
        shifts = {}
        
        for i in range(len(gdf)):
            if i in self.seed_parcels:
                shifts[i] = 0.0  # Seed parcels don't move
            elif i in param_map and len(solution) > 0:
                p_slice = param_map[i]
                dx, dy = solution[p_slice]
                shift_km = np.sqrt(dx**2 + dy**2) / 1000  # Convert m to km
                shifts[i] = shift_km
            else:
                shifts[i] = np.nan  # Isolated parcels
        
        return shifts
    
    def bootstrap_confidence(self, gdf, seeds_df, n_bootstrap=500):
        """Bootstrap analysis for confidence intervals"""
        print(f"Running bootstrap analysis ({n_bootstrap} iterations)...")
        
        percentile_90_values = []
        
        for i in range(n_bootstrap):
            if i % 50 == 0:
                print(f"Bootstrap iteration {i}/{n_bootstrap} ({100*i/n_bootstrap:.1f}%)")
            
            try:
                # Resample seed parcels
                bootstrap_seeds = seeds_df.sample(n=len(seeds_df), replace=True)
                
                # Re-identify seed parcels
                original_seeds = self.seed_parcels.copy()
                self.seed_parcels.clear()
                self.identify_seed_parcels(gdf, bootstrap_seeds)
                
                # Quick solve
                param_map, observations, n_free_params = self.setup_adjustment_system(gdf, bootstrap_seeds)
                
                if n_free_params > 0:
                    solution = self.solve_adjustment(param_map, observations, gdf, n_free_params)
                    shifts = self.compute_parcel_shifts(gdf, param_map, solution)
                    
                    valid_shifts = [s for s in shifts.values() if not np.isnan(s) and s > 0]
                    if len(valid_shifts) > 0:
                        p90 = np.percentile(valid_shifts, 90)
                        percentile_90_values.append(p90)
                
                # Restore original seeds
                self.seed_parcels = original_seeds
                
            except Exception as e:
                continue
        
        if len(percentile_90_values) > 0:
            ci_low = np.percentile(percentile_90_values, 2.5)
            ci_high = np.percentile(percentile_90_values, 97.5)
            return ci_low, ci_high
        else:
            return np.nan, np.nan
    
    def create_output_files(self, gdf, shifts):
        """Create output files and plots"""
        print("Creating output files...")
        
        # Create results dataframe
        output_df = pd.DataFrame({
            'grant_id': gdf['OBJECTID'] if 'OBJECTID' in gdf.columns else gdf.index,
            'is_seed': [i in self.seed_parcels for i in range(len(gdf))],
            'shift_km': [shifts.get(i, np.nan) for i in range(len(gdf))]
        })
        
        output_df.to_csv('results/rigorous_network_adjustment_shifts.csv', index=False)
        
        # Create histogram
        valid_shifts = output_df['shift_km'].dropna()
        valid_shifts = valid_shifts[valid_shifts > 0]  # Remove zeros
        
        if len(valid_shifts) > 0:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.hist(valid_shifts, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Shift Magnitude (km)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Parcel Shift Magnitudes (Rigorous Network Adjustment)')
            plt.grid(True, alpha=0.3)
            
            # Add percentile lines
            if len(valid_shifts) > 0:
                p50 = np.percentile(valid_shifts, 50)
                p90 = np.percentile(valid_shifts, 90)
                p95 = np.percentile(valid_shifts, 95)
                
                plt.axvline(p50, color='red', linestyle='--', label=f'50th percentile: {p50:.3f} km')
                plt.axvline(p90, color='orange', linestyle='--', label=f'90th percentile: {p90:.3f} km')
                plt.axvline(p95, color='purple', linestyle='--', label=f'95th percentile: {p95:.3f} km')
                plt.legend()
            
            # Log scale plot
            plt.subplot(2, 1, 2)
            plt.hist(valid_shifts, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Shift Magnitude (km)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Parcel Shift Magnitudes (Log Scale)')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('plots/rigorous_shift_histogram.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return output_df
    
    def run_full_analysis(self):
        """Run the complete rigorous network adjustment analysis"""
        start_time = time.time()
        
        print("="*70)
        print("RIGOROUS GEODETIC NETWORK ADJUSTMENT ANALYSIS")
        print("="*70)
        
        # Load and prepare data
        gdf, seeds_df = self.load_data()
        gdf = self.reproject_data(gdf)
        
        # Build network
        self.build_adjacency_network(gdf)
        self.identify_seed_parcels(gdf, seeds_df)
        
        # Set up and solve adjustment
        param_map, observations, n_free_params = self.setup_adjustment_system(gdf, seeds_df)
        solution = self.solve_adjustment(param_map, observations, gdf, n_free_params)
        
        # Compute results
        shifts = self.compute_parcel_shifts(gdf, param_map, solution)
        output_df = self.create_output_files(gdf, shifts)
        
        # Statistics
        valid_shifts = output_df['shift_km'].dropna()
        valid_shifts = valid_shifts[valid_shifts > 0]
        
        if len(valid_shifts) > 0:
            p50 = np.percentile(valid_shifts, 50)
            p90 = np.percentile(valid_shifts, 90)
            p95 = np.percentile(valid_shifts, 95)
            
            # Bootstrap CI
            print("\nComputing bootstrap confidence intervals...")
            ci_low, ci_high = self.bootstrap_confidence(gdf, seeds_df, n_bootstrap=100)  # Reduced for speed
            
            # Print results
            print("\n" + "="*50)
            print("=== RIGOROUS NETWORK ADJUSTMENT SUMMARY ===")
            print(f"Parcels: {len(gdf)}")
            print(f"Seed parcels: {len(self.seed_parcels)}")
            print(f"Free parcels: {len(gdf) - len(self.seed_parcels)}")
            print(f"Parcels with non-zero shifts: {len(valid_shifts)}")
            print(f"Median shift: {p50:.3f} km")
            print(f"90th percentile: {p90:.3f} km")
            print(f"95th percentile: {p95:.3f} km")
            if not np.isnan(ci_low) and not np.isnan(ci_high):
                print(f"95% CI (90th pct): [{ci_low:.3f}, {ci_high:.3f}] km")
            else:
                print("95% CI (90th pct): [Unable to compute]")
            print("="*50)
            
        else:
            print("No non-zero shifts found - check seed parcel distribution")
        
        # Runtime
        total_time = time.time() - start_time
        print(f"\nTotal runtime: {total_time/60:.1f} minutes")
        print("="*70)
        
        print(f"\nResults saved to: results/rigorous_network_adjustment_shifts.csv")
        print(f"Plot saved to: plots/rigorous_shift_histogram.png")

def main():
    adjustment = RigorousNetworkAdjustment()
    adjustment.run_full_analysis()

if __name__ == "__main__":
    main() 