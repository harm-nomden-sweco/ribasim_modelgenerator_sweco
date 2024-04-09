"""
Generate overview of downstream objects per object in each basin
Use WS Dommel for testing
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import ribasim
from ribasim_lumping import create_ribasim_lumping_network
import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
warnings.simplefilter("ignore")
pd.options.mode.chained_assignment = None
# change workdir
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Settings #
# define network name, base dir
network_name               = "DeDommel"
source_type                = "hydamo"
base_dir                   = Path("c:/Projecten/51017989_TKI-NHI-oppervlaktewatermodule_-_Modelgenerator/Ribasim modeldata/DeDommel/verwerkt")
# areas (discharge units: afwaterende eenheden)
areas_file                 = Path(base_dir, "3_input", "areas.gpkg")
areas_gpkg_layer           = "areas"
areas_code_column          = "code"
# drainage areas (afvoergebieden)
drainage_areas_file        = Path(base_dir, "3_input", "areas.gpkg")
drainage_areas_gpkg_layer  = "drainage_areas"
# HyDAMO data
hydamo_network_file        = Path(base_dir, "4_ribasim", "hydamo.gpkg")
hydamo_split_network_dx    = 100.0  # split up hydamo hydroobjects in sections of approximate 100 m. Use None to don't split
# Ribasim input
ribasim_input_boundary_file              = Path(base_dir, "3_input", "ribasim_input.gpkg")
ribasim_input_boundary_gpkg_layer        = 'boundaries'
ribasim_input_split_nodes_file           = Path(base_dir, "3_input", "ribasim_input.gpkg")
ribasim_input_split_nodes_gpkg_layer     = "split_nodes"
# directory results
results_dir                = Path("results")
simulation_code            = "."


# option to skip recomputing network everytime
force_recompute_network = False
import pickle
ribasim_network_pickle = results_dir / 'ribasim_network.pkl'
if force_recompute_network or (not ribasim_network_pickle.exists()):
    # Start #
    # Create networkanalysis
    network = create_ribasim_lumping_network(base_dir=base_dir, name=network_name, results_dir=results_dir, crs=28992)

    # Load areas
    network.add_areas_from_file(
        areas_file_path=areas_file,
        layer_name=areas_gpkg_layer,
        areas_code_column=areas_code_column, 
    )
    network.add_drainage_areas_from_file(
        drainage_areas_file_path=drainage_areas_file,
        layer_name=drainage_areas_gpkg_layer,
    )

    # Read HyDAMO network data
    network.add_basis_network(
        source_type=source_type,
        hydamo_network_file=hydamo_network_file,
        hydamo_write_results_to_gpkg=False,
        hydamo_split_network_dx=hydamo_split_network_dx,
    )

    network.add_split_nodes_from_file(
        split_nodes_file_path=ribasim_input_split_nodes_file,
        layer_name=ribasim_input_split_nodes_gpkg_layer,
        crs=28992
    )

    network.add_boundaries_from_file(
        boundary_file_path=ribasim_input_boundary_file,
        layer_name=ribasim_input_boundary_gpkg_layer,
        crs=28992
    )

    split_node_type_conversion = dict(
        stuw="TabulatedRatingCurve",
        brug="TabulatedRatingCurve",
        gemaal="Pump",
        duikersifonhevel="TabulatedRatingCurve",
        openwater="ManningResistance",
        onderdoorlaat="Outlet",
        boundary_connection="ManningResistance"
    )
    # specify translation for specific split_nodes to ribasim-nodes
    split_node_id_conversion = dict(
        #kdu_DR80740070='ManningResistance',
    )

    # generate Ribasim network
    network.generate_ribasim_lumping_network(
        simulation_code=simulation_code,
        remove_isolated_basins=True,
        split_node_type_conversion=split_node_type_conversion,
        split_node_id_conversion=split_node_id_conversion,
    )
    with open(str(ribasim_network_pickle), 'wb') as f:
        pickle.dump(network, f)
else:
    network = pickle.load(open(str(ribasim_network_pickle), 'rb'))



## START OF DOWNSTREAM OBJECTS STUFF ##
from ribasim_lumping.utils.general_functions import split_edges_by_split_nodes
from ribasim_lumping.ribasim_network_generator.generate_ribasim_network import create_graph_based_on_nodes_edges
import networkx as nx

def get_direct_downstream_structures_for_nodes(
        structures: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,    
        nodes: gpd.GeoDataFrame,
        split_nodes: gpd.GeoDataFrame = None,
    ) -> gpd.GeoDataFrame:
    """
    Get structure directly downstream of nodes within the same basin based on shortest path. 
    Use this function to get the node no and the shortest path length to that node for weir and/or pump structures.
    
    Basin code is retrieved from edges.
    The edges will be split on the structure locations before determining the most direct downstream structure.
    Supply split nodes optionally to update structure point geometries with snapped split nodes to avoid duplication
    Parameters
    ----------
    structures (gpd.GeoDataFrame):               
        GeoDataFrame containing all structure geometries. Should include 'code' column
    edges (gpd.GeoDataFrame):
        GeoDataFrame containing edges. Should include a 'basin' column with the associated basin code
    nodes (gpd.GeoDataFrame):
        GeoDataFrame containing nodes (that are compliant with edges). These will be returned as a result with info of downstream structure
    split_nodes (gpd.GeoDataFrame):
        (optional) snapped split nodes to edges/nodes. Should include 'split_node_id' column which contains code that is used in structures geodataframe
    
    Returns
    -------
    GeoDataFrames with nodes with 2 columns added containing direct downstream structure node no and the path length from node to that structure
    """

    structures, edges, nodes_orig = structures.copy(), edges.copy(), nodes.copy()
    # update structure point locations with already snapped points of split nodes
    if split_nodes is not None:
        structures.geometry = [split_nodes.loc[split_nodes['split_node_id'] == c]['geometry'].values[0]
                        if c in split_nodes['split_node_id'] 
                        else g 
                        for c, g in zip(structures['code'], structures.geometry)]
    # split edges on structure locations. this will also regenerate the edges and nodes (but preserves basin in edges)
    structures, edges, nodes = split_edges_by_split_nodes(structures, edges=edges)
    structures.columns = [c.lower() for c in structures.columns]
    # initialize new columns
    nodes['downstream_structure_code'] = np.nan
    nodes['downstream_structure_node_no'] = np.nan
    nodes['downstream_structure_path_length'] = np.nan
    # go over each basin individually to restrict the network size
    basin_nrs = np.sort(edges['basin'].unique())
    basin_nrs = basin_nrs[basin_nrs >= 1]
    for basin_nr in basin_nrs:
        # select nodes, edges and structures within basin
        _edges = edges.loc[edges['basin'] == basin_nr].copy()
        _nodes = nodes.loc[np.isin(nodes['node_no'], np.unique(np.concatenate([_edges['from_node'], _edges['to_node']])))].copy()
        _structures = structures.loc[np.isin(structures['node_no'], _nodes['node_no'])].copy()

        if _structures.empty:
            continue  # skip if no structures are within basin

        # create graph
        _graph = create_graph_based_on_nodes_edges(_nodes, _edges, add_edge_length_as_weight=True)

        # get shortest paths between nodes (keeping in mind direction of edges in graph) and the lengths of those paths
        paths = dict(nx.all_pairs_dijkstra_path(_graph))
        path_lengths = dict(nx.all_pairs_dijkstra_path_length(_graph))

        # select for each node the direct downstream structure (the one with the shortest path to node)
        structure_nodes = {k: v for k, v in zip(_structures['node_no'].values, _structures['code'].values)}
        store = {}
        for k, vps in paths.items():
            length_to_structures = {vp: path_lengths[k][vp] for vp in vps.keys() if vp in structure_nodes.keys()}

            # select the structure and store together with structure code and distance from node to that structure
            if k in structure_nodes.keys():
                store[k] = (k, structure_nodes[k], 0)  # node is a structure
            elif len(length_to_structures.keys()) == 0:
                store[k] = None  # there is no downstream structure within basin for node
            else:
                n = list(length_to_structures.keys())[np.argmin(length_to_structures.values())]
                store[k] = (n, structure_nodes[n], length_to_structures[n])

        # save in nodes geodataframe
        for k, v in store.items():
            if v is not None:
                nodes.loc[nodes['node_no'] == k, 'downstream_structure_node_no'] = v[0]
                nodes.loc[nodes['node_no'] == k, 'downstream_structure_code'] = v[1]
                nodes.loc[nodes['node_no'] == k, 'downstream_structure_path_length'] = v[2]

    # update original nodes geodataframe with downstream structure info using spatial join with max 5 cm buffer
    nodes_updated = nodes_orig.copy()
    nodes_updated = nodes_updated.sjoin_nearest(
        nodes[['downstream_structure_node_no', 'downstream_structure_code', 'downstream_structure_path_length', 'geometry']], 
        how='left', 
        max_distance=0.05
    )

    return nodes_updated














## END OF DOWNSTREAM OBJECTS STUFF ##




print('done')
