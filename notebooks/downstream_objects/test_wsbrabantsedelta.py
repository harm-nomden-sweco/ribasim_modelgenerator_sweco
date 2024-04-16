"""
Generate overview of downstream objects per object in each basin
Use WS Brabantse Delta for testing
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
network_name               = "BrabantseDelta"
source_type                = "hydamo"
base_dir                   = Path("c:/Projecten/51017989_TKI-NHI-oppervlaktewatermodule_-_Modelgenerator/Ribasim modeldata/BrabantseDelta/verwerkt")
# areas (discharge units: afwaterende eenheden)
areas_file                 = Path(base_dir, "3_input", "areas.gpkg")
areas_gpkg_layer           = "areas"
areas_code_column          = "code"
# drainage areas (afvoergebieden)
drainage_areas_file        = Path(base_dir, "3_input", "areas.gpkg")
drainage_areas_gpkg_layer  = "drainage_areas"
# HyDAMO data
hydamo_network_file        = Path(base_dir, "4_ribasim", "hydamo.gpkg")
hydamo_split_network_dx    = 100.0  # split up hydamo hydroobjects in sections of approximate 25 m. Use None to don't split
# Ribasim input
ribasim_input_boundary_file              = Path(base_dir, "3_input", "ribasim_input.gpkg")
ribasim_input_boundary_gpkg_layer        = 'boundaries'
ribasim_input_split_nodes_file           = Path(base_dir, "3_input", "ribasim_input.gpkg")
ribasim_input_split_nodes_gpkg_layer     = "split_nodes"
# directory results
results_dir                = Path("results_brabantsedelta")
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
from ribasim_lumping.utils.general_functions import split_edges_by_split_nodes, snap_points_to_nodes_and_edges
from ribasim_lumping.ribasim_network_generator.generate_ribasim_network import create_graph_based_on_nodes_edges
import networkx as nx
from typing import Union
from tqdm import tqdm

def get_downstream_objects_on_path_for_nodes(
        self = network,
        edges: gpd.GeoDataFrame = None,  
        nodes: gpd.GeoDataFrame = None,
        structures: gpd.GeoDataFrame = None,
        split_nodes: gpd.GeoDataFrame = None,
        cutoff: Union[float, int] = 30000,
    ) -> gpd.GeoDataFrame:
    """
    Get structure directly downstream of nodes within the same basin based on shortest path. 
    Use this function to get the node no and the shortest path length to that node for weir and/or pump structures.
    
    Basin code is retrieved from edges.
    The edges will be split on the structure locations before determining the most direct downstream structure.
    Supply split nodes optionally to update structure point geometries with snapped split nodes to avoid duplication
    Parameters
    ----------
    edges (gpd.GeoDataFrame):
        GeoDataFrame containing edges. Should include a 'basin' column with the associated basin code. If not provided, it will use all edges stored within
        the RibasimLumpingNetwork object
    nodes (gpd.GeoDataFrame):
        GeoDataFrame containing nodes (that are compliant with edges). These will be returned as a result with info of downstream structure. If not provided, 
        it will use all nodes stored within the RibasimLumpingNetwork object
    structures (gpd.GeoDataFrame):               
        (optional) GeoDataFrame containing all structure geometries. Should include 'code' column. If not provided, it will use all structures stored within
        the RibasimLumpingNetwork object. Only weir and pump structures will be considered.
    split_nodes (gpd.GeoDataFrame):
        (optional) snapped split nodes to edges/nodes. Should include 'split_node_id' column which contains code that is used in structures geodataframe. If not 
        provided, it will use all structures stored within the RibasimLumpingNetwork object
    cutoff (float, int):
        (optional) cutoff distance (in m) when search for paths between nodes. Use this option to restrict the amount of possible paths between the nodes to 
        reduce memory load. Defaults to 30000 m (=30 km).
    
    Returns
    -------
    GeoDataFrames with nodes with 2 columns added containing direct downstream structure node no and the path length from node to that structure
    """

    print('Determining downstream structures for each node')
    if nodes is None:
        nodes = self.nodes_gdf
    if edges is None:
        edges = self.edges_gdf
    if structures is None:
        structures = self.get_all_structures(line_as_point=True)
    if split_nodes is None:
        split_nodes = self.split_nodes
    nodes_orig = nodes.copy()  # make backup of nodes gdf to join the new information later back to
    structures, edges, nodes, split_nodes = structures.copy(), edges.copy(), nodes.copy(), split_nodes.copy()
    # make sure nodes and edges are unique
    nodes, edges = nodes.loc[~nodes['node_no'].duplicated()], edges.loc[~edges['edge_no'].duplicated()]
    # only consider weir and pump structures
    structures = structures.loc[[s in ['weir', 'stuw', 'pump', 'gemaal'] for s in structures['object_type']]]
    # snap structures to nodes and edges
    structures = snap_points_to_nodes_and_edges(
        points=structures,
        edges=edges,
        nodes=nodes,
        edges_bufdist=5,
        nodes_bufdist=0.5,
    )
    print('\033[1A', end='\x1b[2K')  # to remove previous printed line from function
    # update structure point locations with already snapped points of split nodes
    if split_nodes is not None:
        structures.geometry = [split_nodes.loc[split_nodes['split_node_id'] == c]['geometry'].values[0]
                        if c in split_nodes['split_node_id'] 
                        else g 
                        for c, g in zip(structures['code'], structures.geometry)]
    # split edges on structure locations. this will also regenerate the edges and nodes (but preserves basin in edges)
    structures, edges, nodes = split_edges_by_split_nodes(structures, edges=edges)
    for i in range(3):
        print('\033[1A', end='\x1b[2K')  # to remove previous printed lines from function
    structures.columns = [c.lower() for c in structures.columns]
    # initialize new columns
    nodes['downstream_structures'] = None
    nodes['downstream_structures'] = nodes['downstream_structures'].astype(object)
        
    # create graph
    graph = create_graph_based_on_nodes_edges(nodes, edges, add_edge_length_as_weight=True)

    # get shortest paths between nodes (keeping in mind direction of edges in graph) and the lengths of those paths
    print(f' - get all paths with lengths between nodes with cutoff={cutoff:.0f}m')
    paths = dict(nx.all_pairs_dijkstra(graph, cutoff=cutoff))

    # select for each node the direct downstream structure (the one with the shortest path to node)
    print(' - filter paths based on direct downstream structures')
    structure_nodes = {k: v for k, v in zip(structures['node_no'].values, structures['code'].values)}
    store = {}
    for k, (vps, _) in paths.items():
        # in paths the distance between paths is stored on index 1 and the path descriptions on index 0
        paths_and_length_to_structures = {vp: (paths[k][1][vp], paths[k][0][vp]) for vp in vps.keys() 
                                            if (vp in structure_nodes.keys())}  # only look at the structures as look from the point of view from the selected node
        
        # exclude current selected structure from lengths
        paths_and_length_to_structures = {_k: _v for _k, _v in paths_and_length_to_structures.items() if _k != k}

        # exclude structures that are located downstream of another structure in the list (check if == 1 because each path contains also the structure of interest)
        length_to_structures = {_k: _v[1] for _k, _v in paths_and_length_to_structures.items()
                                if len([__k for __k in paths_and_length_to_structures.keys() if __k in _v[0]]) == 1}

        # select the structure(s) and store together with structure code(s) and distance(s) from node to that structure
        # note that multiple structures can be selected if there are multiple paths from node to a structure (for example due to bifurcation downstream)
        if len(length_to_structures.keys()) == 0:
            store[k] = None  # there is no downstream structure within basin for node
        else:
            n = list(length_to_structures.keys())
            if None in [structure_nodes[_n] for _n in n]:
                print([structure_nodes[_n] for _n in n])
            store[k] = (n, [structure_nodes[_n] for _n in n], [length_to_structures[_n] for _n in n])

    # save in nodes geodataframe
    for k, v in store.items():
        if v is not None:
            ix = nodes.loc[nodes['node_no'] == k].index.values[0]
            nodes.at[ix, 'downstream_structures'] = pd.DataFrame(data={'node_no': v[0], 'structure_code': v[1], 'path_length_to_structure': v[2]})

    # update original nodes geodataframe with downstream structure info using spatial join with max 5 cm buffer
    print(' - update nodes with downstream structures information')
    nodes_updated = nodes_orig.copy()
    nodes_updated = nodes_updated.sjoin_nearest(
        nodes[['downstream_structures', 'geometry']], 
        how='left', 
        max_distance=0.05
    )

    return nodes_updated

def get_all_structures(
        self = network,
        line_as_point: bool = False
    ) -> gpd.GeoDataFrame:
    """Returns all structures in one GeoDataFrame"""
    structures = pd.DataFrame()
    for s in ['culverts', 'bridges', 'orifices', 'pumps', 'sluices', 'uniweirs', 'weirs']:
        structures = pd.concat([structures, eval(f"network.{s}_gdf")])
    structures.reset_index(drop=True, inplace=True)
    if line_as_point:
        structures['geometry'] = [g.interpolate(0.5, normalized=True) if 'LINESTRING' in str(g) else g
                                  for g in structures['geometry']]
    structures = fill_code_column_structures(structures)
    return gpd.GeoDataFrame(structures, geometry='geometry', crs=network.weirs_gdf.crs)

def fill_code_column_structures(
        structures: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
    # fill code with naam or globalid (if naam not present) if code is empty
    crs = structures.crs
    count = 1
    for i in structures.index.values:
        if pd.isnull(structures.at[i, 'code']):
            if 'naam' in structures.columns:
                structures.at[i, 'code'] = structures.at[i, 'naam']
        if pd.isnull(structures.at[i, 'code']):
            structures.at[i, 'code'] = structures.at[i, 'globalid']
        if pd.isnull(structures.at[i, 'code']):
            structures.at[i, 'code'] = f'geen_code_naam_of_globalid_bekend_{count}'
            count += 1
    return gpd.GeoDataFrame(structures, geometry='geometry', crs=crs)



### THIS PART IS ONLY NEEDED TO ADD FUNCTIONS TO ALREADY LOADED network OBJECT ###
network_backup = pickle.load(open(str(ribasim_network_pickle), 'rb'))
network.__dict__['get_downstream_objects_on_path_for_nodes'] = get_downstream_objects_on_path_for_nodes
network.__dict__['get_all_structures'] = get_all_structures
#########




# get downstream objects. information will be added to nodes gdf so also save the results to nodes gdf
network.nodes_gdf = network.get_downstream_objects_on_path_for_nodes(cutoff=15000)



### THIS PART IS USED TO TRANSFORM INFORMATION IN nodes_updates TO EXPORTABLE GEOPACKAGE SO
### WE CAN CHECK THE RESULTS IN GIS ###
for i, row in network.nodes_gdf.iterrows():
    if row['downstream_structures'] is not None:
        new = [f'({c},{l:.2f})' for c, l in zip(row['downstream_structures']['structure_code'], row['downstream_structures']['path_length_to_structure'])]
        network.nodes_gdf.at[i, 'downstream_structures'] = ', '.join(new)

network.export_to_geopackage(simulation_code=simulation_code)






## END OF DOWNSTREAM OBJECTS STUFF ##




print('done')
