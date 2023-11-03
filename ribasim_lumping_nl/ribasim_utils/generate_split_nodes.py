from typing import List

import geopandas as gpd
import pandas as pd


def get_split_nodes_based_on_type(
    stations: bool = False,
    pumps: bool = False,
    weirs: bool = False,
    orifices: bool = False,
    bridges: bool = False,
    culverts: bool = False,
    uniweirs: bool = False,
    list_gdfs: List[gpd.GeoDataFrame] = None,
    crs: int = 28992
):
    """get split_nodes based on weirs and/or pumps.
    list_gdfs is a list of geodataframes including 
    stations, pumps, weirs, orifices, bridges, culverts, uniweirs"""
    list_objects = [
        stations,
        pumps,
        weirs,
        orifices,
        bridges,
        culverts,
        uniweirs,
        False
    ]
    split_nodes_columns = [
        "node_no", "edge_no", "split_node_id", "geometry", "object_type", "split_type"
    ]
    split_nodes = gpd.GeoDataFrame(
        columns=split_nodes_columns,
        geometry="geometry",
        crs=crs,
    )
    for gdf_include, gdf in zip(list_objects, list_gdfs):
        if gdf_include and gdf is not None:
            gdf = gdf.rename({"structure_id": "split_node_id"}, axis=1)
            split_nodes = pd.concat([split_nodes, gdf])
    return split_nodes[split_nodes_columns]


def add_split_nodes_based_on_selection(
    stations: bool = False,
    pumps: bool = False,
    weirs: bool = False,
    orifices: bool = False,
    bridges: bool = False,
    culverts: bool = False,
    uniweirs: bool = False,
    edges: bool = False,
    list_gdfs: List[gpd.GeoDataFrame] = [],
    structures_ids_to_include: List[str] = [],
    structures_ids_to_exclude: List[str] = [],
    edge_ids_to_include: List[int] = [],
    edge_ids_to_exclude: List[int] = [],
) -> gpd.GeoDataFrame:
    """receive node id's of splitnodes
    by choosing which structures to use as splitnodes locations
    and including or excluding specific nodes as splitnode
    returns splitnodes"""
    # get split_nodes based on type
    split_nodes_structures = get_split_nodes_based_on_type(
        stations=stations,
        pumps=pumps,
        weirs=weirs,
        orifices=orifices,
        bridges=bridges,
        culverts=culverts,
        uniweirs=uniweirs,
        list_gdfs=list_gdfs
    )
    # include split_nodes with id
    all_structures = get_split_nodes_based_on_type(
        stations=True,
        pumps=True,
        weirs=True,
        orifices=True,
        bridges=True,
        culverts=True,
        uniweirs=True,
        list_gdfs=list_gdfs
    )
    structures_to_include = all_structures[
        all_structures.split_node_id.isin(structures_ids_to_include)
    ]
    split_nodes = pd.concat([
        split_nodes_structures, 
        structures_to_include
    ])
    # exclude split_nodes with id
    split_nodes = split_nodes[
        ~split_nodes.split_node_id.isin(structures_ids_to_exclude)
    ]

    # include/exclude edge centers
    if edges or len(edge_ids_to_include) > 1:
        additional_split_nodes = None
        if edges:
            additional_split_nodes = self.edges_gdf.copy()
            if len(edge_ids_to_exclude):
                additional_split_nodes = additional_split_nodes[
                    ~additional_split_nodes.edge_no.isin(edge_ids_to_exclude)
                ]
        elif len(edge_ids_to_include):
            additional_split_nodes = self.edges_gdf[
                self.edges_gdf.edge_no.isin(edge_ids_to_include)
            ]
        if additional_split_nodes is not None:
            additional_split_nodes.geometry = additional_split_nodes.geometry.apply(
                lambda g: g.centroid
            )
            additional_split_nodes["object_type"] = "edge"
            additional_split_nodes["node_no"] = -1
            additional_split_nodes = additional_split_nodes.rename(columns={
                "mesh1d_edge_x": "projection_x",
                "mesh1d_edge_y": "projection_y",
            })
            additional_split_nodes = additional_split_nodes.drop(
                ["start_node_no", "end_node_no", "basin"], axis=1, errors="ignore"
            )
            split_nodes = pd.concat([split_nodes, additional_split_nodes])
            split_nodes = split_nodes.drop_duplicates(subset="edge_no", keep="first")

    split_nodes["node_no"] = -1
    split_nodes = split_nodes.reset_index(drop=True)
    split_nodes.insert(0, "split_node", split_nodes.index + 1)
    
    # print content of all split_nodes included
    print(f"{len(split_nodes)} split locations")
    for obj_type in split_nodes.object_type.unique():
        print(f" - {obj_type}: {len(split_nodes[split_nodes['object_type']==obj_type])}")
    return split_nodes

