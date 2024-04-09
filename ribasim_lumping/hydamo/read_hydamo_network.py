from pathlib import Path
from ..utils.general_functions import read_geom_file, generate_nodes_from_edges, split_edges_by_dx
from shapely.geometry import LineString, Point
from typing import Tuple
import geopandas as gpd
import pandas as pd
import fiona


def add_hydamo_basis_network(
    hydamo_network_file: Path = 'network.gpkg',
    hydamo_split_network_dx: float = None,
    crs: int = 28992,
):
    # ) -> Tuple[
    #     gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, 
    #     gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, 
    #     gpd.GeoDataFrame, gpd.GeoDataFrame
    # ]:
    """
    Load network data from HyDAMO files

    Args:
        hydamo_network_file (Path):         Path to file containing network geometries (hydroobjects)
        hydamo_network_gpkg_layer (str):    Layer name in geopackage. Needed when file is a geopackage
        crs (int):                          (optional) CRS EPSG code. Default 28992 (RD New)
    
    Returns:
        Tuple containing GeoDataFrames with branches, edges nodes
    """

    print('Reading network from HyDAMO files...')
    branches_gdf = read_geom_file(
        filepath=hydamo_network_file, 
        layer_name="hydroobject", 
        crs=crs, 
        remove_z_dim=True
    ).rename(columns={'code': 'branch_id'})[['branch_id', 'geometry']]
    branches_gdf, network_nodes_gdf = generate_nodes_from_edges(branches_gdf)

    # Split up hydamo edges with given distance as approximate length of new edges
    if hydamo_split_network_dx is None:
        edges_gdf = branches_gdf.copy().rename(columns={"branch_id": "edge_id"})
    else:
        edges_gdf = split_edges_by_dx(
            edges=branches_gdf, 
            dx=hydamo_split_network_dx,
        )
    edges_gdf, nodes_gdf = generate_nodes_from_edges(edges_gdf)
    edges_gdf.index.name = "index"

    # Read structures and data according to hydamo-format
    weirs_gdf, culverts_gdf, pumps_gdf, sluices_gdf, closers_gdf = None, None, None, None, None
    
    pumps_gdf = read_geom_file(
        filepath=hydamo_network_file,
        layer_name="gemaal",
        crs=crs
    )
    sluices_gdf = read_geom_file(
        filepath=hydamo_network_file,
        layer_name="sluis",
        crs=crs
    )
    weirs_gdf  = read_geom_file(
        filepath=hydamo_network_file,
        layer_name="stuw",
        crs=crs
    )
    culverts_gdf  = read_geom_file(
        filepath=hydamo_network_file,
        layer_name="duikersifonhevel",
        crs=crs
    )
    closers_gdf = read_geom_file(
        filepath=hydamo_network_file,
        layer_name="afsluitmiddel",
        crs=crs
    )
    if "pomp" in fiona.listlayers(hydamo_network_file):
        pumps_df = gpd.read_file(hydamo_network_file, layer="pomp")
    else:
        pumps_df = None
    
    # set column names to lowercase and return
    results = [branches_gdf, network_nodes_gdf, edges_gdf, nodes_gdf, weirs_gdf, culverts_gdf, pumps_gdf, pumps_df, sluices_gdf, closers_gdf]
    results = [x.rename(columns={c: c.lower() for c in x.columns}) if x is not None else None 
               for x in results]
    return results

