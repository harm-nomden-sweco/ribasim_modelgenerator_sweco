from typing import List, Dict, Any, Union, Optional, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, GeometryCollection


def create_graph_based_on_nodes_edges(
        nodes: gpd.GeoDataFrame, 
        edges: gpd.GeoDataFrame
    ) -> nx.DiGraph:
    """
    create networkx graph based on geographic nodes and edges.
    TODO: maybe a faster implementation possible
    """

    graph = nx.DiGraph()
    if nodes is not None:
        for i, node in nodes.iterrows():
            graph.add_node(node.node_no, pos=(node.geometry.x, node.geometry.y))
    if edges is not None:
        for i, edge in edges.iterrows():
            graph.add_edge(edge.from_node, edge.to_node)
    print(
        f" - create network graph from nodes ({len(nodes)}x) and edges ({len(edges)}x)"
    )
    return graph


def split_graph_based_on_split_nodes(
        graph: nx.DiGraph, 
        split_nodes: gpd.GeoDataFrame, 
        edges: gpd.GeoDataFrame
    ) -> Tuple[nx.DiGraph, gpd.GeoDataFrame]:
    """
    Split networkx graph at split_edge or split_node. It removes the original edges(s)/node(s) which are the same as split_edge and
    split_node and inserts new edges and nodes such that the graph becomes disconnected at the split point. After this edges don't
    connect to 1 node (at split point) but each end in each own new node. Because of this removing and adding edges and nodes in the
    graph, these new nodes no in graph are added to split_nodes gdf and also returned as result of this function.
    """

    split_nodes = split_nodes.copy()  # copy to make sure gdf variable is not linked
    split_nodes['graph_node_no'] = pd.Series([-1] * len(split_nodes), index=split_nodes.index, dtype=object)  # force dtype object to be able to insert tuples

    # split on edge: delete edge, create 2 nodes, create 2 edges
    # if all edge no in split nodes gdf are -1, than no splitting of edges are done
    # TODO: although edge stuff below works, it is actually better to split network at split nodes at earlier stage
    #       this will result in all edge no being -1 and only values for node no. so maybe check on that all edge_no
    #       values need to be -1 should be better.
    if "edge_no" not in split_nodes.columns:
        split_nodes["edge_no"] = -1
    split_nodes_edges = split_nodes[split_nodes.edge_no != -1].copy()

    split_edges = edges[
        edges.edge_no.isin(split_nodes_edges.edge_no.values)
    ].copy()
    # assert len(split_nodes_edges) == len(split_edges)
    split_edges = split_edges[["from_node", "to_node"]].to_dict("tight")["data"]

    split_edges = [coor for coor in split_edges if coor in graph.edges]

    split_nodes_edges["new_node_no1"] = (
        998_000_000_000 + split_nodes_edges.edge_no * 1_000 + 1
    )
    split_nodes_edges["new_node_no2"] = (
        998_000_000_000 + split_nodes_edges.edge_no * 1_000 + 2
    )
    split_nodes_edges["new_node_pos"] = split_nodes_edges.geometry.apply(
        lambda x: (x.x, x.y)
    )
    split_nodes_edges["upstream_node_no"] = [e[0] for e in split_edges]
    split_nodes_edges["downstream_node_no"] = [e[1] for e in split_edges]

    # remove splitted edges from graph and insert the newly split ones
    graph.remove_edges_from(split_edges)
    for i_edge, new in split_nodes_edges.iterrows():
        graph.add_node(new.new_node_no1, pos=new.new_node_pos)
        graph.add_node(new.new_node_no2, pos=new.new_node_pos)
        graph.add_edge(new.upstream_node_no, new.new_node_no1)
        graph.add_edge(new.new_node_no2, new.downstream_node_no)
    # update split nodes gdf with new node no
    new_graph_node_no = [(x1, x2) for x1, x2 in zip(split_nodes_edges['new_node_no1'], split_nodes_edges['new_node_no1'])]
    split_nodes.loc[split_nodes_edges.index, 'graph_node_no'] = pd.Series(new_graph_node_no, index=split_nodes_edges.index, dtype=object)

    # split_node: delete node and delete x edges, create x nodes, create x edges
    if "node_no" not in split_nodes.columns:
        split_nodes["node_no"] = -1
    split_nodes_nodes = split_nodes[split_nodes.node_no != -1]
    new_graph_node_no = []
    for split_node_id in split_nodes_nodes.node_no.values:
        if split_node_id not in graph:
            new_graph_node_no.append(-1)
            continue
        split_node_pos = graph.nodes[split_node_id]["pos"]
        split_edges = [e for e in list(graph.edges) if split_node_id in e]
        
        # remove old edges and node and insert new ones
        graph.remove_edges_from(split_edges)
        graph.remove_node(split_node_id)
        new_graph_no = []
        for i_edge, new_edge in enumerate(split_edges):
            new_node_id = 999_000_000_000 + split_node_id * 1_000 + i_edge
            graph.add_node(new_node_id, pos=split_node_pos)
            new_edge_adj = [e if e != split_node_id else new_node_id for e in new_edge]
            graph.add_edge(new_edge_adj[0], new_edge_adj[1])
            new_graph_no.append(new_node_id)
        new_graph_node_no.append(tuple(new_graph_no))
    # update split nodes gdf with new node no
    split_nodes.loc[split_nodes_nodes.index, 'graph_node_no'] = pd.Series(new_graph_node_no, index=split_nodes_nodes.index, dtype=object)
    print(f" - split network graph at split locations ({len(split_nodes)}x)")
    return graph, split_nodes


def add_basin_code_from_network_to_nodes_and_edges(
        graph: nx.DiGraph,
        nodes: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        split_nodes: gpd.GeoDataFrame,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    add basin (subgraph) code to nodes and edges
    """

    edges, nodes, split_nodes = edges.copy(), nodes.copy(), split_nodes.copy()  # copy to make sure gdf variable is not linked
    subgraphs = list(nx.weakly_connected_components(graph))
    if nodes is None or edges is None:
        return None, None
    nodes["basin"] = -1
    edges["basin"] = -1
    # prepare indexer to speed-up finding original node no for graph node no
    ix = split_nodes.index[(split_nodes['graph_node_no'] == -1) | pd.isna(split_nodes['graph_node_no'])]
    split_nodes.loc[ix, 'graph_node_no'] = pd.Series([(x,) for x in split_nodes.loc[ix, 'node_no']], index=ix, dtype=object)
    orig_node_indexer = {gn: no for no, gns in zip(split_nodes['node_no'], split_nodes['graph_node_no']) for gn in list(gns)}
    for i, subgraph in enumerate(subgraphs):
        # because in the graph nodes and edges can be changed to generate subgraphs we need to find
        # the original node no for the changed nodes. this information is stored in split_nodes_gdf
        node_ids = list(subgraph)
        orig_node_ids = [orig_node_indexer[n] if n in orig_node_indexer.keys() else n for n in node_ids]

        edges.loc[edges["from_node"].isin(orig_node_ids) & edges["to_node"].isin(orig_node_ids), "basin"] = i + 1
        nodes.loc[nodes["node_no"].isin(orig_node_ids), "basin"] = i + 1
    print(f" - define numbers Ribasim-Basins ({len(subgraphs)}x) and join edges/nodes")
    return nodes, edges


def check_if_split_node_is_used(
        split_nodes: gpd.GeoDataFrame, 
        nodes: gpd.GeoDataFrame, 
        edges: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
    """
    check whether split_nodes are used, split_nodes and split_edges
    """

    split_nodes = split_nodes.copy()  # copy to make sure gdf variable is not linked
    split_nodes["status"] = True

    # check if edges connected to split_nodes have the same basin code
    split_node_ids = [v for v in split_nodes.node_no.values if v != -1]
    split_nodes_not_used = []
    for split_node_id in split_node_ids:
        from_nodes = list(edges[edges.from_node == split_node_id].to_node.values)
        to_nodes = list(edges[edges.to_node == split_node_id].from_node.values)
        neighbours = nodes[nodes.node_no.isin(from_nodes + to_nodes)]
        if len(neighbours.basin.unique()) == 1:
            split_nodes_not_used.append(split_node_id)
    split_nodes.loc[split_nodes[split_nodes.node_no.isin(split_nodes_not_used)].index, "status"] = False

    # check if nodes connected to split_edge have the same basin code
    split_edge_ids = [v for v in split_nodes.edge_no.values if v != -1]
    split_edges_not_used = []
    for split_edge_id in sorted(split_edge_ids):
        end_nodes = list(edges[edges.edge_no == split_edge_id].to_node.values)
        start_nodes = list(edges[edges.edge_no == split_edge_id].from_node.values)
        neighbours = nodes[nodes.node_no.isin(end_nodes + start_nodes)]
        if len(neighbours.basin.unique()) == 1:
            split_edges_not_used.append(split_edge_id)
    split_nodes.loc[split_nodes[split_nodes.edge_no.isin(split_edges_not_used)].index, "status"] = False

    split_nodes["object_type"] = split_nodes["object_type"].fillna("manual")
    split_nodes["split_type"] = split_nodes["object_type"]
    split_nodes.loc[~split_nodes.status, "split_type"] = "no_split"
    print(f" - check whether each split location results in a split ({len(split_nodes.loc[~split_nodes['status']])}x not used)")
    return split_nodes


def create_basin_areas_based_on_drainage_unit_areas(
        edges: gpd.GeoDataFrame, 
        areas: gpd.GeoDataFrame,
        laterals: gpd.GeoDataFrame = None,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    find areas with spatial join on edges. add subgraph code to areas
    and combine all areas with certain subgraph code into one basin
    """

    if areas is None:
        return None, None
    else:
        areas = areas[["area_code", "geometry"]].copy()
    if edges is None:
        areas["basin"] = -1
        return areas, None
    else:
        edges = edges.copy()

    def get_area_code_for_lateral(laterals, areas):
        selected_areas = areas[[laterals.find(area) != -1 for area in areas["area_code"]]]
        if len(selected_areas)==0:
            return None
        else:
            return selected_areas["area_code"].values[0]

    def get_basin_code_from_lateral(area_code, laterals_join):
        basins = list(laterals_join["basin"][(laterals_join["area_code_included"]==area_code)&(laterals_join["basin"].isna()==False)].values)
        if len(basins) == 0:
            return -1
        else:
            return basins[0]

    if laterals is not None:
        laterals = laterals.copy()
        laterals["area_code_included"] = laterals["id"].apply(lambda x: get_area_code_for_lateral(x, areas))
        laterals_join = laterals.sjoin(
            gpd.GeoDataFrame(edges['basin'], geometry=edges['geometry'].buffer(1)),
            op="intersects",
            how="left",
        ).drop(columns=["index_right"])
        laterals_join["basin"] = laterals_join["basin"].fillna(-1).astype(int)
        areas["basin"] = areas.apply(lambda x: get_basin_code_from_lateral(x["area_code"], laterals_join), axis = 1)
        basin_areas = areas.dissolve(by="basin").explode().reset_index().drop(columns=["level_1"])

    else:
        edges_sel = edges.loc[edges["basin"] != -1].copy()
        # fix invalid area geometries
        areas["area"] = areas.index
        areas_orig = areas.copy()
        areas.geometry = areas.make_valid()
        # due to make_valid() GeometryCollections can be generated. Only take the (multi)polygon from those collections
        areas.geometry = [[g for g in gs.geoms if (isinstance(g, Polygon) or isinstance(g, MultiPolygon))][0] 
                        if isinstance(gs, GeometryCollection) 
                        else gs
                        for gs in areas.geometry]
        # spatial join edges to areas that intersect
        areas = areas.sjoin(edges_sel[["basin", "geometry"]])
        # we want to select the edge which is the longest within an area to ultimately select 
        # the right basin code
        edge_lengths = [gpd.GeoDataFrame(edges_sel.loc[ir].to_frame().T, geometry='geometry').clip(a).geometry.length.values[0] 
                        for a, ir in zip(areas.geometry, areas['index_right'])]
        areas['edge_length'] = edge_lengths
        areas = areas.drop(columns=["index_right"]).reset_index(drop=True)
        areas = areas.groupby(by=["area", "basin"], as_index=False).agg(
            {"edge_length": "sum"}
        )
        # this sorts first such that max edge length is first item and in drop_duplicates that first item will be kept
        # effectively method to get areas with basin code based on max edge length within area
        areas = areas.sort_values(
            by=["area", "edge_length"], ascending=[True, False]
        ).drop_duplicates(subset=["area"], keep="first")
        # insert area geometries back into gdf
        areas = (
            areas[["area", "basin", "edge_length"]]
            .sort_values(by="area")
            .merge(areas_orig, how="outer", left_on="area", right_on="area")
        )
        areas = gpd.GeoDataFrame(areas, geometry="geometry", crs=edges.crs)
        areas = areas.sort_values(by="area")
        areas.geometry = areas.make_valid()  # dissolve can fail because of incorrect geoms. fix those first
        basin_areas = areas.dissolve(by="basin").reset_index().drop(columns=["area"])

    basin_areas["basin"] = basin_areas["basin"].astype(int)
    basin_areas["area_ha"] = basin_areas.geometry.area / 10000.0
    basin_areas["color_no"] = basin_areas.index % 50
    print(
        f" - define for each Ribasim-Basin the related basin area ({len(basin_areas)}x)"
    )
    return areas, basin_areas


def create_basins_based_on_subgraphs_and_nodes(
        graph: nx.DiGraph, 
        nodes: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
    """
    create basin nodes based on basin_areas or nodes
    """
    connected_components = list(nx.weakly_connected_components(graph))
    centralities = {}
    for i, component in enumerate(connected_components):
        subgraph = graph.subgraph(component).to_undirected()
        centrality_subgraph = nx.closeness_centrality(subgraph)
        centralities.update(
            {node: centrality for node, centrality in centrality_subgraph.items()}
        )

    centralities = pd.DataFrame(
        dict(node_no=list(centralities.keys()), centrality=list(centralities.values()))
    )
    centralities = centralities[centralities["node_no"] < 900_000_000_000]
    basins_temp = nodes[['node_no', 'basin', 'geometry']].merge(
        centralities, 
        how="outer", 
        left_on="node_no", 
        right_on="node_no"
    )
    basins_temp = basins_temp[basins_temp["basin"] != -1].sort_values(
        by=["basin", "centrality"], ascending=[True, False]
    )
    basins = basins_temp.groupby(by="basin").first().reset_index().set_crs(nodes.crs)
    no_nodes_basin = basins_temp.groupby(by="basin")["basin"].count()
    no_nodes_basin.name = "no_nodes_basin"
    basins = basins.merge(no_nodes_basin, how='left', left_on="basin", right_index=True)
    
    basins1 = basins[basins.no_nodes_basin!=2]
    basins2 = basins[basins.no_nodes_basin==2]
    if ~basins2.empty:
        basins_temp2 = basins_temp[basins_temp.basin.isin(basins2.basin.values)]
        basins2a = basins_temp2.groupby("basin").first().reset_index().rename(columns={"node_no": "node_no1"})
        basins2b = basins_temp2.groupby("basin").last().reset_index()[["basin", "node_no"]].rename(columns={"node_no": "node_no2"})
        basins2x = basins2a.merge(basins2b, how='left', on='basin')

        basins2y = basins2x[["basin", "node_no1", "node_no2"]].merge(
            edges[["basin", "edge_no", "from_node", "to_node"]], 
            how='left', 
            left_on=["basin", "node_no1"], 
            right_on=["basin", "from_node"]
        ).fillna(-1).astype(int)
        basins2z = basins2y.merge(
            edges[["basin", "edge_no", "from_node", "to_node"]], 
            how='left', 
            left_on=["basin", "node_no2"], 
            right_on=["basin", "from_node"]
        ).fillna(-1).astype(int)
        basins2z.loc[basins2z["edge_no_x"]!=-1, "edge_no"] = basins2z["edge_no_x"]
        basins2z.loc[basins2z["edge_no_y"]!=-1, "edge_no"] = basins2z["edge_no_y"]
        basins2z.loc[basins2z["from_node_x"]!=-1, "from_node"] = basins2z["from_node_x"]
        basins2z.loc[basins2z["from_node_y"]!=-1, "from_node"] = basins2z["from_node_y"]
        basins2z.loc[basins2z["to_node_x"]!=-1, "to_node"] = basins2z["to_node_x"]
        basins2z.loc[basins2z["to_node_y"]!=-1, "to_node"] = basins2z["to_node_y"]
        basins2z = basins2z.fillna(-1).astype(int).merge(edges[["edge_no", "geometry"]], how="left", on="edge_no")
        basins2s = basins2z[
            ((basins2z.from_node==basins2z.node_no1) & (basins2z.to_node==basins2z.node_no2)) | 
            ((basins2z.from_node==basins2z.node_no2) & (basins2z.to_node==basins2z.node_no1))
        ]
        basins2s = gpd.GeoDataFrame(
            basins2s,
            geometry=[v.centroid for v in basins2s.geometry.values],
            crs=basins.crs
        )
        basins2_geometry = basins2["geometry"]
        basins2 = basins2.drop(columns=["geometry"]).merge(basins2s[["basin", "geometry"]], how="left", on="basin")
        basins2["geometry"] = basins2["geometry"].fillna(basins2_geometry)

    basins = pd.concat([basins1, basins2])
    basins = basins[basins.geometry != None]
    print(f" - create final locations Ribasim-Basins ({len(basins)}x)")
    return basins


def check_if_nodes_edges_within_basin_areas(nodes, edges, basin_areas):
    """
    check whether nodes assigned to a basin are also within the polygon assigned to that basin
    """
    
    edges, nodes, basin_areas = edges.copy(), nodes.copy(), basin_areas.copy()  # copy to make sure gdf variable is not linked
    if basin_areas is None:
        nodes["basin_area"] = -1
        nodes["basin_check"] = True
        edges["basin_area"] = -1
        edges["basin_check"] = True
        return nodes, edges

    nodes = nodes.drop(columns=["basin_area"], errors="ignore")
    nodes = gpd.sjoin(nodes, basin_areas[["geometry", "basin"]], how="left").drop(
        columns=["index_right"]
    )
    nodes["basin_right"] = nodes["basin_right"].fillna(-1).astype(int)
    nodes = nodes.rename(columns={"basin_left": "basin", "basin_right": "basin_area"})
    nodes["basin_check"] = nodes["basin"] == nodes["basin_area"]

    edges = edges.drop(columns=["basin_area"], errors="ignore")
    edges = gpd.sjoin(
        edges, basin_areas[["geometry", "basin"]], how="left", predicate="within"
    ).drop(columns=["index_right"])
    edges = edges.rename(columns={"basin_left": "basin", "basin_right": "basin_area"})
    edges["basin_area"] = edges["basin_area"].fillna(-1).astype(int)
    edges["basin_check"] = edges["basin"] == edges["basin_area"]
    return nodes, edges


def create_basin_connections(
        split_nodes: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        nodes: gpd.GeoDataFrame,
        basins: gpd.GeoDataFrame,
        crs: int = 28992,
    ) -> gpd.GeoDataFrame:
    """
    create basin connections
    """
    
    conn = split_nodes.rename(columns={"geometry": "geom_split_node"})
 
    # check if split_node is used (split_type) or is a hard cut
    conn = conn[(conn["split_type"] != "no_split") & (conn["split_type"] != "harde_knip")]

    # use different approach for: 
    # (1) splitnodes that are structures and on an edge and 
    # (2) splitnodes that are original d-hydro nodes

    # (1) splitnodes that are located on an edge
    conn_struct = conn.loc[conn["edge_no"] != -1].drop(["node_no", "from_node", "to_node"], axis=1, errors='ignore')
    # merge with edge to find us and ds nodes
    conn_struct = conn_struct.merge(
        edges[["from_node", "to_node", "edge_no"]],
        left_on="edge_no",
        right_on="edge_no"
    )
    # TODO: check for each edge the maximum absolute flow direction, in case of negative, reverse from_node/to_node
    # merge with node to find us and ds basin
    conn_struct_us = conn_struct.merge(
        nodes[["node_no", "basin"]],
        left_on="from_node",
        right_on="node_no",
    )
    conn_struct_us = conn_struct_us.drop(columns=["from_node", "to_node", "node_no", "edge_no"])
    conn_struct_ds = conn_struct.merge(
        nodes[["node_no", "basin"]],
        left_on="to_node", 
        right_on="node_no", 
    ).drop(columns=["from_node", "to_node", "node_no", "edge_no"])

    # (2) splitnodes that are original d-hydro nodes
    # merge splitnodes add connected edges
    conn_nodes = conn.loc[conn["node_no"] != -1].drop(["edge_no", "from_node", "to_node"], axis=1, errors='ignore')

    conn_nodes_ds = conn_nodes.merge(
        edges[["basin", "from_node", "to_node", "edge_no"]],
        left_on="node_no",
        right_on="from_node",
    ).drop(columns=["from_node", "to_node", "node_no", "edge_no"])
    conn_nodes_us = conn_nodes.merge(
        edges[["basin", "from_node", "to_node", "edge_no"]],
        left_on="node_no",
        right_on="to_node",
    ).drop(columns=["from_node", "to_node", "node_no", "edge_no"])

    # TODO: check for each edge the maximum absolute flow direction, in case of negative, cut and past in other dataframe.
    # Combine (1) en (2)
    conn_ds = pd.concat([conn_nodes_ds, conn_struct_ds])
    conn_us = pd.concat([conn_nodes_us, conn_struct_us])

    # merge splitnodes with basin DOWNSTREAM
    conn_ds = conn_ds.merge(
        basins[["basin", "geometry"]],
        left_on="basin",
        right_on="basin"
    ).rename(columns={"geometry": "geom_basin"})
    conn_ds["connection"] = "split_node_to_basin"
    conn_ds["geometry"] = conn_ds.apply(lambda x: LineString([x.geom_split_node, x.geom_basin]), axis=1)

    # merge splitnodes with basin UPSTREAM
    conn_us = conn_us.merge(
        basins[["basin", "geometry"]],
        left_on="basin",
        right_on="basin"
    ).rename(columns={"geometry": "geom_basin"})
    conn_us["connection"] = "basin_to_split_node"
    conn_us["geometry"] = conn_us.apply(lambda x: LineString([x.geom_basin, x.geom_split_node]), axis=1)

    basin_connections = pd.concat([conn_ds, conn_us]).drop(
        columns=["geom_basin", "geom_basin", "geom_split_node"]
    )
    basin_connections = gpd.GeoDataFrame(
        basin_connections, 
        geometry='geometry', 
        crs=crs
    )
    print(f" - create connections between Basins and split locations ({len(basin_connections)}x)")
    return basin_connections


def create_boundary_connections(
        boundaries: gpd.GeoDataFrame,
        nodes: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        basins: gpd.GeoDataFrame,
        split_nodes: gpd.GeoDataFrame,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    create boundary-basin connections
    """

    print(f" - create Ribasim-Edges between Boundaries and Basins")
    _split_nodes = split_nodes[(split_nodes["split_type"] != "no_split") & (split_nodes["split_type"] != "harde_knip")]
    _split_nodes_extra = split_nodes[(split_nodes["split_type"] == "no_split") | (split_nodes["split_type"] == "harde_knip")]

    if boundaries is None or nodes is None or basins is None:
        return None, split_nodes
    
    # merge boundaries with nodes and basins
    boundaries_conn = boundaries[
        ["boundary", "quantity", "geometry"]
    ].sjoin_nearest(
        nodes[["node_no", "geometry", "basin"]]
    )
    
    boundaries_conn = boundaries_conn.drop(
        columns=["index_right"]
    ).rename(
        columns={"node_no": "boundary_node_no"}
    )
    boundaries_conn.crs = nodes.crs
    boundaries_conn = boundaries_conn.merge(
        basins[["basin", "geometry"]], how="left", on="basin"
    ).rename(columns={"geometry_x": "geometry_boundary", "geometry_y": "geometry_basin"})
    
    # Discharge boundaries (1 connection, always inflow)
    dischargebnd_conn_in = boundaries_conn[
        boundaries_conn.quantity == "dischargebnd"
    ]
    if dischargebnd_conn_in.empty:
        dischargebnd_conn_in["geometry"] = None
    else:
        dischargebnd_conn_in.loc[:, "geometry"] = dischargebnd_conn_in.apply(
            lambda x: LineString([x["geometry_boundary"], x["geometry_basin"]]),
            axis=1,
        )
    dischargebnd_conn_in['connection'] = 'boundary_to_basin'
    
    # Water level boundaries (additional split_node, 2 connections)
    waterlevelbnd_conn = boundaries_conn[
        boundaries_conn.quantity == "waterlevelbnd"
    ]

    if not waterlevelbnd_conn.empty:

        def midpoint_two_points(p1, p2):
            return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
                
        waterlevelbnd_conn.loc[:, "midpoint"] = waterlevelbnd_conn.apply(
            lambda row: midpoint_two_points(row["geometry_boundary"], row["geometry_basin"]),
            axis=1,
        )

        # Inflow connection
        waterlevelbnd_conn_in1 = waterlevelbnd_conn[
            waterlevelbnd_conn.boundary_node_no.isin(edges.from_node)
        ]
        waterlevelbnd_conn_in2 = waterlevelbnd_conn_in1.copy()
        if waterlevelbnd_conn_in1.empty:
            waterlevelbnd_conn_in1["geometry"] = None
            waterlevelbnd_conn_in2["geometry"] = None
            waterlevelbnd_conn_in1["split_node_id"] = None
            waterlevelbnd_conn_in2["split_node_id"] = None
        else:
            waterlevelbnd_conn_in1.loc[:, "geometry"] = waterlevelbnd_conn_in1.apply(
                lambda x: LineString([x["geometry_boundary"], x["midpoint"]]), axis=1
            )
            waterlevelbnd_conn_in2.loc[:, "geometry"] = waterlevelbnd_conn_in2.apply(
                lambda x: LineString([x["midpoint"], x["geometry_basin"]]),
                axis=1,
            )
            if "boundary_node_id" in waterlevelbnd_conn_in1.columns:
                waterlevelbnd_conn_in1["split_node_id"] = "BoundConn_" + waterlevelbnd_conn_in1["boundary_node_id"].astype(str)
                waterlevelbnd_conn_in2["split_node_id"] = "BoundConn_" + waterlevelbnd_conn_in2["boundary_node_id"].astype(str)
            else:
                waterlevelbnd_conn_in1["split_node_id"] = "BoundConn_" + waterlevelbnd_conn_in1["boundary_node_no"].astype(str)
                waterlevelbnd_conn_in2["split_node_id"] = "BoundConn_" + waterlevelbnd_conn_in2["boundary_node_no"].astype(str)
        waterlevelbnd_conn_in1['connection'] = 'boundary_to_split_node'
        waterlevelbnd_conn_in2['connection'] = 'split_node_to_basin'

        # Outflow connection
        waterlevelbnd_conn_out1 = waterlevelbnd_conn[
            waterlevelbnd_conn.boundary_node_no.isin(edges.to_node)
        ]
        waterlevelbnd_conn_out2 = waterlevelbnd_conn_out1.copy()
        if waterlevelbnd_conn_out1.empty:
            waterlevelbnd_conn_out1["geometry"] = None
            waterlevelbnd_conn_out2["geometry"] = None
            waterlevelbnd_conn_out1["split_node_id"] = ""
            waterlevelbnd_conn_out2["split_node_id"] = ""
        else:
            waterlevelbnd_conn_out1.loc[:, "geometry"] = waterlevelbnd_conn_out1.apply(
                lambda x: LineString([x["geometry_basin"], x["midpoint"]]),
                axis=1,
            )
            waterlevelbnd_conn_out2.loc[:, "geometry"] = waterlevelbnd_conn_out2.apply(
                lambda x: LineString([x["midpoint"], x["geometry_boundary"]]),
                axis=1,
            )
            if "boundary_node_id" in waterlevelbnd_conn_in1.columns:
                waterlevelbnd_conn_out1["split_node_id"] = "BoundConn_" + waterlevelbnd_conn_out1["boundary_node_id"].astype(str)
                waterlevelbnd_conn_out2["split_node_id"] = "BoundConn_" + waterlevelbnd_conn_out2["boundary_node_id"].astype(str)
            else:
                waterlevelbnd_conn_out1["split_node_id"] = "BoundConn_" + waterlevelbnd_conn_out1["boundary_node_no"].astype(str)
                waterlevelbnd_conn_out2["split_node_id"] = "BoundConn_" + waterlevelbnd_conn_out2["boundary_node_no"].astype(str)
        waterlevelbnd_conn_out1['connection'] = 'basin_to_split_node'
        waterlevelbnd_conn_out2['connection'] = 'split_node_to_boundary'
        
        boundaries_conn = gpd.GeoDataFrame(
            pd.concat([
                dischargebnd_conn_in,
                waterlevelbnd_conn_in1,
                waterlevelbnd_conn_in2,
                waterlevelbnd_conn_out1,
                waterlevelbnd_conn_out2,
            ], ignore_index=True), 
            geometry='geometry', 
            crs=split_nodes.crs
        )

        additional_split_nodes = gpd.GeoDataFrame(
            (boundaries_conn[["split_node_id", "midpoint"]]
            .drop_duplicates(subset='split_node_id')
            .rename({"midpoint": "geometry"}, axis=1)),
            geometry='geometry', 
            crs=split_nodes.crs
        ).dropna(how='all').drop_duplicates()

        additional_split_nodes['node_no'] = -1
        additional_split_nodes['edge_no'] = -1
        additional_split_nodes['object_type'] = "boundary_connection"
        additional_split_nodes['split_type'] = "boundary_connection"
        additional_split_nodes['status'] = True
        additional_split_nodes['split_node'] = -1

        split_nodes = gpd.GeoDataFrame(
            pd.concat([_split_nodes, _split_nodes_extra, additional_split_nodes]),
            geometry='geometry', 
            crs=split_nodes.crs
        )

    else:
        boundaries_conn = dischargebnd_conn_in.copy()
        split_nodes = gpd.GeoDataFrame(
            pd.concat([_split_nodes, _split_nodes_extra]),
            geometry='geometry', 
            crs=split_nodes.crs
        )

    split_nodes = split_nodes.reset_index(drop=True)
    split_nodes['split_node'] = split_nodes.index + 1
    
    boundaries_conn = boundaries_conn[
        [x for x in 
         ["connection", "boundary", "boundary_node_no", "boundary_node_id", 
         "basin", "split_node_id", "geometry"]
         if x in boundaries_conn.columns]
    ].merge(
        split_nodes[['split_node_id', 'split_node']], 
        how='left',
        on='split_node_id'
    ).sort_values(['boundary', 'basin']).reset_index(drop=True)
    
    boundaries_conn['split_node'] = boundaries_conn['split_node'].fillna(-1).astype(int)

    return boundaries_conn, split_nodes


def regenerate_node_ids(
        boundaries: gpd.GeoDataFrame,
        split_nodes: gpd.GeoDataFrame, 
        basins: gpd.GeoDataFrame,
        basin_connections: gpd.GeoDataFrame,
        boundary_connections: gpd.GeoDataFrame,
        basin_areas: gpd.GeoDataFrame,
        nodes: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        areas: gpd.GeoDataFrame,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Regenerate ribasim node-id for nodes and edges
    """

    boundaries, split_nodes, basins, basin_connections = boundaries.copy(), split_nodes.copy(), basins.copy(), basin_connections.copy()
    boundary_connections, basin_areas, nodes, edges, areas = boundary_connections.copy(), basin_areas.copy(), nodes.copy(), edges.copy(), areas.copy()

    print(f" - regenerate node-ids Ribasim-Nodes and Ribasim-Edges")
    # boundaries
    if boundaries is not None:
        if "boundary_node_id" in boundaries.columns:
            boundaries = boundaries.drop(columns=["boundary_node_id"])
        boundaries.insert(1, "boundary_node_id", boundaries["boundary"])
            
        len_boundaries = len(boundaries)
    else:
        len_boundaries = 0

    # split_nodes
    if "split_node_node_id" not in split_nodes.columns:
        split_nodes.insert(loc=1, column="split_node_node_id", value=split_nodes["split_node"] + len_boundaries)
    else:
        split_nodes["split_node_node_id"] = split_nodes["split_node"] + len_boundaries
    len_split_nodes = len(split_nodes)

    # basins
    if "basin_node_id" not in basins.columns:
        basins.insert(loc=1, column="basin_node_id", value=basins["basin"] + len_split_nodes + len_boundaries)
    else:
        basins["basin_node_id"] = basins["basin"] + len_split_nodes + len_boundaries
    len_basins = len(basins)

    # basin_connections
    basin_connections["split_node_node_id"] = basin_connections["split_node"] + len_boundaries
    basin_connections["basin_node_id"] = basin_connections["basin"] + len_split_nodes + len_boundaries
    basin_connections["from_node_id"] = basin_connections.apply(
        lambda x: x["basin_node_id"] if x["connection"].startswith("basin") else x["split_node_node_id"], 
        axis=1
    )
    basin_connections["to_node_id"] = basin_connections.apply(
        lambda x: x["basin_node_id"] if x["connection"].endswith("basin") else x["split_node_node_id"], 
        axis=1
    )
    
    # boundary_connections
    boundary_connections["boundary_node_id"] = boundary_connections["boundary"]
    boundary_connections["split_node_node_id"] = boundary_connections["split_node"] + len_boundaries
    boundary_connections["basin_node_id"] = boundary_connections["basin"] + len_split_nodes + len_boundaries
    boundary_connections["from_node_id"] = boundary_connections.apply(
        lambda x: x["basin_node_id"] if x["connection"].startswith("basin") else (
            x["boundary_node_id"] if x["connection"].startswith("boundary") else x["split_node_node_id"]
        ), axis=1
    ).fillna(-1).astype(int)
    boundary_connections["to_node_id"] = boundary_connections.apply(
        lambda x: x["basin_node_id"] if x["connection"].endswith("basin") else (
            x["boundary_node_id"] if x["connection"].endswith("boundary") else x["split_node_node_id"]
        ), axis=1
    ).fillna(-1).astype(int)

    basin_areas = basin_areas.merge(basins[["basin", "basin_node_id"]], on="basin")

    areas["basin_node_id"] = areas["basin"].apply(lambda x: x + len_split_nodes + len_boundaries if x>0 else -1)

    mapping_basins_node_id = basins.set_index("basin")["basin_node_id"].to_dict()
    nodes["basin_node_id"] = nodes["basin"].replace(mapping_basins_node_id)
    edges["basin_node_id"] = edges["basin"].replace(mapping_basins_node_id)

    # connections = pd.concat([
    #     basin_connections[["from_node_id", "to_node_id"]], 
    #     boundary_connections[["from_node_id", "to_node_id"]]
    # ])
    # split_nodes = gpd.GeoDataFrame(split_nodes.merge(
    #     connections.set_index("to_node_id"), 
    #     how="left", 
    #     left_on="split_node_node_id", 
    #     right_index=True
    # ), geometry="geometry", crs=split_nodes.crs)
    # split_nodes = gpd.GeoDataFrame(split_nodes.merge(
    #     connections.set_index("from_node_id"), 
    #     how="left", 
    #     left_on="split_node_node_id", 
    #     right_index=True
    # ), geometry="geometry", crs=split_nodes.crs)

    # the above actions can result in duplicate entries in tables. only keep one records of those duplicates
    boundaries = boundaries.loc[~boundaries.duplicated()].copy()
    split_nodes = split_nodes.loc[~split_nodes.duplicated()].copy()
    basins = basins.loc[~basins.duplicated()].copy()
    basin_areas = basin_areas.loc[~basin_areas.duplicated()].copy()
    nodes = nodes.loc[~nodes.duplicated()].copy()
    edges = edges.loc[~edges.duplicated()].copy()
    areas = areas.loc[~areas.duplicated()].copy()

    return boundaries, split_nodes, basins, basin_areas, nodes, edges, areas


def generate_ribasim_types_for_all_split_nodes(
        boundaries: gpd.GeoDataFrame, 
        split_nodes: gpd.GeoDataFrame, 
        basins: gpd.GeoDataFrame, 
        split_node_type_conversion: Dict, 
        split_node_id_conversion: Dict,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """"
    Generate Ribasim Types for all split nodes
    """

    print(f" - define Ribasim-Nodes types based on input conversion table(s)")
    # Basins
    basins["ribasim_type"] = "Basin"
    basins["name"] = "Basin"

    # Boundaries
    boundary_conversion = {
        "dischargebnd": "FlowBoundary", 
        "waterlevelbnd": "LevelBoundary"
    }
    boundaries["ribasim_type"] = boundaries["quantity"].replace(boundary_conversion)
    # Split nodes
    removed_split_nodes = None
    if not split_nodes[~split_nodes.status].empty:
        removed_split_nodes = split_nodes[~split_nodes.status].copy()
        print(f"   * {len(removed_split_nodes)} split_nodes resulting in no_split")
        split_nodes = split_nodes[split_nodes.status]

    split_nodes["ribasim_type"] = "TabulatedRatingCurve" 
    split_nodes_conversion = {
        "weir": "TabulatedRatingCurve",
        "uniweir": "TabulatedRatingCurve",
        "pump": "Pump",
        "culvert":"ManningResistance",
        "manual": "ManningResistance",
        "orifice": "TabulatedRatingCurve",
        "boundary_connection": "ManningResistance"
    }
    if isinstance(split_node_type_conversion, Dict):
        for key, value in split_node_type_conversion.items():
            split_nodes_conversion[key] = value
    split_nodes["ribasim_type"] = split_nodes["split_type"].replace(split_nodes_conversion)

    if isinstance(split_node_id_conversion, Dict):
        for key, value in split_node_id_conversion.items():
            if len(split_nodes[split_nodes["split_node_id"] == key]) == 0:
                print(f"   * split_node type conversion id={key} (type={value}) does not exist")
            split_nodes.loc[split_nodes["split_node_id"] == key, "ribasim_type"] = value

    # add removed split nodes back into gdf
    if removed_split_nodes is not None:
        split_nodes = pd.concat([split_nodes, removed_split_nodes], axis=0)
    return boundaries, split_nodes, basins


def check_basins_connected_to_basin_areas(
        basins: gpd.GeoDataFrame, 
        basin_areas: gpd.GeoDataFrame,
        boundary_connections: gpd.GeoDataFrame,
    ):
    """
     Check if basin locations are connected with a basin area (ignoring basins generated for boundary connections)
    """
    print(' - check if basins are connected to a basin area (ignoring basins generated for boundary connections)')
    _basins = basins.loc[~np.isin(basins['basin'].values, boundary_connections['basin'].values)]
    _basins = _basins.loc[~np.isin(_basins['basin'].values, basin_areas['basin'].values)]
    if not _basins.empty:
        print('   - Following basins are not connected to a basin area (either due to incorrect network input or multiple basins in the same basin area):')
        print(f'      Basins ', end="", flush=True)
        for b in sorted(_basins['basin'].values):
            print(f"{b}, ", end="", flush=True)
    print("")


def remove_isolated_basins_and_update_administration(
        basins: gpd.GeoDataFrame,
        basin_areas: gpd.GeoDataFrame,
        areas: gpd.GeoDataFrame,
        basin_connections: gpd.GeoDataFrame,
        boundary_connections: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        nodes: gpd.GeoDataFrame,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Remove isolated basins (including 1-on-1 connected basin areas) based on basin and boundary connections.
    Update the basin administration in basin areas, areas, edges and nodes
    Returns basin, basin areas, areas, edges and nodes
    """
    basins, basin_areas, edges, nodes = basins.copy(), basin_areas.copy(), edges.copy(), nodes.copy()

    # get isolated basins
    connected_basins = np.unique(basin_connections['basin'].tolist() + boundary_connections['basin'].tolist())
    isolated_basins = basins.loc[~np.isin(basins['basin'].values, connected_basins), 'basin'].values
    print(f' - removing isolated basins ({len(isolated_basins)}x)')
    if len(isolated_basins) == 0:
        return basins, basin_areas, edges, nodes  # no isolated basins so no further steps needed

    # get mapping of which basin are located within a basin area
    _basin_areas = basin_areas.sjoin(basins, how='left')
    basin_area_to_basin = {k: _basin_areas.loc[[k], 'basin_right'].values for k in _basin_areas.index.unique()}
    basin_area_to_basin = {k: v for k, v in basin_area_to_basin.items() if not all(np.isnan(v))}

    # get mapping of removed isolated basins to new basin based on the location within the same basin area
    basin_to_basin_area = {_v: k for k, v in basin_area_to_basin.items() for _v in v}
    isolated_basin_to_basin = {ib: basin_area_to_basin[basin_to_basin_area[ib]] for ib in isolated_basins if ib in basin_to_basin_area.keys()}
    isolated_basin_to_basin = {k: v[np.isin(v, basins.loc[np.isin(basins['basin'].values, connected_basins)]['basin'])] for k, v in isolated_basin_to_basin.items()}
    isolated_basin_to_basin = {k: int(v[0]) if len(v) > 0 else -(1000 + k) for k, v in isolated_basin_to_basin.items()}

    # update basins, basin areas, edges and nodes. if removed basin is 1-on-1 connected to a basin area, the basin area will be removed.
    # this also means that the basin and basin area of associated nodes and edges of that removed basin will be removed
    basins = basins.loc[np.isin(basins['basin'].values, connected_basins)]
    basin_areas['basin'] = [isolated_basin_to_basin[b] if b in isolated_basin_to_basin.keys() else b for b in basin_areas['basin']]
    edges['basin'] = [isolated_basin_to_basin[b] if b in isolated_basin_to_basin.keys() else b for b in edges['basin']]
    nodes['basin'] = [isolated_basin_to_basin[b] if b in isolated_basin_to_basin.keys() else b for b in nodes['basin']]
    # remove 1-on-1 connected basin areas
    edges['basin_area'] = [-1 if b < -10 else ba for b, ba in zip(edges['basin'], edges['basin_area'])]
    nodes['basin_area'] = [-1 if b < -10 else ba for b, ba in zip(nodes['basin'], nodes['basin_area'])]
    basin_areas = basin_areas.loc[basin_areas['basin'] >= 0]  
    # update areas
    areas.loc[np.isin(areas['basin'].values, isolated_basins), 'basin'] = np.nan

    return basins, basin_areas, areas, edges, nodes


def generate_ribasim_network_using_split_nodes(
        nodes: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        split_nodes: gpd.GeoDataFrame,
        areas: gpd.GeoDataFrame,
        boundaries: gpd.GeoDataFrame,
        laterals: gpd.GeoDataFrame,
        use_laterals_for_basin_area: bool,
        remove_isolated_basins: bool,
        split_node_type_conversion: Dict, 
        split_node_id_conversion: Dict,
        crs: int = 28992,
    ) -> Dict:
    """create basins (nodes) and basin_areas (large polygons) and connections (edges)
    based on nodes, edges, split_nodes and areas (discharge units).
    This function calls all other functions
    """

    print("Create Ribasim network using Network and Split nodes:")
    network_graph = None
    basin_areas = None
    basins = None
    network_graph = create_graph_based_on_nodes_edges(
        nodes=nodes,
        edges=edges
    )
    network_graph, split_nodes = split_graph_based_on_split_nodes(
        graph=network_graph, 
        split_nodes=split_nodes,
        edges=edges
    )
    nodes, edges = add_basin_code_from_network_to_nodes_and_edges(
        graph=network_graph, 
        split_nodes=split_nodes,
        nodes=nodes,
        edges=edges
    )
    split_nodes = check_if_split_node_is_used(
        split_nodes=split_nodes,
        nodes=nodes,
        edges=edges
    )
    basins = create_basins_based_on_subgraphs_and_nodes(
        graph=network_graph, 
        nodes=nodes,
        edges=edges
    )
    if use_laterals_for_basin_area:
        areas, basin_areas = create_basin_areas_based_on_drainage_unit_areas(
            edges=edges, 
            areas=areas,
            laterals=laterals
        )
    else:
        areas, basin_areas = create_basin_areas_based_on_drainage_unit_areas(
            edges=edges, 
            areas=areas
        )
    nodes, edges = check_if_nodes_edges_within_basin_areas(
        nodes=nodes, 
        edges=edges, 
        basin_areas=basin_areas
    )
    basin_connections = create_basin_connections(
        split_nodes=split_nodes,
        basins=basins,
        nodes=nodes,
        edges=edges,
        crs=crs
    )
    boundary_connections, split_nodes = create_boundary_connections(
        boundaries=boundaries,
        split_nodes=split_nodes,
        basins=basins,
        nodes=nodes,
        edges=edges
    )
    if remove_isolated_basins:
        basins, basin_areas, areas, edges, nodes = remove_isolated_basins_and_update_administration(
            basins=basins, 
            basin_areas=basin_areas,
            areas=areas,
            basin_connections=basin_connections,
            boundary_connections=boundary_connections,
            edges=edges,
            nodes=nodes,
        )
    boundaries, split_nodes, basins, basin_areas, nodes, edges, areas = regenerate_node_ids(
        boundaries=boundaries,
        split_nodes=split_nodes,
        basins=basins,
        basin_connections=basin_connections,
        boundary_connections=boundary_connections,
        basin_areas=basin_areas,
        nodes=nodes,
        edges=edges,
        areas=areas
    )
    boundaries, split_nodes, basins = generate_ribasim_types_for_all_split_nodes(
        boundaries=boundaries, 
        split_nodes=split_nodes, 
        basins=basins, 
        split_node_type_conversion=split_node_type_conversion, 
        split_node_id_conversion=split_node_id_conversion
    )
    check_basins_connected_to_basin_areas(
        basins=basins, 
        basin_areas=basin_areas,
        boundary_connections=boundary_connections,
    )
    return dict(
        basin_areas=basin_areas,
        basins=basins,
        areas=areas,
        nodes=nodes,
        edges=edges,
        split_nodes=split_nodes,
        network_graph=network_graph,
        basin_connections=basin_connections,
        boundary_connections=boundary_connections,
    )
