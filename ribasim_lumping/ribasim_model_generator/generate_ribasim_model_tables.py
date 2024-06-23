import geopandas as gpd
import numpy as np
import pandas as pd


def generate_basin_static_table(basin_h, basin_a, basins, decimals=3):
    if basin_h is None:
        return pd.DataFrame(
            data={
                "node_id": np.repeat(basins.basin_node_id.values, 2), 
                "level": [0.0, 1.0]*len(basins), 
                "area": [1000.0, 1000.0]*len(basins)
            }
        )
        
    basin_profile = pd.DataFrame(
        columns=['node_id', 'level', 'area', 'remarks']
    )
    for basin_node_id in basin_h.columns:
        basin_profile_col = basin_h[[basin_node_id]].reset_index(drop=True)
        basin_profile_col.columns = ['level']
        basin_profile_col["node_id"] = basin_node_id
        basin_profile_col['area'] = basin_a[[basin_node_id]].reset_index(drop=True)
        basin_profile = pd.concat([basin_profile, basin_profile_col])
    basin_profile["remarks"] = ""
    basin_profile = basin_profile[basin_profile['level'].notna()]
    basin_profile["area"] = basin_profile["area"].replace(0.0, 0.001)
    basin_profile = basin_profile.reset_index(drop=True)
    basin_profile["level_diff"] = basin_profile["level"].diff()
    
    for node_id in basin_profile.node_id.unique():
        basin_profile.loc[(basin_profile['node_id'] == node_id).idxmax(), 'level_diff'] = np.nan

    basin_profile = basin_profile[
        (basin_profile["level_diff"].diff(1) > 0.0001) | (basin_profile["level_diff"].isna())
    ].reset_index(drop=True)
    return basin_profile.drop(columns=["level_diff"])


def generate_basin_time_table_laterals(basins, basin_areas, laterals, laterals_data, saveat):
    laterals_basins = (laterals[["id", "geometry"]]
                       .sjoin(basin_areas[["basin_node_id", "geometry"]]).drop(columns=["index_right"])
                       [["id", "basin_node_id"]])
    if saveat is not None:
        laterals_data = laterals_data.resample(f"{saveat}S").interpolate()
    
    timeseries = pd.DataFrame()
    for basin_no in basins["basin_node_id"].to_numpy():
        if basin_no in basin_areas["basin_node_id"].to_numpy():
            laterals_basin = laterals_basins[laterals_basins["basin_node_id"]==basin_no]["id"].to_numpy()
            laterals_basin = [l for l in laterals_basin if l in laterals_data.columns]
            timeseries_basin = laterals_data[laterals_basin].sum(axis=1)
            timeseries_basin.name = "drainage"
            timeseries_basin = timeseries_basin.to_frame()
        else:
            timeseries_basin = pd.DataFrame(index=laterals_data.index, columns=["drainage"])
            timeseries_basin["drainage"] = 0.0
        timeseries_basin.index.name = "time"
        timeseries_basin = timeseries_basin.reset_index()

        timeseries_basin["potential_evaporation"] = 0.0
        timeseries_basin["precipitation"] = 0.0
        timeseries_basin["infiltration"] = 0.0
        timeseries_basin["urban_runoff"] = 0.0
        timeseries_basin["node_id"] = basin_no
        
        timeseries = pd.concat([
            timeseries,
            timeseries_basin
        ])
    timeseries = timeseries.sort_values(["time", "node_id"]).reset_index(drop=True)
    timeseries = timeseries[["time", "node_id", "precipitation", "potential_evaporation", "drainage", "infiltration", "urban_runoff"]]
    return timeseries

def generate_basin_time_table_laterals_areas_data(basins, areas, laterals_areas_data):
    timeseries = pd.DataFrame()
    for basin_no in basins["basin_node_id"].values:
        areas_basin = list(areas[areas['basin_node_id'] == basin_no]['area_code'].unique())
        timeseries_basin = laterals_areas_data[areas_basin].sum(axis=1).to_frame().rename(columns={0: 'Netto_flux'}).reset_index()
        
        timeseries_basin["drainage"] = timeseries_basin["Netto_flux"][timeseries_basin["Netto_flux"]>0]
        timeseries_basin["drainage"] = timeseries_basin["drainage"].fillna(0.0)

        timeseries_basin["infiltration"] = timeseries_basin["Netto_flux"][timeseries_basin["Netto_flux"]<0]
        timeseries_basin["infiltration"] = timeseries_basin["infiltration"].fillna(0.0)
        
        timeseries_basin["potential_evaporation"] = 0.0
        timeseries_basin["precipitation"] = 0.0
        timeseries_basin["urban_runoff"] = 0.0
        timeseries_basin["node_id"] = basin_no

        timeseries = pd.concat([
            timeseries,
            timeseries_basin
        ])
    timeseries = timeseries.sort_values(["time", "node_id"]).reset_index(drop=True)
    timeseries = timeseries[["time", "node_id", "precipitation", "potential_evaporation", "drainage", "infiltration", "urban_runoff"]]
    return timeseries


def generate_basin_time_table_laterals_drainage_per_ha(basins, basin_areas, laterals_drainage_per_ha):
    laterals_drainage_per_ha.name = "drainage"
    laterals_drainage_per_ha = laterals_drainage_per_ha.resample("H").interpolate()
    drainage_m3_s_ha = laterals_drainage_per_ha.to_frame() / 1000.0
    drainage_m3_s_ha.index.name = "time"

    basin_areas_ha = basin_areas.set_index('basin_node_id')

    timeseries = pd.DataFrame()
    for basin_no in basins["basin_node_id"].values:
        timeseries_basin = drainage_m3_s_ha.reset_index()
        timeseries_basin["potential_evaporation"] = 0.0
        timeseries_basin["precipitation"] = 0.0
        timeseries_basin["infiltration"] = 0.0
        timeseries_basin["urban_runoff"] = 0.0
        timeseries_basin["node_id"] = basin_no

        if basin_no in basin_areas_ha.index:
            area = basin_areas_ha.loc[basin_no, 'area_ha']
            timeseries_basin["drainage"] = timeseries_basin["drainage"] * area
        else:
            timeseries_basin["drainage"] = 0.0

        timeseries = pd.concat([
            timeseries,
            timeseries_basin
        ])
    timeseries = timeseries.sort_values(["time", "node_id"]).reset_index(drop=True)
    timeseries = timeseries[["time", "node_id", "precipitation", "potential_evaporation", "drainage", "infiltration", "urban_runoff"]]
    return timeseries


def generate_tabulated_rating_curve(
    basins_outflows, tabulated_rating_curves, basin_h, \
    edge_q_df, weir_q_df, uniweir_q_df, orifice_q_df, culvert_q_df, bridge_q_df, pump_q_df, set_name
):
    basins_outflows_sel = basins_outflows.copy()
    basins_outflows_sel.columns = ["__".join(c) for c in basins_outflows_sel.columns]
    basins_outflows_sel = basins_outflows_sel[["general__split_node", "structure__crestwidth", set_name + "__crestlevel"]]
    basins_outflows_sel.columns = ["split_node", "crestwidth", "crestlevel"]
    
    tabulated_rating_curves = tabulated_rating_curves.merge(
        basins_outflows_sel, 
        how="left", 
        left_on="split_node", 
        right_on="split_node", 
    )

    curves = pd.DataFrame()
    for i, trc in tabulated_rating_curves.iterrows():
        basin_node_id = trc["from_node_id"]
        water_levels_basin = basin_h[basin_node_id]
        targetlevel = trc["crestlevel"]
        
        split_type = trc["split_type"]
        split_node_name = trc["split_node_id"]
        
        discharges_list = dict(
            weir=weir_q_df,
            universalWeir=uniweir_q_df,
            orifice=orifice_q_df,
            culvert=culvert_q_df,
            bridge=bridge_q_df,
            pump=pump_q_df,
            edge=edge_q_df,
        )
        discharge_df = discharges_list[split_type]
        # if discharge_df is None:
        #     curve = 
        discharges = discharge_df[split_node_name]

        curve = pd.concat([water_levels_basin, discharges.replace(-999.0, np.nan)], axis=1)
        curve.columns = ["level", "flow_rate"]
        curve.iloc[0]["flow_rate"] = 0.0
        curve.iloc[1]["flow_rate"] = 0.0
        curve.iloc[2]["flow_rate"] = 0.0
        curve.iloc[3]["flow_rate"] = 0.0
        curve["node_id"] = trc["split_node_node_id"]
        curve = curve.interpolate().reset_index(drop=True)
        curve.loc[curve["flow_rate"] < 0.0, "flow_rate"] = 0.0

        if curve["flow_rate"].max() < 0.01:
            print(f" x basin_node_id {basin_node_id}: no discharge over split_node ({split_type}): {trc.split_node_id} in D-HYDRO simulation")

            def weir_formula(crestlevel, crestwidth, waterlevel):
                return 2.0/3.0 * max(0.0, waterlevel - crestlevel)**(3.0/2.0) * (2*9.81)**0.5 * crestwidth

            curve.loc[3:, "level"] = [i*0.05 + trc.crestlevel for i in range(0, len(curve)-3)]
            curve["flow_rate"] = curve.apply(lambda x: weir_formula(trc.crestlevel, trc.crestwidth, x["level"]), axis=1).fillna(0.0)

        curves = pd.concat([curves, curve])
    return curves.drop_duplicates().reset_index(drop=True)


def generate_manning_resistances(manningresistance, set_name):
    return pd.DataFrame(
        data={
            "node_id": manningresistance["split_node_node_id"],
            "length": [750.0]*len(manningresistance),
            "manning_n": [0.04]*len(manningresistance),
            "profile_width": [5.0]*len(manningresistance),
            "profile_slope": [3.0]*len(manningresistance),
        }
    )


def generate_boundary_time_table(boundaries, boundaries_data, boundary_type):
    if boundaries.empty:
        return None

    # boundary table from csv with timeseries per boundary_node
    timeseries = pd.DataFrame()
    for i_boundary, boundary in boundaries.iterrows():
        # print(boundary)
        boundary_no = boundary["name"]

        # timeseries_boundary = boundaries_csv_data[boundary_no].sum(axis=1).to_frame().rename(columns={0: 'flow'}).reset_index()
        timeseries_boundary = boundaries_data[[boundary_no]].reset_index()
        timeseries_boundary= timeseries_boundary.rename(columns={boundary_no: boundary_type})
        timeseries_boundary[boundary_type] = timeseries_boundary[boundary_type][timeseries_boundary[boundary_type]>0]
        timeseries_boundary[boundary_type] = timeseries_boundary[boundary_type].fillna(0.0)

        timeseries_boundary["node_id"] = boundary["boundary_node_id"]

        timeseries = pd.concat([
            timeseries,
            timeseries_boundary
        ])
    timeseries = timeseries.sort_values(["time", "node_id"]).reset_index(drop=True)
    timeseries = timeseries[["time","node_id", boundary_type]]
    return timeseries


def generate_boundary_static_data(boundaries, boundaries_static_data):
    # test
    dummyvalue = 999
    boundary_data = pd.DataFrame()
    for boundary in boundaries:
        boundary_no = boundary["name"]

        boundary_data_boundary = boundaries_static_data[boundary_no].sum(axis=1).to_frame().rename(columns={0: 'flow'}).reset_index()
        boundary_data_boundary["flow"] = boundary_data_boundary["flow"].fillna(dummyvalue)
        boundary_data_boundary["node_id"] = boundary_no

        boundary_data = pd.concat([
            boundary_data,
            boundary_data_boundary
        ])
    boundary_data = boundary_data.sort_values(["node_id"]).reset_index(drop=True)
    boundary_data = boundary_data[["node_id", "flow"]]
    return boundary_data


def generate_basin_state(basin_h_initial, basin_profile, basin_h, set_name):
    if basin_h_initial is None:
        basin_state = basin_h.loc[(set_name, "targetlevel")].rename("level").reset_index().rename(columns={"basin_node_id": "node_id"})
        print('basin-state: generated based on targetlevel')
    else:                            
        basin_state = basin_h_initial.rename("level").reset_index().rename(columns={"basin_node_id": "node_id"})
        print('basin-state: generated as defined')
    # check whether initial water level is higher than bedlevel, if not: use bedlevel
    bedlevel = basin_profile.groupby("node_id").min()[["level"]].rename(columns={"level": "bedlevel"})
    state = basin_state.set_index("node_id")[["level"]]
    state = bedlevel.merge(state, how='left', left_index=True, right_index=True).max(axis=1)
    state.name = "level"
    basin_state = basin_state.drop(columns=["level"]).merge(state, how='left', left_on="node_id", right_index=True)
    return basin_state


def generate_basin_subgrid(basins_nodes_h_relation):
    basin_subgrid = basins_nodes_h_relation[["node_no", "basin_node_id", "basin_h", "node_no_h"]]
    basin_subgrid["x"] = basins_nodes_h_relation.geometry.x
    basin_subgrid["y"] = basins_nodes_h_relation.geometry.y
    basin_subgrid = basin_subgrid.rename(columns={
        "node_no": "subgrid_id", 
        "basin_node_id": "node_id", 
        "basin_h": "basin_level", 
        "node_no_h": "subgrid_level"
    })
    basin_subgrid = basin_subgrid.round(5).sort_values(by=["subgrid_id", "node_id", "basin_level"])
    basin_subgrid["basin_level_diff"] = basin_subgrid["basin_level"].diff()
    basin_subgrid["subgrid_level_diff"] = basin_subgrid["subgrid_level"].diff()

    for node_id in basin_subgrid.node_id.unique():
        basin_subgrid.loc[(basin_subgrid['node_id'] == node_id).idxmax(), ['basin_level_diff', 'subgrid_level_diff']] = np.nan

    basin_subgrid = basin_subgrid[
        ((basin_subgrid["basin_level_diff"].diff(1) > 0.0001) | (basin_subgrid["basin_level_diff"].isna())) & 
        ((basin_subgrid["subgrid_level_diff"].diff(1) > 0.0001) | (basin_subgrid["subgrid_level_diff"].isna()))
    ].reset_index(drop=True)
    return basin_subgrid.drop(columns=["basin_level_diff", "subgrid_level_diff"])


def generate_ribasim_model_tables(dummy_model, basin_h, basin_a, basins, areas, basin_areas, basins_nodes_h_relation,
    laterals, laterals_data, boundaries, boundaries_data, split_nodes, basins_outflows, set_name, 
    method_boundaries, boundaries_timeseries_data, 
    method_laterals, laterals_areas_data, laterals_drainage_per_ha, basin_h_initial, 
    saveat, edge_q_df, weir_q_df, uniweir_q_df, orifice_q_df, culvert_q_df, bridge_q_df, pump_q_df):

    # create tables for BASINS
    tables = dict()
    print('basin-profile: generated')
    tables['basin_profile'] = generate_basin_static_table(basin_h, basin_a, basins, decimals=3)

    # create tables for INITIAL STATE
    print('basin-state: generated')
    tables["basin_state"] = generate_basin_state(basin_h_initial, tables["basin_profile"], basin_h, set_name)

    # create subgrid
    if basins_nodes_h_relation is not None:
        print("subgrid: based on water level relation basin and nodes")
        tables['basin_subgrid'] = generate_basin_subgrid(basins_nodes_h_relation)

    # create laterals table
    if method_laterals == 1:
        if laterals is None or laterals_data is None:
            raise ValueError("method_laterals = 1 requires laterals and laterals_areas_data")
        tables['basin_time'] = generate_basin_time_table_laterals(
            basins=basins, 
            basin_areas=basin_areas, 
            laterals=laterals,
            laterals_data=laterals_data,
            saveat=saveat
        )
        print('laterals: based on lateral inflow according to dhydro network')
    elif method_laterals == 2:
        if laterals_areas_data is None:
            raise ValueError("method_laterals = 2 requires laterals_areas_data")
        tables['basin_time'] = generate_basin_time_table_laterals_areas_data(
            basins=basins, 
            areas=areas, 
            laterals_areas_data=laterals_areas_data
        )
        print('laterals: based on lateral inflow (timeseries) per area')
    elif method_laterals == 3:
        if laterals_drainage_per_ha is None:
            raise ValueError("method_laterals = 3 requires laterals_drainage_per_ha")
        tables['basin_time'] = generate_basin_time_table_laterals_drainage_per_ha(
            basins=basins, 
            basin_areas=basin_areas, 
            laterals_drainage_per_ha=laterals_drainage_per_ha
        )
        print('laterals: based on homogeneous lateral inflow timeseries')
    else:
        raise ValueError('method_laterals should be 1, 2 or 3')

    # create tables for BOUNDARIES
    if method_boundaries and boundaries_timeseries_data is not None:
        flow_boundaries = boundaries[boundaries['ribasim_type']=="FlowBoundary"]
        print(f'flowboundaries: based on timeseries ({len(flow_boundaries)} flowboundaries)')
        tables['flow_boundary_time'] = generate_boundary_time_table(
            flow_boundaries, 
            boundaries_timeseries_data,
            boundary_type='flow_rate',
        )  
        tables['flow_boundary_static'] = None

        level_boundaries = boundaries[boundaries['ribasim_type']=="LevelBoundary"]
        print(f'levelboundaries: based on timeseries ({len(level_boundaries)} levelboundaries)')
        tables['level_boundary_time'] = generate_boundary_time_table(
            level_boundaries, 
            boundaries_timeseries_data,
            boundary_type='level',
        )  
        tables['level_boundary_static'] = None
    else:
        # oud
        level_boundaries = boundaries[boundaries['ribasim_type']=="LevelBoundary"]
        tables['level_boundary_static'] = pd.DataFrame(
            data={
                "node_id": level_boundaries["boundary_node_id"],
                "level": [7.15] * len(level_boundaries),
            }
        )
        tables['level_boundary_time'] = None

        flow_boundaries = boundaries[boundaries['ribasim_type']=="FlowBoundary"]
        tables['flow_boundary_static'] = pd.DataFrame(
            data={
                "node_id": flow_boundaries["boundary_node_id"],
                "flow_rate": [0.0] * len(flow_boundaries),
            }
        )
        tables['flow_boundary_time'] = None
        
    # create tables for PUMPS
    pumps = split_nodes[split_nodes['ribasim_type'] == 'Pump']
    print(f"pumps: generated ({len(pumps)} pumps)")
    tables['pump_static'] = pd.DataFrame(
        data={
            "node_id": pumps["split_node_node_id"],
            "flow_rate": [0] * len(pumps),
        }
    )

    # create tables for OUTLETS
    outlets = split_nodes[split_nodes['ribasim_type'] == 'Outlet']
    print(f"outlets: generated ({len(outlets)} outlets)")
    tables['outlet_static'] = pd.DataFrame(
        data={
            "node_id": outlets["split_node_node_id"],
            "flow_rate": [0] * len(outlets),
        }
    )

    # create tables for TABULATED RATING CURVES
    tabulated_rating_curves = split_nodes[split_nodes['ribasim_type'] == 'TabulatedRatingCurve']
    print(f"tabulated_rating_curves: generated ({len(tabulated_rating_curves)} tabulated_rating_curves)")
    tables['tabulated_rating_curve_static'] = generate_tabulated_rating_curve(
        basins_outflows, tabulated_rating_curves, 
        basin_h, edge_q_df, weir_q_df, uniweir_q_df, 
        orifice_q_df, culvert_q_df, bridge_q_df, pump_q_df, set_name
    )
    
    # create tables for MANNINGRESISTANCE
    manningresistance = split_nodes[split_nodes['ribasim_type'] == 'ManningResistance']
    print(f"manningresistance: generated ({len(manningresistance)} manningresistance)")
    tables['manningresistance_static'] = generate_manning_resistances(manningresistance, set_name)

    return tables
