# Import all the neccesary libraries 
import yaml
import pandas as pd
import os
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from glob import glob
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')
tqdm.pandas(desc="Progress!")

# Function to read the configuration file
def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


# Function to track the position of the truck
def truck_position(df, name):
    plant_1 = Polygon([(9.489763, 47.660213), (9.491629, 47.661182), (9.492552, 47.660907), (9.494827, 47.660633),
                       (9.497251, 47.658797), (9.490556, 47.655126), (9.487123, 47.653854), (9.483626, 47.655834),
                       (9.485106, 47.657482), (9.48766, 47.659231), (9.489763, 47.660213)])
    plant_2 = Polygon([(9.466138, 47.667208), (9.46352, 47.667251), (9.462512, 47.661471), (9.464314, 47.658335),
                       (9.473004, 47.658581), (9.473948, 47.662439), (9.471889, 47.664925), (9.466138, 47.667208)])

    location = []
    logic = []

    flag = 1
    value = ''

    print('*****************************************************************************************************')
    print('Finding the position of the truck')
    print('*****************************************************************************************************')

    for row in tqdm(df.to_dict('records'), desc=name):
        if plant_2.contains(Point(row['lon'], row['lat'])):
            if flag == 0:
                logic.append(1)
            else:
                logic.append(0)
            location.append('2')
            value = '2-road'
            flag = 1

        elif plant_1.contains(Point(row['lon'], row['lat'])):
            if flag == 0:
                logic.append(1)
            else:
                logic.append(0)
            location.append('1')
            value = '1-road'
            flag = 1

        else:
            location.append(value)
            if flag == 1:
                logic.append(1)
                flag = 0
            else:
                logic.append(0)

    df['location'] = location
    df['logic'] = logic

    return df


def travel_time_less_3(new_df):
    indexes = []
    for index, row in new_df.iterrows():
        if index < len(new_df) - 1:
            if index % 2 != 0:
                if (new_df.iloc[index, :]['Time_stamp'] - new_df.iloc[index - 1, :]['Time_stamp']) < pd.Timedelta(
                        minutes=3):
                    indexes.append(index)
                    indexes.append(index - 1)
    new_df.drop(new_df.index[indexes], inplace=True)
    new_df.reset_index(drop=True, inplace=True)
    return new_df


def travel_time_information(new_df):
    flag1 = 0
    flag2 = 1
    flag3 = 0
    flag4 = 1
    start1 = []
    end1 = []
    start2 = []
    end2 = []
    final_df1 = pd.DataFrame()
    final_df2 = pd.DataFrame()

    for index, row in new_df.iterrows():
        if flag1 == 0:
            if row['location'] == '2-road':
                start2.append(row['Time_stamp'])
                flag1 = 1
                flag2 = 0
            continue

        if flag2 == 0:
            if row['location'] == '1':
                end1.append(row['Time_stamp'])
                flag2 = 1
                flag1 = 0
            continue

    for index, row in new_df.iterrows():
        if flag3 == 0:
            if row['location'] == '1-road':
                start1.append(row['Time_stamp'])
                flag3 = 1
                flag4 = 0
            continue

        if flag4 == 0:
            if row['location'] == '2':
                end2.append(row['Time_stamp'])
                flag4 = 1
                flag3 = 0
            continue

    final_df1['start_plant1'] = start1
    final_df1['end_plant2'] = end2

    final_df2['start_plant2'] = start2
    final_df2['end_plant1'] = end1

    final_df1['travel_time'] = final_df1['end_plant2'] - final_df1['start_plant1']
    final_df2['travel_time'] = final_df2['end_plant1'] - final_df2['start_plant2']

    final_df1['travel_time'] = final_df1['travel_time'].apply(lambda x: x.total_seconds() / 60)
    final_df2['travel_time'] = final_df2['travel_time'].apply(lambda x: x.total_seconds() / 60)

    return final_df1, final_df2


# Function to get the GPS and Speed inforamtion of the travel time
def fetch_gps_and_speed_infromation(config_path, final_df1, final_df2, df):
    print('*****************************************************************************************************')
    print('Fetching GPS and Speed Information')
    print('*****************************************************************************************************')
    final_df2['GPS_2_1_lat'] = final_df2.progress_apply(
        lambda x: df[(df['Time_stamp'].between(x['start_plant2'], x['end_plant1']))]['lat'].values, axis=1)
    final_df2['GPS_2_1_lon'] = final_df2.progress_apply(
        lambda x: df[(df['Time_stamp'].between(x['start_plant2'], x['end_plant1']))]['lon'].values, axis=1)
    final_df1['GPS_1_2_lat'] = final_df1.progress_apply(
        lambda x: df[(df['Time_stamp'].between(x['start_plant1'], x['end_plant2']))]['lat'].values, axis=1)
    final_df1['GPS_1_2_lon'] = final_df1.progress_apply(
        lambda x: df[(df['Time_stamp'].between(x['start_plant1'], x['end_plant2']))]['lon'].values, axis=1)
    final_df2['speed_2_1'] = final_df2.progress_apply(
        lambda x: df[(df['Time_stamp'].between(x['start_plant2'], x['end_plant1']))][
            'WheelBasedVehicleSpeed'].values,
        axis=1)
    final_df1['speed_1_2'] = final_df1.progress_apply(
        lambda x: df[(df['Time_stamp'].between(x['start_plant1'], x['end_plant2']))][
            'WheelBasedVehicleSpeed'].values,
        axis=1)
    final_df1['Average_Speed'] = final_df1['speed_1_2'].progress_apply(
        lambda x: np.mean([i for i in x]))
    final_df2['Average_Speed'] = final_df2['speed_2_1'].progress_apply(
        lambda x: np.mean([i for i in x]))

# Function to find the route information taken by the truck
def fetch_route_information(config_path, final_df1, final_df2, name):
    config = read_params(config_path)
    gps_preprocessed_data1_path = config["data_source"]["gps_preprocessed_data1"]
    gps_preprocessed_data2_path = config["data_source"]["gps_preprocessed_data2"]
    gps_preprocessed_data1_cols = config["data_source"]["gps_preprocessed_data1_cols"]
    gps_preprocessed_data2_cols = config["data_source"]["gps_preprocessed_data2_cols"]
    routes_2_1 = []
    routes_1_2 = []
    route_1 = Polygon([(9.484763, 47.658797), (9.481587, 47.660994), (9.478283, 47.663277), (9.482145, 47.664867),
                       (9.487467, 47.661168), (9.484763, 47.658797)])
    route_2 = Polygon([(9.474764, 47.658277), (9.4806, 47.658971), (9.481115, 47.657815), (9.475107, 47.656832),
                       (9.474764, 47.658277)])
    route_3 = Polygon([(9.485664, 47.665792), (9.490042, 47.668624), (9.496651, 47.665503), (9.497509, 47.661977),
                       (9.494247, 47.660763), (9.485664, 47.665792)])
    route_4 = Polygon([(9.475193, 47.669665), (9.476137, 47.665908), (9.48266, 47.666312), (9.482231, 47.669896),
                       (9.475193, 47.669665)])

    for index, row in final_df2.iterrows():
        for i in range(len(final_df2['GPS_2_1_lat'][index])):
            if route_4.contains(Point(final_df2['GPS_2_1_lon'][index][i], final_df2['GPS_2_1_lat'][index][i])):
                routes_2_1.append(4)
                break
            elif route_3.contains(Point(final_df2['GPS_2_1_lon'][index][i], final_df2['GPS_2_1_lat'][index][i])):
                routes_2_1.append(3)
                break
            elif route_2.contains(Point(final_df2['GPS_2_1_lon'][index][i], final_df2['GPS_2_1_lat'][index][i])):
                routes_2_1.append(2)
                break
            elif route_1.contains(Point(final_df2['GPS_2_1_lon'][index][i], final_df2['GPS_2_1_lat'][index][i])):
                routes_2_1.append(1)
                break

    for index, row in final_df1.iterrows():
        for i in range(len(final_df1['GPS_1_2_lat'][index])):
            if route_4.contains(Point(final_df1['GPS_1_2_lon'][index][i], final_df1['GPS_1_2_lat'][index][i])):
                routes_1_2.append(4)
                break
            elif route_3.contains(Point(final_df1['GPS_1_2_lon'][index][i], final_df1['GPS_1_2_lat'][index][i])):
                routes_1_2.append(3)
                break
            elif route_2.contains(Point(final_df1['GPS_1_2_lon'][index][i], final_df1['GPS_1_2_lat'][index][i])):
                routes_1_2.append(2)
                break
            elif route_1.contains(Point(final_df1['GPS_1_2_lon'][index][i], final_df1['GPS_1_2_lat'][index][i])):
                routes_1_2.append(1)
                break

    final_df2['route_2_1'] = routes_2_1
    final_df1['route_1_2'] = routes_1_2

    final_df1 = final_df1[gps_preprocessed_data1_cols]
    final_df2 = final_df2[gps_preprocessed_data2_cols]

    os.makedirs(gps_preprocessed_data1_path, exist_ok=True)
    os.makedirs(gps_preprocessed_data2_path, exist_ok=True)

    final_df1.to_csv(gps_preprocessed_data1_path + '/' + str(name) + '.csv', sep=",", index=False)
    final_df2.to_csv(gps_preprocessed_data2_path + '/' + str(name) + '.csv', sep=",", index=False)

# Function to preprocess the GPS data
def preprocess_gps_data(config_path):
    config = read_params(config_path)

    gps_cleaned_data_path = config["data_source"]["gps_cleaned_data"]
    gps_cleaned_data_cols = config["data_source"]["gps_cleaned_data_cols"]

    for path in np.sort(glob(gps_cleaned_data_path + '/*csv')):
        df = pd.read_csv(path, sep=",", usecols=gps_cleaned_data_cols, encoding='utf-8')
        df['Time_stamp'] = pd.to_datetime(df['Time_stamp'], infer_datetime_format=True)

        name = path[-11:-4]

        df = truck_position(df, name)
        new_df = df[df['logic'] == 1]
        new_df.reset_index(drop=True, inplace=True)

        new_df = travel_time_less_3(new_df)

        final_df1, final_df2 = travel_time_information(new_df)

        fetch_gps_and_speed_infromation(config_path, final_df1, final_df2, df)

        fetch_route_information(config_path, final_df1, final_df2, name)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    preprocess_gps_data(config_path=parsed_args.config)
