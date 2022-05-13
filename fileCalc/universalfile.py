
import numpy as np
np.seterr(all='raise')

import pandas as pd

from dash import dcc
from dash import html

from dash import dash_table
import dash_bootstrap_components as dbc

IMHOTEP_LOGO = "https://imhotep.industries/wp-content/uploads/Imhotep-Industries-Logo-lang.png"
HOME_URL = "https://imhotep.industries/"

def create_navbar():
    """
    Creates the navigation bar at the top of the page

    Returns
    -------
    A navigation bar from dash_bootstrap_components
    """

    navbar = dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src=IMHOTEP_LOGO, height="30px")),
                            dbc.Col(dbc.NavbarBrand("IMHOTEP", className="ms-2")),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="https://imhotep.industries/",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                dbc.Collapse(

                    id="navbar-collapse",
                    is_open=False,
                    navbar=True,
                ),
            ]
        ),
        color="primary",
        dark=True,
    )

    return navbar
def create_output_table(df):
    columns = [

        {"name": "Hours (h)",
         "id": "Hours (h)", "type": "numeric", "format": {'specifier': ','}},
        {"name": "Temperature (C)",
         "id": "Temperature (C)", "type": "numeric", "format": {'specifier': ','}},
        {"name": "Humidity (%)",
         "id": "Humidity (%)", "type": "numeric", "format": {'specifier': ','}},
        {"name": "Output [l/24h]",
         "id": "Output [l/24h]", "type": "numeric", "format": {'specifier': ','}},
        {"name": "Efficiency[Wh/l]",
         "id": "Efficiency[Wh/l]", "type": "numeric", "format": {'specifier': ','}},
        {"name": "Total output[Liter]",
         "id": "Total output[Liter]", "type": "numeric", "format": {'specifier': ','}},

        {"name": "Energy Consumption[kWh]",
         "id": "Energy Consumption[kWh]", "type": "numeric", "format": {'specifier': ','}}

    ]

    data = df.sort_values("Hours (h)", ascending=False).to_dict("records")

    return dash_table.DataTable(
        id='new-table',
        columns=columns,
        data=data,

        active_cell={"row": 0, "column": 0},

        derived_virtual_data=data,
        style_table={
            "minHeight": "80vh",
            "height": "80vh",
            "overflowY": "scroll",
            "overflowX": "scroll"
        },
        style_cell={
            "whitespace": "normal",
            "height": "auto",
            "width": "auto",
            "fontFamily": "verdana",


        },
        style_header={
            "textAlign": "center",
            "height": "auto",
            "width": "auto",
            "fontSize": 14
        },
        style_data={
            "fontSize": 10,
            'fontWeight': 'bold'
        },
        fill_width=False
        ,
        style_data_conditional=[
            {
                "if": {"column_id": "country"},
                "width": "80px",
                "textAlign": "left",
                "textDecoration": "underline",
                "cursor": "pointer"
            },
            {
                "if": {"column_id": "place"},
                "width": "100px",
                "textAlign": "left",
                "cursor": "pointer"
            },
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#fafbfb"
            }
        ],
    )


def universal_df(df_csv):
    # define constants
    np.seterr(all='warn')

    MINIMUM_TEMP = 12.6
    MAXIMUM_TEMP = 40
    TEMPXHUMIDITY = 1200

    # add column header
    columns = ['Year', 'Month', 'Day', 'Hour', 'Temperature', 'rainfall', 'rel. Humidity',
               'dew point', 'surface temperature', 'air pressure', 'Solar radiation']
    df_csv.columns = ['Year', 'Month', 'Day', 'Hour', 'Temperature', 'rainfall',
                      'rel. Humidity', 'dew point', 'surface temperature',
                      'air pressure', 'Solar radiation']

    # new temperature

    def product_temperature_humidity(df, temperature, humidity):
        df["tempxhumidity"] = df[temperature] * df[humidity]

        return df["tempxhumidity"].to_list(), df

    tempxhumidity, df_with_tempxhumidity = product_temperature_humidity(
        df_csv, 'Temperature', 'rel. Humidity')

    # getting the new temperature

    def new_temperature(df, temperature, tempxhumidity):
        conditions = [df[temperature] < MINIMUM_TEMP,
                      df[temperature] > MAXIMUM_TEMP, df[tempxhumidity] < TEMPXHUMIDITY]
        choices = [0, 0, 0]
        df["new_temperature"] = np.select(
            conditions, choices, default=df[temperature])

        return df["new_temperature"].to_list(), df

    new_temperature, df_new_temp = new_temperature(
        df_with_tempxhumidity, "Temperature", "tempxhumidity")

    # getting the new temperature between 12.5 and 13.75

    def temperature_125(df, temperature):
        conditions = [df[temperature] < 12.5,  df[temperature] < 13.75]
        choices = [0, df[temperature]]
        df["temp_125"] = np.select(
            conditions, choices, default=0)

        return df["temp_125"].to_list(), df

    temp_125, df_new_temp_125 = temperature_125(
        df_with_tempxhumidity, "new_temperature")

    # getting the new humidity list between 12.5 and 13.75

    def humidity_125(df, temp_125, humidity):
        conditions = [df[temp_125] > 0]
        choices = [df[humidity]]
        df["humidity_125"] = np.select(
            conditions, choices, default=0)

        return df["humidity_125"].to_list(), df

    humidity_125, df_new_humidity_125 = humidity_125(
        df_new_temp_125, "temp_125", "rel. Humidity")

    # getting the new temperature between 13.75 and 16.25

    def temperature_1375(df, temperature):
        conditions = [df[temperature] < 13.75,  df[temperature] < 16.25]
        choices = [0, df[temperature]]
        df["temp_1375"] = np.select(
            conditions, choices, default=0)

        return df["temp_1375"].to_list(), df

    temp_1375, df_new_temp_1375 = temperature_1375(
        df_new_humidity_125, "new_temperature")

    # getting the new humidity list between 13.75 and 16.25

    def humidity_1375(df, temp_1375, humidity):
        conditions = [df[temp_1375] > 0]
        choices = [df[humidity]]
        df["humidity_1375"] = np.select(
            conditions, choices, default=0)

        return df["humidity_1375"].to_list(), df

    humidity_1375, df_new_humidity_1375 = humidity_1375(
        df_new_temp_1375, "temp_1375", "rel. Humidity")

    # getting the new temperature between 16.25 and 18.75

    def temperature_1625(df, temperature):
        conditions = [df[temperature] < 16.25,  df[temperature] < 18.75]
        choices = [0, df[temperature]]
        df["temp_1625"] = np.select(
            conditions, choices, default=0)

        return df["temp_1625"].to_list(), df

    temp_1625, df_new_temp_1625 = temperature_1625(
        df_new_humidity_1375, "new_temperature")

    # getting the new humidity list between 16.25 and 18.75

    def humidity_1625(df, temp_1625, humidity):
        conditions = [df[temp_1625] > 0]
        choices = [df[humidity]]
        df["humidity_1625"] = np.select(
            conditions, choices, default=0)

        return df["humidity_1625"].to_list(), df

    humidity_1625, df_new_humidity_1625 = humidity_1625(
        df_new_temp_1625, "temp_1625", "rel. Humidity")

    # getting the new temperature between 18.75 and 21.25

    def temperature_1875(df, temperature):
        conditions = [df[temperature] < 18.75,  df[temperature] < 21.25]
        choices = [0, df[temperature]]
        df["temp_1875"] = np.select(
            conditions, choices, default=0)

        return df["temp_1875"].to_list(), df

    temp_1875, df_new_temp_1875 = temperature_1875(
        df_new_humidity_1625, "new_temperature")

    # getting the new humidity list between 18.75 and 21.25

    def humidity_1875(df, temperature, humidity):
        conditions = [df[temperature] > 0]
        choices = [df[humidity]]
        df["humidity_1875"] = np.select(
            conditions, choices, default=0)

        return df["humidity_1875"].to_list(), df

    humidity_1875, df_new_humidity_1875 = humidity_1875(
        df_new_temp_1875, "temp_1875", "rel. Humidity")

    # getting the new temperature between 21.25 and 23.75

    def temperature_2125(df, temperature):
        conditions = [df[temperature] < 21.25,  df[temperature] < 23.75]
        choices = [0, df[temperature]]
        df["temp_2125"] = np.select(
            conditions, choices, default=0)

        return df["temp_2125"].to_list(), df

    temp_2125, df_new_temp_2125 = temperature_2125(
        df_new_humidity_1875, "new_temperature")

    # getting the new humidity list between 18.75 and 21.25

    def humidity_2125(df, temperature, humidity):
        conditions = [df[temperature] > 0]
        choices = [df[humidity]]
        df["humidity_2125"] = np.select(
            conditions, choices, default=0)

        return df["humidity_2125"].to_list(), df

    humidity_2125, df_new_humidity_2125 = humidity_2125(
        df_new_temp_2125, "temp_2125", "rel. Humidity")

    # getting the new temperature between 23.75 and 26.25

    def temperature_2375(df, temperature):
        conditions = [df[temperature] < 23.75,  df[temperature] < 26.25]
        choices = [0, df[temperature]]
        df["temp_2375"] = np.select(
            conditions, choices, default=0)

        return df["temp_2375"].to_list(), df

    temp_2375, df_new_temp_2375 = temperature_2375(
        df_new_humidity_2125, "new_temperature")

    # getting the new humidity list between 18.75 and 21.25

    def humidity_2375(df, temperature, humidity):
        conditions = [df[temperature] > 0]
        choices = [df[humidity]]
        df["humidity_2375"] = np.select(
            conditions, choices, default=0)

        return df["humidity_2375"].to_list(), df

    humidity_2375, df_new_humidity_2375 = humidity_2375(
        df_new_temp_2375, "temp_2375", "rel. Humidity")

    # getting the new temperature between 26.25 and 28.75

    def temperature_2625(df, temperature):
        conditions = [df[temperature] < 26.25,  df[temperature] < 28.75]
        choices = [0, df[temperature]]
        df["temp_2625"] = np.select(
            conditions, choices, default=0)

        return df["temp_2625"].to_list(), df

    temp_2625, df_new_temp_2625 = temperature_2625(
        df_new_humidity_2375, "new_temperature")

    # getting the new humidity list between 26.25 and 28.75

    def humidity_2625(df, temperature, humidity):
        conditions = [df[temperature] > 0]
        choices = [df[humidity]]
        df["humidity_2625"] = np.select(
            conditions, choices, default=0)

        return df["humidity_2625"].to_list(), df

    humidity_2625, df_new_humidity_2625 = humidity_2625(
        df_new_temp_2625, "temp_2625", "rel. Humidity")

    # getting the new temperature between  28.75 and 31.25

    def temperature_2875(df, temperature):
        conditions = [df[temperature] < 28.75,  df[temperature] < 31.25]
        choices = [0, df[temperature]]
        df["temp_2875"] = np.select(
            conditions, choices, default=0)

        return df["temp_2875"].to_list(), df

    temp_2875, df_new_temp_2875 = temperature_2875(
        df_new_humidity_2375, "new_temperature")

    # getting the new humidity list between 26.25 and 28.75

    def humidity_2875(df, temperature, humidity):
        conditions = [df[temperature] > 0]
        choices = [df[humidity]]
        df["humidity_2875"] = np.select(
            conditions, choices, default=0)

        return df["humidity_2875"].to_list(), df

    humidity_2875, df_new_humidity_2875 = humidity_2875(
        df_new_temp_2875, "temp_2875", "rel. Humidity")

    # getting the new temperature between  31.25 and 33.75

    def temperature_3125(df, temperature):
        conditions = [df[temperature] < 31.25,  df[temperature] < 33.75]
        choices = [0, df[temperature]]
        df["temp_3125"] = np.select(
            conditions, choices, default=0)

        return df["temp_3125"].to_list(), df

    temp_3125, df_new_temp_3125 = temperature_3125(
        df_new_humidity_2875, "new_temperature")

    # getting the new humidity list between 26.25 and 28.75

    def humidity_3125(df, temperature, humidity):
        conditions = [df[temperature] > 0]
        choices = [df[humidity]]
        df["humidity_3125"] = np.select(
            conditions, choices, default=0)

        return df["humidity_3125"].to_list(), df

    humidity_3125, df_new_humidity_3125 = humidity_3125(
        df_new_temp_3125, "temp_3125", "rel. Humidity")

    # getting the new temperature between  31.25 and 33.75

    def temperature_3375(df, temperature):
        conditions = [df[temperature] < 33.75,  df[temperature] < 36.25]
        choices = [0, df[temperature]]
        df["temp_3375"] = np.select(
            conditions, choices, default=0)

        return df["temp_3375"].to_list(), df

    temp_3375, df_new_temp_3375 = temperature_3375(
        df_new_humidity_3125, "new_temperature")
    # getting the new humidity list between 26.25 and 28.75

    def humidity_3375(df, temperature, humidity):
        conditions = [df[temperature] > 0]
        choices = [df[humidity]]
        df["humidity_3375"] = np.select(
            conditions, choices, default=0)

        return df["humidity_3375"].to_list(), df

    humidity_3375, df_new_humidity_3375 = humidity_3375(
        df_new_temp_3375, "temp_3375", "rel. Humidity")

    # getting the new temperature between  36.25 and 38.75

    def temperature_3625(df, temperature):
        conditions = [df[temperature] < 36.25,  df[temperature] < 38.75]
        choices = [0, df[temperature]]
        df["temp_3625"] = np.select(
            conditions, choices, default=0)

        return df["temp_3625"].to_list(), df

    temp_3625, df_new_temp_3625 = temperature_3625(
        df_new_humidity_3375, "new_temperature")

    # getting the new humidity list between 26.25 and 28.75

    def humidity_3625(df, temperature, humidity):
        conditions = [df[temperature] > 0]
        choices = [df[humidity]]
        df["humidity_3625"] = np.select(
            conditions, choices, default=0)

        return df["humidity_3625"].to_list(), df

    humidity_3625, df_new_humidity_3625 = humidity_3625(
        df_new_temp_3625, "temp_3625", "rel. Humidity")

    # getting the new temperature between  38.75 and 40

    def temperature_3875(df, temperature):
        conditions = [df[temperature] < 38.75,  df[temperature] < 40]
        choices = [0, df[temperature]]
        df["temp_3875"] = np.select(
            conditions, choices, default=0)

        return df["temp_3875"].to_list(), df

    temp_3875, df_new_temp_3875 = temperature_3875(
        df_new_humidity_3625, "new_temperature")

    # getting the new humidity list between 38.75 and 40

    def humidity_3875(df, temperature, humidity):
        conditions = [df[temperature] > 0]
        choices = [df[humidity]]
        df["humidity_3875"] = np.select(
            conditions, choices, default=0)

        return df["humidity_3875"].to_list(), df

    humidity_3875, df_new_humidity_3875 = humidity_3875(
        df_new_temp_3875, "temp_3875", "rel. Humidity")

    temperature_val95 = np.array(
        [12.50, 15.00, 17.50, 20.00, 22.50, 25.00, 27.50, 30.00, 32.50, 35.00, 37.50, 40.00])
    temperature_val85 = np.array(
        [15.00, 17.50, 20.00, 22.50, 25.00, 27.50, 30.00, 32.50, 35.00, 37.50, 40.00])
    temperature_val75 = np.array(
        [15.00, 17.50, 20.00, 22.50, 25.00, 27.50, 30.00, 32.50, 35.00, 37.50, 40.00])
    temperature_val65 = np.array(
        [20.00, 22.50, 25.00, 27.50, 30.00, 32.50, 35.00, 37.50, 40.00])
    temperature_val55 = np.array(
        [22.50, 25.00, 27.50, 30.00, 32.50, 35.00, 37.50, 40.00])
    temperature_val45 = np.array([27.50, 30.00, 32.50, 35.00, 37.50, 40.00])
    temperature_val35 = np.array([37.50, 40.00])
    temperature_val30 = np.array([40.00])

    rel_humidity_val95 = np.array(
        [2055, 3071, 4039, 5127, 6351, 7775, 9396, 10240, 10168, 10674, 10890, 7573])

    rel_humidity_val85 = np.array(
        [2435, 3293, 4279, 5361, 6574, 7977, 8779, 9078, 8672, 9117, 6143])
    rel_humidity_val75 = np.array(
        [1821, 2550, 3454, 4418, 5522, 6603, 7641, 7769, 8067, 7629, 4938])
    rel_humidity_val65 = np.array(
        [2603, 3449, 4416, 5469, 6567, 6699, 6689, 6685, 4634])
    rel_humidity_val55 = np.array(
        [2479, 3308, 4228, 5201, 6334, 6981, 6316, 4764])
    rel_humidity_val45 = np.array([2936, 3754, 4683, 5711, 6274, 4428])
    rel_humidity_val35 = np.array([4747, 4186])
    rel_humidity_val30 = np.array([3717])

    etemperature_val95 = np.array(
        [12.50, 15.00, 17.50, 20.00, 22.50, 25.00, 27.50, 30.00, 32.50, 35.00, 37.50, 40.00])
    etemperature_val85 = np.array(
        [15.00, 17.50, 20.00, 22.50, 25.00, 27.50, 30.00, 32.50, 35.00, 37.50, 40.00])
    etemperature_val75 = np.array(
        [15.00, 17.50, 20.00, 22.50, 25.00, 27.50, 30.00, 32.50, 35.00, 37.50, 40.00])
    etemperature_val65 = np.array(
        [20.00, 22.50, 25.00, 27.50, 30.00, 32.50, 35.00, 37.50, 40.00])
    etemperature_val55 = np.array(
        [22.50, 25.00, 27.50, 30.00, 32.50, 35.00, 37.50, 40.00])
    etemperature_val45 = np.array([27.50, 30.00, 32.50, 35.00, 37.50, 40.00])
    etemperature_val35 = np.array([37.50, 40.00])
    etemperature_val30 = np.array([40.00])

    erel_humidity_val95 = np.array(
        [427, 327, 289, 285, 282, 260, 260, 211, 210, 194, 183, 206])
    erel_humidity_val85 = np.array(
        [394, 328, 310, 301, 308, 268, 249, 253, 252, 229, 264])
    erel_humidity_val75 = np.array(
        [454, 415, 351, 329, 331, 365, 323, 319, 289, 283, 338])
    erel_humidity_val65 = np.array(
        [416, 383, 354, 352, 379, 366, 357, 338, 373])
    erel_humidity_val55 = np.array([479, 419, 389, 395, 384, 377, 378, 394])
    erel_humidity_val45 = np.array([484, 467, 443, 432, 425, 459])
    erel_humidity_val35 = np.array([495, 532])
    erel_humidity_val30 = np.array([582])

    # Polynomial Regression
    def polyfit(x, y, degree):
        results = {}

        coeffs = np.polyfit(x, y, degree)

        # Polynomial Coefficients
        results['polynomial'] = coeffs.tolist()

        # r-squared
        p = np.poly1d(coeffs)
        # fit values, and mean
        yhat = p(x)                         # or [p(z) for z in x]
        ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
        # or sum([ (yihat - ybar)**2 for yihat in yhat])
        ssreg = np.sum((yhat-ybar)**2)
        # or sum([ (yi - ybar)**2 for yi in y])
        sstot = np.sum((y - ybar)**2)
        results['determination'] = ssreg / sstot

        return results

    polyfit95 = polyfit(temperature_val95, rel_humidity_val95, 7)
    polyfit85 = polyfit(temperature_val85, rel_humidity_val85, 7)
    polyfit75 = polyfit(temperature_val75, rel_humidity_val75, 7)
    polyfit65 = polyfit(temperature_val65, rel_humidity_val65, 6)
    polyfit55 = polyfit(temperature_val55, rel_humidity_val55, 4)
    polyfit45 = polyfit(temperature_val45, rel_humidity_val45, 4)

    epolyfit95 = polyfit(etemperature_val95, erel_humidity_val95, 7)
    epolyfit85 = polyfit(etemperature_val85, erel_humidity_val85, 7)
    epolyfit75 = polyfit(etemperature_val75, erel_humidity_val75, 7)
    epolyfit65 = polyfit(etemperature_val65, erel_humidity_val65, 6)
    epolyfit55 = polyfit(etemperature_val55, erel_humidity_val55, 4)
    epolyfit45 = polyfit(etemperature_val45, erel_humidity_val45, 4)

    def get_energy_produced(x, humidity):
        energy_production = 0

        det195 = 2.344496535541091e-06 * x ** 7
        det295 = -0.00041928402707698027*x ** 6
        det395 = 0.031113708979578766*x ** 5
        det495 = -1.2326377462543892*x ** 4
        det595 = 27.862716563505973*x ** 3
        det695 = -353.6727744502594*x ** 2
        det795 = 2264.9878536342453*x
        coffc95 = -4957.2296042802245
        equation_det95 = det195 + det295 + det395 + \
            det495 + det595 + det695 + det795 + coffc95

        det185 = 1.3968851540355475e-05 * x ** 7
        det285 = -0.002623012810405102*x ** 6
        det385 = 0.20692103528969402*x ** 5
        det485 = -8.876348315533564*x ** 4
        det585 = 223.28480723435672*x ** 3
        det685 = -3288.6587654516597*x ** 2
        det785 = 26214.023057227456*x
        coffc85 = -86729.02097641154
        equation_det85 = det185 + det285 + det385 + \
            det485 + det585 + det685 + det785 + coffc85

        det175 = 6.2529971989610685e-06 * x ** 7
        det275 = -0.0012767874509946886 * x ** 6
        det375 = 0.11039322353046178 * x ** 5
        det475 = -5.223196380132347 * x ** 4
        det575 = 145.57222684864027 * x ** 3
        det675 = -2381.4787721511084 * x ** 2
        det775 = 21095.147027309253 * x
        coffc75 = -77377.52097934668
        equation_det75 = det175 + det275 + det375 + \
            det475 + det575 + det675 + det775 + coffc75

        det165 = -0.00012174222222563283 * x ** 6
        det265 = 0.023426625641663024 * x ** 5
        det365 = -1.841939145348268 * x ** 4
        det465 = 75.71513473393188*x ** 3
        det565 = -1715.3256423139992 * x ** 2
        det665 = 20294.869035503307*x
        coffc65 = -97535.65035229283
        equation_det65 = det165 + det265 + det365 + det465 + det565 + det665 + coffc65

        det155 = 0.011224242424246269 * x ** 4
        det255 = -1.4471515151519818 * x ** 3
        det355 = 69.68742424244526 * x ** 2
        det455 = -1487.3916666670825 * x
        coffc55 = 12273.821969700017
        equation_det55 = det155 + det255 + det355 + det455 + coffc55

        det145 = 0.00906666666664215 * x ** 4
        det245 = -1.0977777777746178 * x ** 3
        det345 = 49.72333333318251 * x ** 2
        det445 = -1006.203174600018 * x

        coffc45 = 8196.60714283278
        equation_det45 = det145 + det245 + det345 + det445 + coffc45

        det135 = 14.8*x
        coffc35 = -60
        equation_det35 = det135 + coffc35

        if humidity >= 95:
            energy_production = equation_det95
        elif humidity >= 85:
            energy_production = equation_det85
        elif humidity >= 75:
            energy_production = equation_det75
        elif humidity >= 65:
            energy_production = equation_det65
        elif humidity >= 55:
            energy_production = equation_det55
        elif humidity >= 45:
            energy_production = equation_det45
        elif humidity >= 35:
            energy_production = equation_det35
        elif humidity >= 30:
            energy_production = 3717
        else:
            energy_production = 0

        return energy_production

    def get_water_production(x, humidity):
        water_production = 0

        det195 = -0.0001974092250294437 * x ** 7
        det295 = 0.03460852287695878*x ** 6
        det395 = -2.534795712508418*x ** 5
        det495 = 100.37039799282297*x ** 4
        det595 = -2318.100146302681*x ** 3
        det695 = 31216.965121905137*x ** 2
        det795 = -226608.55758609873*x
        coffc95 = 685057.8252151513
        equation_det95 = det195 + det295 + det395 + \
            det495 + det595 + det695 + det795 + coffc95

        det185 = -0.00037878438840906485 * x ** 7
        det285 = 0.07040809829809523*x ** 6
        det385 = -5.507211879526259*x ** 5
        det485 = 234.8109908400761*x ** 4
        det585 = -5891.96291357112*x ** 3
        det685 = 87006.34329868172*x ** 2
        det785 = -699855.8430203507*x
        coffc85 = 2366685.1117692445
        equation_det85 = det185 + det285 + det385 + \
            det485 + det585 + det685 + det785 + coffc85

        det175 = -0.00014377112978561035 * x ** 7
        det275 = 0.02645644549024419 * x ** 6
        det375 = -2.049668433989134 * x ** 5
        det475 = 86.57998310711562 * x ** 4
        det575 = -2152.96854499648 * x ** 3
        det675 = 31527.748964503073 * x ** 2
        det775 = -251569.41460437252 * x
        coffc75 = 844451.7272707144
        equation_det75 = det175 + det275 + det375 + \
            det475 + det575 + det675 + det775 + coffc75

        det165 = -0.006899484444400455 * x ** 6
        det265 = 1.2179154051213736 * x ** 5
        det365 = -88.5496834183821*x ** 4
        det465 = 3391.65050255115*x ** 3
        det565 = -72147.80210018768 * x ** 2
        det665 = 808386.2580219887*x
        coffc65 = -3727022.7249404043
        equation_det65 = det165 + det265 + det365 + det465 + det565 + det665 + coffc65

        det155 = -0.10237575757579169 * x ** 4
        det255 = 9.28747474747901 * x ** 3
        det355 = -292.16378787898503 * x ** 2
        det455 = 3943.2809884599956 * x
        coffc55 = -17864.26352816869
        equation_det55 = det155 + det255 + det355 + det455 + coffc55

        det145 = -1.0304000000009672 * x ** 4
        det245 = 130.63466666681603*x ** 3
        det345 = -6183.160000008537*x ** 2
        det445 = 129859.01904783308*x

        coffc45 = -1019668.1428591344
        equation_det45 = det145 + det245 + det345 + det445 + coffc45

        det135 = -224.4*x
        coffc35 = 13162
        equation_det35 = det135 + coffc35

        if humidity >= 95:
            water_production = equation_det95
        elif humidity >= 85:
            water_production = equation_det85
        elif humidity >= 75:
            water_production = equation_det75
        elif humidity >= 65:
            water_production = equation_det65
        elif humidity >= 55:
            water_production = equation_det55
        elif humidity >= 45:
            water_production = equation_det45
        elif humidity >= 35:
            water_production = equation_det35
        elif humidity >= 30:
            water_production = 3717
        else:
            water_production = 0

        return water_production

    temperature_columns = ['temp_125', 'temp_1375', 'temp_1625', 'temp_1875', 'temp_2125', 'temp_2375',
                           'temp_2625', 'temp_2875', 'temp_3125', 'temp_3375', 'temp_3625', 'temp_3875']

    humidity_columns = ['humidity_125', 'humidity_1375', 'humidity_1625', 'humidity_1875', 'humidity_2125',
                        'humidity_2375', 'humidity_2625', 'humidity_2875', 'humidity_3125', 'humidity_3375', 'humidity_3625',
                        'humidity_3875']

    def columns_value(df, columns):
        limit = 0
        columns_count = []
        for column_name in columns:
            # Select column 'C' from the dataframe
            column = df[column_name]
            # Get count of values greater than 20 in the column 'C'
            count = column[column > limit].count()
            sum_count = df[column_name].sum()
            temp_average = sum_count / count
            columns_count.append(temp_average)
        return columns_count

    temp_average = columns_value(df_new_humidity_3875, temperature_columns)
    humidity_average = columns_value(df_new_humidity_3875, humidity_columns)

    temp_dict = {'Temperature': temp_average, 'Humidity': humidity_average}
    df_temp_humidity = pd.DataFrame(temp_dict)

    new_df_temp_humidity = df_temp_humidity.fillna(0)

    list_of_single_temperature = list(new_df_temp_humidity['Temperature'])
    list_of_single_humidity = list(new_df_temp_humidity['Humidity'])

    def water_output(temp_column, humidity_column):
        water_prod = []
        for i, j in zip(temp_column, humidity_column):
            new_water_prod = get_water_production(i, j)
            water_prod.append(new_water_prod)

        return water_prod

    def energy_output(temp_column, humidity_column):
        energy_prod = []
        for i, j in zip(temp_column, humidity_column):
            new_energy_prod = get_energy_produced(i, j)
            energy_prod.append(new_energy_prod)

        return energy_prod

    water_output = water_output(
        list_of_single_temperature, list_of_single_humidity)
    energy_output = energy_output(
        list_of_single_temperature, list_of_single_humidity)

    temp_dict = {'Temperature': temp_average, 'Humidity': humidity_average}
    df_temp_humidity = pd.DataFrame(temp_dict)

    def hours_col(df, columns):
        limit = 0
        columns_count = []
        for column_name in columns:
            # Select column 'hours' from the dataframe
            column = df[column_name]
            # Get count of values greater than 20 in the column 'C'
            count = column[column > limit].count()

            columns_count.append(count)
        return columns_count

    hours = hours_col(df_new_humidity_3875, temperature_columns)

    final_water_dict = {'Hours (h)': hours, 'Temperature (C)': list_of_single_temperature,
                        'Humidity (%)': list_of_single_humidity,
                        'Output [l/24h]': water_output, 'Efficiency[Wh/l]': energy_output}
    df_output = pd.DataFrame(final_water_dict)

    df_output["Total output[Liter]"] = df_output["Hours (h)"] * \
        df_output["Output [l/24h]"] / 24
    df_output["Energy Consumption[kWh]"] = df_output["Efficiency[Wh/l]"] * \
        df_output["Total output[Liter]"] / 1000

    total_hours = df_output['Hours (h)'].sum()
    total_water_output = df_output['Total output[Liter]'].sum()
    total_energy_consumption = df_output['Energy Consumption[kWh]'].sum()

    average_efficiency = df_output['Energy Consumption[kWh]'].sum(
    ) / df_output['Total output[Liter]'].sum()

    yields = ["max. annual water yield:", "annual power consumption:",
              "max. annual operating hours:", "average efficiency"]
    values = [total_water_output, total_energy_consumption/1000,
              total_hours, average_efficiency]
    metrics = ["litres per year", "MWh per year", "hours per year", " "]

    df_metrics = pd.DataFrame(
        {'Phantor Yields': yields,
         'Phantor Values': values,
         'Phantor Metrics': metrics
         })

    return df_new_humidity_3875, df_output,df_metrics, total_hours, total_water_output, total_energy_consumption, average_efficiency

def create_metric_table(df):
    columns = [
        {"name": "Phantor Yields",
         "id": "Phantor Yields", "type": "text"},

        {"name": "Phantor Values",
         "id": "Phantor Values", "type": "numeric", "format": {'specifier': ','}},
        {"name": "Phantor Metrics",
                 "id": "Phantor Metrics", "type": "text"}
    ]

    data = df.to_dict("records")

    return dash_table.DataTable(
        id='metric-table',
        columns=columns,
        data=data,
        fixed_rows={"headers": True},
        active_cell={"row": 0, "column": 0},
        sort_action="native",
        derived_virtual_data=data,
        style_table={
            "minHeight": "80vh",
            "height": "80vh",
            "overflowY": "scroll",
            "overflowX": "scroll"
        },
        style_cell={
            "whitespace": "normal",
            'textAlign': 'left',
            "height": "auto",

            "fontFamily": "verdana"
        },
        style_header={
            "textAlign": "center",
            "width": "auto",
            "fontSize": 14
        },
        style_data={
            "fontSize": 10,
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                "if": {"column_id": "country"},
                "width": "80px",
                "textAlign": "left",
                "textDecoration": "underline",
                "cursor": "pointer"
            },
            {
                "if": {"column_id": "place"},
                "width": "100px",
                "textAlign": "left",
                "cursor": "pointer"
            },
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#fafbfb"
            }
        ],
    )

def create_main_table(df):
    columns = [

        {"name": "Year",
         "id": "Year", "type": "numeric"},
        {"name": "Month",
         "id": "Month", "type": "numeric", "format": {'specifier': ','}},
        {"name": "Day",
         "id": "Day", "type": "numeric", "format": {'specifier': ','}},
        {"name": "Hour",
         "id": "Hour", "type": "numeric", "format": {'specifier': ','}},
        {"name": "Temperature",
         "id": "Temperature", "type": "numeric", "format": {'specifier': ','}},
        {"name": "rainfall",
         "id": "rainfall", "type": "numeric", "format": {'specifier': ','}},

        {"name": "rel. Humidity",
         "id": "rel. Humidity", "type": "numeric", "format": {'specifier': ','}},
        {"name": "dew point",
         "id": "dew point", "type": "numeric", "format": {'specifier': ','}},
        {"name": "surface temperature",
         "id": "surface temperature", "type": "numeric", "format": {'specifier': ','}},
        {"name": "air pressure",
         "id": "air pressure", "type": "numeric", "format": {'specifier': ','}},
        {"name": "Solar radiation",
         "id": "Solar radiation", "type": "numeric", "format": {'specifier': ','}},
        {"name": "tempxhumidity",
         "id": "tempxhumidity", "type": "numeric", "format": {'specifier': ','}},
        {"name": "new_temperature",
         "id": "new_temperature", "type": "numeric", "format": {'specifier': ','}},

        {"name": "temp_125",
         "id": "temp_125", "type": "numeric", "format": {'specifier': ','}},

        {"name": "humidity_125",
         "id": "humidity_125", "type": "numeric", "format": {'specifier': ','}},
        {"name": "temp_1375",
         "id": "temp_1375", "type": "numeric", "format": {'specifier': ','}},
        {"name": "humidity_1375",
         "id": "humidity_1375", "type": "numeric", "format": {'specifier': ','}},
        {"name": "temp_1625",
         "id": "temp_1625", "type": "numeric", "format": {'specifier': ','}},
        {"name": "humidity_1625",
         "id": "humidity_1625", "type": "numeric", "format": {'specifier': ','}},
        {"name": "temp_1875",
         "id": "temp_1875", "type": "numeric", "format": {'specifier': ','}},

        {"name": "humidity_1875",
         "id": "humidity_1875", "type": "numeric", "format": {'specifier': ','}},

        {"name": "temp_2125",
         "id": "temp_2125", "type": "numeric", "format": {'specifier': ','}},
        {"name": "humidity_2125",
         "id": "humidity_2125", "type": "numeric", "format": {'specifier': ','}},
        {"name": "temp_2375",
         "id": "temp_2375", "type": "numeric", "format": {'specifier': ','}},
        {"name": "humidity_2375",
         "id": "humidity_2375", "type": "numeric", "format": {'specifier': ','}},
        {"name": "temp_2625",
         "id": "temp_2625", "type": "numeric", "format": {'specifier': ','}},
        {"name": "humidity_2625",
         "id": "humidity_2625", "type": "numeric", "format": {'specifier': ','}},

        {"name": "temp_2875",
         "id": "temp_2875", "type": "numeric", "format": {'specifier': ','}},

        {"name": "humidity_2875",
         "id": "humidity_2875", "type": "numeric", "format": {'specifier': ','}},
        {"name": "temp_3125",
         "id": "temp_3125", "type": "numeric", "format": {'specifier': ','}},
        {"name": "humidity_3125",
         "id": "humidity_3125", "type": "numeric", "format": {'specifier': ','}},
        {"name": "temp_3375",
         "id": "temp_3375", "type": "numeric", "format": {'specifier': ','}},
        {"name": "humidity_3375",
         "id": "humidity_3375", "type": "numeric", "format": {'specifier': ','}},
        {"name": "temp_3625",
         "id": "temp_3625", "type": "numeric", "format": {'specifier': ','}},
        {"name": "humidity_3625",
         "id": "humidity_3625", "type": "numeric", "format": {'specifier': ','}},

        {"name": "temp_3875",
         "id": "temp_3875", "type": "numeric", "format": {'specifier': ','}},
        {"name": "humidity_3875",
         "id": "humidity_3875", "type": "numeric", "format": {'specifier': ','}}

    ]

    data = df.to_dict("records")

    return dash_table.DataTable(
        id='main-table',
        columns=columns,
        data=data,
        active_cell={"row": 0, "column": 0},
        derived_virtual_data=data,
        style_table={
            "minHeight": "80vh",
            "height": "80vh",
            "overflowY": "scroll",
            "overflowX": "scroll"
        },
        style_cell={
            "whitespace": "normal",
            "height": "auto",
            "width": "auto",
            "fontFamily": "verdana"
        },
        style_header={
            "textAlign": "center",
            "width": "auto",
            "fontSize": 14
        },
        style_data={
            "fontSize": 10,
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                "if": {"column_id": "Year"},
                "width": "80px",
                "textAlign": "left",
                "textDecoration": "underline",
                "cursor": "pointer"
            },
            {
                "if": {"column_id": "place"},
                "width": "100px",
                "textAlign": "left",
                "cursor": "pointer"
            },
            {
                "if": {"row_index": "odd"},
                "backgroundColor": "#fafbfb"
            }
        ],
    )
