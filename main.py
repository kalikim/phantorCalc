import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from fileCalc.universalfile import *


import dash_bootstrap_components as dbc

import pandas as pd

# URLs
IMHOTEP_LOGO = "https://imhotep.industries/wp-content/uploads/Imhotep-Industries-Logo-lang.png"
HOME_URL = "https://imhotep.industries/"



app = dash.Dash(__name__, external_stylesheets=[
        dbc.themes.MATERIA
    ],
                suppress_callback_exceptions=True)

navbar = create_navbar()

app.layout = html.Div([ navbar,dbc.Row(dbc.Col(html.H2("   PHANTOR CALCULATOR"), width={"size": 6, "offset": 3}, )
                    ),

    # this code section taken from Dash docs https://dash.plotly.com/dash-core-components/upload
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop  or ',
            html.A('Select CSV File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-div'),
    html.Div(id='output-datatable'),
])


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), header=None, delimiter=r"\s+")

            intial_df, output_df, df_metrics, total_hours, total_water_output, total_energy_consumption, average_efficiency = universal_df(
                df)

            print(intial_df)
            output_table = create_output_table(output_df)
            metric_table = create_metric_table(df_metrics)
            main_table = create_main_table(intial_df)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))



    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])


    return html.Div([
        dbc.Row(
            [
                dbc.Col(html.Div(), md=1),

                dbc.Col(html.Div(html.H5(filename)), md=5),
                dbc.Col(html.Div(html.H6(datetime.datetime.fromtimestamp(date))), md=5),

                dbc.Col(html.Div(), md=1),
            ])
        ,
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(html.Div(), md=1),

                dbc.Col(html.Div(dbc.Card(
        dbc.CardBody(
            [
                html.H5("OUTPUT TABLE", className="card-title"),
                output_table,

            ]
        )
    )

                    ), md=5),
                dbc.Col(html.Div(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("OVERALL OUTPUT", className="card-title"),
                                metric_table,

                            ]
                        )
                    )

                    ), md=5),


                dbc.Col(html.Div(), md=1),
            ]),
        dbc.Row(
            [
                dbc.Col(html.Div(), md=1),

                dbc.Col(html.Div(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("MAIN TABLE", className="card-title"),
                                main_table,

                            ]
                        )
                    )


                    ), md=10),

                dbc.Col(html.Div(), md=1),
            ]),
        dcc.Store(id='stored-data', data=df.to_dict('records')),



       ##tabless
        ##dcc.Store(id='stored-data', data=df.to_dict('records')),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser

        dbc.Card(
        dbc.CardBody(
            [
                html.H5("Phantor World Map", className="card-title"),
                html.P(
                    "In response to the increasing demand for new ways of producing drinking water and based on the latest scientific findings, we developed an atmospheric water generator."
                ),

            ]
        )
    )
    ])


@app.callback(Output('output-datatable', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children



navbar = create_navbar()


if __name__ == '__main__':
    app.run_server(host="localhost", port=8000, debug=False)
