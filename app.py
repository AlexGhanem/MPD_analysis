# coding: utf-8

# #Setting the Stage

# We will import all necessary libraries

import pandas as pd
import geopandas as gp
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from itertools import cycle
from dash.dependencies import Input, Output
import json
from google.cloud import bigquery
from google.oauth2 import service_account

pd.options.plotting.backend = "plotly"

#Setting up the connection to the project's BigQuery SQL database
creds = service_account.Credentials.from_service_account_file("./Data/data-key-viewer.json")
project_id="dash-app-318517"
client = bigquery.Client(credentials=creds, project=project_id)

# Time to load the date!!!

districts_geo = gp.read_file('./Data/Police_Districts/Police_Districts.shp')
df_arrests = client.query("select * from mpd_dash_database.arrests").to_dataframe()
data_full = client.query("select * from mpd_dash_database.stops").to_dataframe()

#dropping na values from important columns. they each have less than 0.01% na
data_full.dropna(subset=['stop_district','stop_time','stop_duration_minutes','race_ethnicity'], inplace=True)


#removing ages = to unknown - count < 0.02%
data_full = data_full[data_full['age']!='Unknown']


#children ages are set to "Juvenile" string type. Converting that to 16. Assuming mean.
data_full['age'].replace('Juvenile',16, inplace=True)


#mapping age data to integer
data_full['age']=data_full['age'].map(int)



#creating a stop datetime column with date from both date and time columns for ease of analysis
data_full['stop_datetime'] = pd.to_datetime(data_full['stop_date']+'T'+data_full['stop_time'], format= r'%Y-%m-%dT%H:%M')



data_full['race_ethnicity'].replace(['Unknown','Multiple'],'Other',inplace=True )

#Processing the data

#grouping by district and adding metrics to the geofile

districts_geo['avg_age'] = data_full.groupby('stop_district').mean()['age'].values
districts_geo['avg_stop_duration'] = data_full.groupby('stop_district').mean()['stop_duration_minutes'].values
districts_geo['count_child'] = data_full[data_full['age']==16].groupby('stop_district').count()['stop_duration_minutes'].values
districts_geo['count_adult'] = data_full[data_full['age']>16].groupby('stop_district').count()['stop_duration_minutes'].values
districts_geo['person_searches'] = data_full[data_full['person_search_or_protective_pat_down']==1].groupby('stop_district').count()['stop_duration_minutes'].values
districts_geo['property_searches'] = data_full[data_full['property_search_or_protective_pat_down']==1].groupby('stop_district').count()['stop_duration_minutes'].values
districts_geo['person_warrant'] = data_full[data_full['person_search_warrant']==1].groupby('stop_district').count()['stop_duration_minutes'].values
districts_geo['property_warrant'] = data_full[data_full['property_search_warrant']==1].groupby('stop_district').count()['stop_duration_minutes'].values


districts_geo.index=range(1,8)



#adding a weekday column
week = ['Monday', 'Tuesday', 'Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']
def date_to_weekday(date):
  return week[date.weekday()]

data_full['weekdays'] = data_full['stop_datetime'].map(date_to_weekday)


#creating an array with the counts of stops by hour of day

df = data_full[['stop_time','stop_date']].copy()
df['stop_time'] = pd.to_datetime(df['stop_time'], format='%H:%M')
df.set_index('stop_time', inplace=True)
hourly_count = df['stop_date'].resample('H').count()


#creating an array with the counts of daily stops

df = data_full[['stop_date','stop_duration_minutes']].copy()
df['stop_date'] = pd.to_datetime(df['stop_date'], format='%Y-%m-%d')
df.set_index('stop_date', inplace=True)
daily_count = df['stop_duration_minutes'].resample('D').count()


#getting the rolling weekly average

daily_count['rolling avg'] = daily_count.rolling(7).mean()


#formating the tieme data
df_arrests['Arrest_Hour'] = pd.to_datetime(df_arrests['Arrest_Hour'],format='%H')
df_arrests['Arrest_Hour'] = df_arrests['Arrest_Hour'].dt.strftime('%I %p')


# #The Dashboard


config = {'displayModeBar':False}

template = "plotly_dark"

map_to_hist = {
    'avg_age':'age','count_child':'age','count_adult':'age',
    'person_searches':'person_search_or_protective_pat_down',
    'avg_stop_duration':'stop_duration_minutes',
    'person_warrant':'person_search_warrant','property_warrant':'property_search_warrant',
    'property_searches':'property_search_or_protective_pat_down'
    }

num_map = {'age':16,'person_search_or_protective_pat_down':1,'person_search_warrant':1,'property_search_warrant':1,'property_search_or_protective_pat_down':1}

q_dur = data_full['stop_duration_minutes'].quantile(0.99)
q_age = data_full['age'].quantile(0.99)

dropdown_style = {'background-color':'#303030'}

age_form = html.Div(
    [
        dbc.Label('Age Filter'),
        dbc.Row(
            [
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Input(
                                type="number",
                                id="data_min",
                                placeholder="min",
                                min=18,
                                max=120,
                                step=1,
                                value=18,
                                style={"width": 75,},
                            ),
                        ]
                    ),
                    width=4,
                ),
                dbc.Col(
                    dbc.FormGroup(
                        [
                            dbc.Input(
                                type="number",
                                id="data_max",
                                placeholder="max",
                                min=18,
                                max=120,
                                step=1,
                                value=120,
                                style={"width": 75},
                            ),
                        ]
                    ),
                ),
            ],
            form=True,
        ),
        dbc.FormText('Enter age Between 18 and 120'),
    ]
)


stops_content = html.Div(
    [
          dbc.Card(
                        [
                            dbc.CardHeader('Stops by Time Data'),
                            dbc.CardBody(
                                [
                                    dcc.Dropdown(
                                    id='time format',
                                    options=[
                                        {'label': 'Time of Day', 'value':'hourly'},
                                        {'label': 'Daily Stops', 'value':'daily'},
                                        {'label': 'Day of the Week', 'value':'day of the week'},
                                    ],
                                    value='daily',
                                    searchable=False,
                                    clearable=False,
                                    style={'width':'60%', 'color':'#212121'}
                                ),
                                dbc.Spinner(
                                  dcc.Graph(id='datetime', config=config),
                                  ),
                                ]
                            )
                        ],
                    ),

        dbc.Card(
            [
                dbc.CardHeader('Interactive Map and Descriptive Chart'),
                dbc.CardBody(
                    [
                        dcc.Dropdown(
                            id='district-dropdown',
                            options=[
                                {'label': 'Age of those Stopped', 'value': 'avg_age'},
                                {'label': 'Stop Duration (in mins)', 'value': 'avg_stop_duration'},
                                {'label': 'Children (<18) Stopped', 'value': 'count_child'},
                                {'label': 'Adults Stopped', 'value':'count_adult'},
                                {'label': 'Personal Patdowns or Searches', 'value': 'person_searches'},
                                {'label':'Person Search Following Warrant', 'value':'person_warrant'},
                                {'label':'Property Searches', 'value':'property_searches'},
                                {'label':'Property Search Following Warrant','value':'property_warrant'}
                            ],
                            searchable=False,
                            value='count_child',
                            clearable=False,
                            style={'width':'50%','color':'#212121'}
                        ),

                        dbc.Button('Reset To All Districts', id='d_reset', className="mr-1"),

                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        dcc.Markdown('selecting a district updates the chart'),
                                                        dbc.Spinner(
                                                                dcc.Graph(id='DC map', config=config)
                                                        ),
                                                    ]
                                                )
                                            ]
                                        )
                                    ],
                                ),
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        dbc.Spinner(
                                                            dcc.Graph(id='Hist', config=config)   
                                                        )
                                                    ]
                                                )
                                            ]
                                        )
                                    ],
                                ),
                            ]
                        )
                    ]
                )
            ]
        ),
    ]
)

arrest_content = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    age_form,
                                ]
                            ),
                        ]
                        ),
                        dbc.Card(
                        [
                            dbc.CardHeader("Selection Pie Chart"),
                            dbc.CardBody(
                                [
                                   html.Div(
                                       [
                                           html.P('Number of arrests selected: ', style={'display':'inline','font-size':'larger'}),
                                           html.P(id='select_num',style={'color':'#2A9FD6','display':'inline','font-size':'larger'})
                                       ]
                                       ),
                                   dbc.RadioItems(
                                       id='choiceopie',
                                       options=[
                                           {'label': 'By Race', 'value':1},
                                           {'label': 'By Charge', 'value': 2}
                                       ],
                                       value=1,
                                       inline=True,
                                   ),
                                   dbc.Spinner(dcc.Graph(id='sliceopie')),
                                ]
                            ),
                        ]
                        ),
                        dbc.Card(),
                    ],
                    width = 4,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                        [
                            dbc.CardHeader("DC Arrest Map"),
                            dbc.CardBody(
                                [
                                    dbc.Label('Year picker'),
                                    dcc.Slider(
                                        id='year_slider',
                                        min=2013,
                                        max=2020,
                                        value=2013,
                                        marks={
                                            2013: '2013',
                                            2014: '2014',
                                            2015: '2015',
                                            2016: '2016',
                                            2017: '2017',
                                            2018: '2018',
                                            2019: '2019',
                                            2020: '2020'
                                        },
                                        included=False,
                                    ),
                                    dbc.Alert(
                                        "Use the selection tools on the map to update the pie chart",
                                        color='secondary',
                                        dismissable=True,
                                        is_open=True,
                                        ),
                                    dbc.Alert(
                                        "Double click anywhere in the map to deselect",
                                        color='secondary',
                                        dismissable=True,
                                        is_open=True,
                                        ),
                                    dbc.Spinner(
                                        dcc.Graph(id='arrest_map')
                                    ),
                                ]
                            ),
                        ],

                    ),
                         dbc.Card(),
                    ],
                ),
            ]
        )
    ]
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

server = app.server

color_map = {'BLACK':'#150485','WHITE':'#F2A07B','UNK':'#C62A88','ASIAN':'#FF4301', 'UNKNOWN':'#C62A88'}

app.layout = dbc.Container(
  [
    html.H1("MPD Public Data Analysis", style={'text-align':'center','color':'#2A9FD6'}),
    html.Br(),
    dbc.Tabs(
      [
        dbc.Tab(label='MPD Stops', tab_id='stops'),
        dbc.Tab(label='MPD Arrests', tab_id='arrests'),
      ],
      id='tabs',
      active_tab='arrests'
    ),
    html.Div(id="tab-content")
  ]
)

@app.callback(Output('tab-content', 'children'), Input('tabs', 'active_tab'))
def render_output(active_tab):
    if (active_tab=='stops'):
        return stops_content
    elif (active_tab=='arrests'):
        return arrest_content

@app.callback(
    [Output('datetime','figure'), Output('DC map','figure')],
    [Input('district-dropdown', 'value'), Input('time format', 'value')])

def update_output(value, timeperiod):

  if(timeperiod=='hourly'):
    timeseries= px.bar(hourly_count, color_discrete_sequence=['#B14D8E'], labels={'stop_time':'','stop_date':'count' })
    timeseries.update_xaxes(tickformat='%I %p')
    timeseries.update_layout(showlegend=False)

  else:
    if(timeperiod == 'daily'):
      timeseries = px.line(daily_count,labels={'stop_date':''} ,color_discrete_sequence=['#D8A6C6'])
      timeseries.add_scatter(x=daily_count.index, y=daily_count['rolling avg']) 
      names = cycle(['Daily Count', '7-day Rolling Average'])
      timeseries.for_each_trace(lambda t:  t.update(name = next(names)))
      
    elif(timeperiod =='day of the week'):
      timeseries = px.histogram(data_full['weekdays'],
       category_orders={'value':['Monday', 'Tuesday', 'Wednesday',  'Thursday', 'Friday', 'Saturday', 'Sunday']},
       labels={'value':'days of the week'}, 
       color_discrete_sequence=['#B14D8E'], )
      timeseries.update_layout(showlegend=False)


  timeseries.update_layout({
    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    }, template= template)
  timeseries.update_traces(hovertemplate=None)

  if value =='count_child':
    labels = {'count_child':'count'}
  elif value == 'count_adult':
      labels = {'count_adult':'count'}   
  elif(value=='avg_age'):
    labels ={'avg_age': 'Average'}
  elif(value=='person_searches'):
   labels = {'person_searches': 'count'}
  elif(value=='avg_stop_duration'):
    labels = {'avg_stop_duration': 'Average'}
  elif(value=='person_warrant'):
    labels={'person_warrant':'count'}
  elif(value=='property_warrant'):
    labels={'property_warrant':'count'}
  elif(value=='property_searches'):
    labels={'property_searches':'count'}

  labels['index'] = 'District'

  fig = px.choropleth(districts_geo,geojson=districts_geo.geometry, locations=districts_geo.index,
                    color=value,projection='mercator', labels=labels,
                    color_continuous_scale=px.colors.sequential.Magenta)
  
  fig.update_geos(fitbounds="locations", visible=False)
  fig.update_layout(
      {
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',},
        dragmode=False, template=template,
        geo=dict(bgcolor= 'rgba(48,48,48,48)'), 
        clickmode='event+select'
      )
  
  return timeseries,fig

def create_histogram(metric, df, d):

  labels={'race_ethnicity':'race', 
          'stop_duration_minutes':'stop duration average',
          'property_search_or_protective_pat_down':'Property Search',
          'person_search_or_protective_pat_down':'Person Search',
          'person_search_warrant':'Person Search Warrant',
          'property_search_warrant':'Property Search Warrant',
          }

  col = map_to_hist[metric]
  if(d != None):
    title= "Viewing data from police district {}".format(d)
  else:
    title= "Viewing data from all Districts"

  if(metric == 'avg_age'):
    hist=px.histogram(
        df,col,color='race_ethnicity',
        color_discrete_sequence=px.colors.sequential.Magenta,labels=labels)
  elif(metric=='avg_stop_duration'):
    df1=df.groupby('race_ethnicity').mean()
    hist=px.bar(
        df1,df1.index,'stop_duration_minutes',
        color_discrete_sequence=px.colors.sequential.Magenta,labels=labels)
  elif(metric!='count_adult'):
    hist = px.pie(df[df[col]==num_map[col]], 
              names='race_ethnicity',
              color_discrete_sequence=px.colors.sequential.Magenta,labels=labels)
  else:
      hist = px.pie(df[df[col]>16], 
              names='race_ethnicity',
              color_discrete_sequence=px.colors.sequential.Magenta,labels=labels)

  hist.update_layout({
      'plot_bgcolor': 'rgba(0, 0, 0, 0)',
      'paper_bgcolor': 'rgba(0, 0, 0, 0)',
      }, template = template, title=title)
  return hist
  
@app.callback(Output('Hist','figure'), 
    [Input('DC map', 'clickData'), Input('district-dropdown','value')])
def update_histogram(clickData, metric):
  d = None

  if(clickData != None):
    d = str(clickData['points'][0]['location'])+'D'
    df = data_full[data_full['stop_district'] == d]
  else:
    df = data_full.copy()

  if(metric!='count_child'):
    df = df[df['age'] > 16]


  return create_histogram(metric, df, d)

@app.callback(Output("DC map", "clickData"), [Input('d_reset','n_clicks')])
def update_selected_data(reset_btn):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if('d_reset' in changed_id):
        return None

def create_map(df):
    map_fig = px.scatter_mapbox(
        df, lat='Arrest_Latitude', lon='Arrest_Longitude', 
        color='Defendant_Race',mapbox_style='carto-darkmatter', 
        zoom=12,opacity=0.8,
        category_orders={"Defendant_Race":['BLACK','WHITE','ASIAN','UNK']},
        template=template, color_discrete_map=color_map,
        hover_data={
            'Defendant_Race':True,
            'Defendant_Ethnicity':True,
            'Defendant_Sex':True,
            'Arrest_Category':True,
            'Age':True,
            'Arrest_Latitude':False,
            'Arrest_Longitude':False,
            'Arrest_Date':True,
            'Arrest_Hour':True,

            }
        )
    map_fig.update_layout(
        showlegend=False, 
        margin = dict(l = 0, r = 0, t = 0, b = 0),
        )
    return map_fig

def create_pie(df, choice):
    if choice == 1:
        pie = px.pie(
            df, names='Defendant_Race', color='Defendant_Race',
            color_discrete_map=color_map,
            )
        
    elif(choice==2):
        dfg = df[df['Arrest_Category']!= 'Other Crimes'].groupby(['Arrest_Category']).size().to_frame ().sort_values(
        [0], ascending=False).head(10).reset_index()
        pie=px.pie(
            dfg,names='Arrest_Category', values=0,
            labels={'0':'Count'},color_discrete_sequence=px.colors.qualitative.T10
            )

    pie.update_layout(
                {'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)'},
                template= template,
                showlegend=False,
                margin = dict(l = 0, r = 0, t = 0, b = 0),
                )
    pie.update_traces(
        textposition = 'inside', 
        textinfo='percent+label',)

    return pie

@app.callback(
    Output('arrest_map','figure'),
    [
        Input('year_slider','value'), 
        Input('data_min','value'), 
        Input('data_max','value')
    ]
)
def update_map(y,min,max):
  df = df_arrests[df_arrests['Arrest_Year']==y]
    
  df = df[df['Age'] >= min]
  df = df[df['Age'] <= max]

  return create_map(df)
    
@app.callback(
  [
      Output('sliceopie', 'figure'),
      Output('select_num','children')
  ],
  [
    Input('year_slider','value'), 
    Input('data_min','value'), 
    Input('data_max','value'), 
    Input('choiceopie','value'),
    Input('arrest_map','selectedData')
  ]
)
def update_pie(y,min,max,choice,selectedData):
  lat=[]
  lon=[]
  df = df_arrests[df_arrests['Arrest_Year']==y]
    
  df = df[df['Age'] >= min]
  df = df[df['Age'] <= max]

  if(selectedData):
    for point in selectedData['points']:
      lat.append(point['lat'])
      lon.append(point['lon'])
      
    df=df[(df['Arrest_Latitude'].isin(lat)) & (df['Arrest_Longitude'].isin(lon))]
    
  s = df['Age'].size
  return create_pie(df,choice), s



if __name__ == '__main__':
    app.run_server(debug=False)



