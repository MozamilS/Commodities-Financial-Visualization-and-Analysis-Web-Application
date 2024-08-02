import base64
import requests
import sched
import time
from datetime import datetime, timedelta

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import boto3
from botocore.exceptions import NoCredentialsError
from st_files_connection import FilesConnection

import streamlit as st
from st_files_connection import FilesConnection
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly_express as px
import plotly.graph_objects as go
import os
import json
import datetime 

import pytz


from dotenv import load_dotenv

load_dotenv()

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
s3_bucket_name = os.getenv("S3_BUCKET_NAME")
openai_api_key = os.getenv("OPENAI_API_KEY")
commentary_file_path = 'commentary.json'

def delete_commentary(conn):

    date_df = conn.read(f"{s3_bucket_name}/data-folder/date_df.csv", input_format="csv", ttl=600)
    # Define the timezone as Canada's Mountain Standard Time
    date_df = pd.DataFrame(date_df)
    mst = pytz.timezone('Canada/Mountain')

    # Get the current date in MST
    current_date = datetime.datetime.now(mst).strftime("%Y-%m-%d")

    if (current_date == date_df['upload_date'].iloc[0]):
        file_path = 'commentary.json'
        # Check if the file exists
        if os.path.exists(file_path):
            # Try to delete the file
            try:
                os.remove(file_path)
                print(f"The file '{file_path}' has been deleted successfully.")
            except Exception as e:
                print(f"An error occurred while deleting the file: {e}")
        else:
            st.warning(f"The file '{file_path}' does not exist.")




def create_commentary_file():
    data = {
        "WTI": {   
            "historic_price": False, 
            "sma_ema_returns": False, 
            "auto_correlation": False, 
            "std_dev_area": False, 
            "lm_test_pred_vis": False
        },
        "NATGAS": {   
            "historic_price": False, 
            "sma_ema_returns": False, 
            "auto_correlation": False, 
            "std_dev_area": False, 
            "lm_test_pred_vis": False
        },
        "WHEAT": {   
            "historic_price": False, 
            "sma_ema_returns": False, 
            "auto_correlation": False, 
            "std_dev_area": False, 
            "lm_test_pred_vis": False
        },
        "CORN": {   
            "historic_price": False, 
            "sma_ema_returns": False, 
            "auto_correlation": False, 
            "std_dev_area": False, 
            "lm_test_pred_vis": False
        },
        "SOYBEANS": {   
            "historic_price": False, 
            "sma_ema_returns": False, 
            "auto_correlation": False, 
            "std_dev_area": False, 
            "lm_test_pred_vis": False
        },
        "GLOBCOMINDX": {   
            "historic_price": False, 
            "sma_ema_returns": False, 
            "auto_correlation": False, 
            "std_dev_area": False, 
            "lm_test_pred_vis": False
        }
    }
    
    if not os.path.isfile(commentary_file_path):
        with open(commentary_file_path, 'w') as file:
            json.dump(data, file, indent=4) 



def get_commentary(fig, title, type):
    if not os.path.isfile(commentary_file_path):
        create_commentary_file()

    with open(commentary_file_path, 'r') as file:
        data = json.load(file)
        if data[title].get(type) is not False:
            st.write("Commentary: ")
            st.write(data[title][type])
        else:
            img_bytes = fig.to_image(format="png")
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
            }
            payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": "Provide a detailed breakdown of this chart in a single paragraph."
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 300
            }

            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()  
                content = response.json()["choices"][0]['message']['content']
                

                string_series = pd.Series([content])
                string_series = string_series.str.replace('$', '\$')
                updated_content = string_series.iloc[0]
                st.write("Commentary: ")
                st.write(updated_content)

                data[title][type] = updated_content
                with open(commentary_file_path, 'w') as file:
                    json.dump(data, file, indent=4)

            except requests.RequestException as e:
                st.write(f"An error occurred while contacting OpenAI: {e}")


st.set_page_config(layout="wide")

#get custom css for color scheme and fonts
def formatting_color_text():
    page_bg_img = """
    <style>
    /* Main and Sidebar background color */
    [data-testid="stAppViewContainer"] {background-color: black;}
    [data-testid="stSidebar"] {background-color: #333333;}

    /* General text color */
    body, .stText, .stMarkdown {color: #FF4500;}

    /* Headings text color */
    h1, h2, h3, h4, h5, h6 {color: #ffffff;}

    /* Customize dropdown selector */
    .stSelectbox [role='listbox'] {background-color: #333333; color: #00ff00;}

    /* Customize DataFrame */
    .dataframe {background-color: black; color: #00ff00;}

    /* Customize input fields */
    input {background-color: #333333; color: #00ff00;}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    

def data_store(conn):

    df_price_and_returns_monthly = conn.read(f"{s3_bucket_name}/data-folder/final_df.csv", input_format="csv", ttl=600)
    df_geom_cumm_return_monthly = conn.read(f"{s3_bucket_name}/data-folder/final_cumm_geo.csv", input_format="csv", ttl=600)
    df_annual_geo_returns = conn.read(f"{s3_bucket_name}/data-folder/annual_returns.csv", input_format="csv", ttl=600)
    df_arith_avg_annual_return = conn.read(f"{s3_bucket_name}/data-folder/final_df_avg_annual_arith.csv", input_format="csv", ttl=600)
    df_windsored_avg_return_annual = conn.read(f"{s3_bucket_name}/data-folder/final_df_avg_annual_arith.csv", input_format="csv", ttl=600)
    df_windsored_avg_return_monthly = conn.read(f"{s3_bucket_name}/data-folder/final_df_ang_annual_windsored.csv", input_format="csv", ttl=600)
    df_std_dev_sample_annual = conn.read(f"{s3_bucket_name}/data-folder/std_dev_sample_annual_df.csv", input_format="csv", ttl=600)
    df_min = conn.read(f"{s3_bucket_name}/data-folder/min_returns_df.csv", input_format="csv", ttl=600)
    df_max = conn.read(f"{s3_bucket_name}/data-folder/max_returns_df.csv", input_format="csv", ttl=600)
    df_final_sma = conn.read(f"{s3_bucket_name}/data-folder/final_df_sma.csv", input_format="csv", ttl=600)
    df_final_ema = conn.read(f"{s3_bucket_name}/data-folder/final_df_ema.csv", input_format="csv", ttl=600)
    df_auto_corr = conn.read(f"{s3_bucket_name}/data-folder/acf_values_returns.csv", input_format="csv", ttl=600)
    df_corr_matrix = conn.read(f"{s3_bucket_name}/data-folder/corr_martix.csv", input_format="csv", ttl=600)
    df_metric_table = conn.read(f"{s3_bucket_name}/data-folder/metric_table.csv", input_format="csv", ttl=600)

    df_ml_traintest = conn.read(f"{s3_bucket_name}/data-folder/df_flatten_test.csv", input_format="csv", ttl=600)
    df_ml_forecast = conn.read(f"{s3_bucket_name}/data-folder/df_future_pred.csv", input_format="csv", ttl=600)
    
    return (df_price_and_returns_monthly, df_geom_cumm_return_monthly, df_annual_geo_returns, 
            df_arith_avg_annual_return, df_windsored_avg_return_annual, df_windsored_avg_return_monthly, df_std_dev_sample_annual, 
            df_min, df_max, df_final_sma, df_final_ema, df_auto_corr, df_corr_matrix, df_metric_table,df_ml_traintest,df_ml_forecast) 

    #st.dataframe(df_autocorrelations)
    # commodities = [col.split()[0] for col in df_autocorrelations.columns]
    # selected_commodity = st.selectbox('Select a commodity to filter', commodities)
    # filtered_columns = [col for col in df_autocorrelations.columns if col.startswith(selected_commodity)]
    # filtered_df = df_autocorrelations[filtered_columns]
    # st.dataframe(filtered_df)

def filter_aggregation(df_price_and_returns_monthly, df_geom_cumm_return_monthly, df_annual_geo_returns, 
                       df_arith_avg_annual_return, df_windsored_avg_return_annual, df_windsored_avg_return_monthly, df_std_dev_sample_annual, df_min, df_max, 
                       df_final_sma, df_final_ema, df_auto_corr, df_corr_matrix, df_metric_table,df_ml_traintest,df_ml_forecast):

    # print(df_geom_cumm_return_monthly)
    # print(df_annual_geo_returns)
    # print(df_arith_avg_annual_return)
    # print(df_windsored_avg_return_annual)
    # print(df_windsored_avg_return_monthly)
    # print(df_std_dev_sample_annual)
    # print(df_final_sma)
    # print(df_final_ema)
    # print(df_auto_corr)
    
    """
    Index Adjsutment and datetime incorporation
    """
    #monthly index conversion - datetime - index is yyyy-mm-dd
    df_list_filter_agg_monthly_index = [
    df_price_and_returns_monthly,df_geom_cumm_return_monthly,df_final_sma,df_final_ema,
    ]

    for i in df_list_filter_agg_monthly_index:
        i.set_index(i.columns[0],inplace=True)
        i.index = i.index.astype(str)  # Convert index to string
        i.index = i.index.str[:10]  # Strip the time part if it exists
        i.index = pd.to_datetime(i.index, format='%Y-%m-%d')
        i.index.name = 'Date'

    #ignore year data as intiger will be fine - index is yyyy
    #ignore auto corr as it will be fine as integer index is 0-10

    #ignore min and max and index 1,2,3,4,5 is fine jsut convert to datetime for last column for both - yyyy-mm-dd
    df_min_max_index = [df_min, df_max]
    for i in df_min_max_index:
        i.iloc[:, -1] = i.iloc[:, -1].astype(str)  # Convert to string
        i.iloc[:, -1] = i.iloc[:, -1].apply(lambda x: x.split('T')[0] if 'T' in x else x.split(' ')[0])

    #alter matrix tables  - cmdty and metric 
    matrixs_index = [df_corr_matrix, df_metric_table]
    matrixs_index_names = ['CMDTY', 'Metric']
    for i,j in zip(matrixs_index,matrixs_index_names):
        i.set_index(i.columns[0],inplace=True)
        i.index = i.index.astype(str)  # Convert index to string
        i.index.name = j
    
    #std dev table
    df_std_dev_sample_annual = df_std_dev_sample_annual.set_index('index')
    


    ###drop down selection###
    #get list
    commodities_filter_list = [col.split()[0] for col in df_geom_cumm_return_monthly.columns]
    selected_commodity = st.sidebar.selectbox('Select A Commodity', commodities_filter_list)

    # Apply filtering to data
    filtered_columns_price_mret = [col for col in df_price_and_returns_monthly.columns if col.startswith(selected_commodity)]
    df_filtered_price_mom_mret = df_price_and_returns_monthly[filtered_columns_price_mret]

    filtered_columns_geomcumm_mret = [col for col in df_geom_cumm_return_monthly.columns if col.startswith(selected_commodity)]
    df_filtered_columns_month_geo = df_geom_cumm_return_monthly[filtered_columns_geomcumm_mret]

    filtered_columns_ann_geo = [col for col in df_annual_geo_returns.columns if col.startswith(selected_commodity)]
    df_filtered_columns_ann_geo = df_annual_geo_returns[filtered_columns_ann_geo]

    filtered_columns_ann_arith = [col for col in df_arith_avg_annual_return.columns if col.startswith(selected_commodity)]
    df_filtered_columns_ann_arith = df_arith_avg_annual_return[filtered_columns_ann_arith]

    filtered_columns_ann_wind = [col for col in df_windsored_avg_return_annual.columns if col.startswith(selected_commodity)]
    df_filtered_columns_ann_wind = df_windsored_avg_return_annual[filtered_columns_ann_wind]

    filtered_columns_wind_mret = [col for col in df_windsored_avg_return_monthly.columns if col.startswith(selected_commodity)]
    df_filtered_columns_wind_mret = df_windsored_avg_return_monthly[filtered_columns_wind_mret]

    filtered_columns_ann_stddev = [col for col in df_std_dev_sample_annual.columns if col.startswith(selected_commodity)]
    df_filtered_columns_ann_stddev = df_std_dev_sample_annual[filtered_columns_ann_stddev]

    df_filtered_columns_min = df_min[df_min['Column'].str.startswith(selected_commodity)]
    df_filtered_columns_min = df_filtered_columns_min.drop(columns = {'index', 'Column'})

    df_filtered_columns_max = df_max[df_max['Column'].str.startswith(selected_commodity)]
    df_filtered_columns_max = df_filtered_columns_max.drop(columns = {'index', 'Column'})
    #df_filtered_columns_max = df_filtered_columns_max.rename(columns={'Minimum Return':'Maximum Return', 'Corresponding Min Index': 'Corresponding Max Index'})

    df_filtered_columns_test_pred = df_ml_traintest[df_ml_traintest['commodity'].str.startswith(selected_commodity)]

    df_filtered_columns_forecast = df_ml_forecast[df_ml_forecast['commodity'].str.startswith(selected_commodity)]

    filtered_columns_sma = [col for col in df_final_sma.columns if col.startswith(selected_commodity)]
    df_filtered_columns_sma = df_final_sma[filtered_columns_sma]

    filtered_columns_ema = [col for col in df_final_ema.columns if col.startswith(selected_commodity)]
    df_filtered_columns_ema = df_final_ema[filtered_columns_ema]

    filtered_columns_autcorr = [col for col in df_auto_corr.columns if col.startswith(selected_commodity)]
    df_filtered_columns_autcorr = df_auto_corr[filtered_columns_autcorr]

    filtered_columns_corrmatrix = [col for col in df_corr_matrix.columns if col.startswith(selected_commodity)]
    df_filtered_columns_corrmatrix = df_corr_matrix[filtered_columns_corrmatrix]

    filtered_columns_metric = [col for col in df_metric_table.columns if col.startswith(selected_commodity)]
    df_filtered_columns_metric = df_metric_table[filtered_columns_metric]
    
    
    return (df_filtered_price_mom_mret,df_filtered_columns_month_geo,df_filtered_columns_ann_geo,df_filtered_columns_ann_arith,df_filtered_columns_ann_wind,df_filtered_columns_wind_mret,
            df_filtered_columns_ann_stddev,df_filtered_columns_min,df_filtered_columns_max,df_filtered_columns_sma,df_filtered_columns_ema,df_filtered_columns_autcorr,df_filtered_columns_corrmatrix,
            df_filtered_columns_metric,df_filtered_columns_test_pred,df_filtered_columns_forecast,selected_commodity)

def historic_price(df_filtered_price_mom_mret,width=650,height=300):
    #basic
    fig = px.line(df_filtered_price_mom_mret, x=df_filtered_price_mom_mret.index,y=df_filtered_price_mom_mret.iloc[:,0])
    fig.update_layout(title=f"20Y Price: {df_filtered_price_mom_mret.columns[0]} MOM", yaxis_title = '$ USD',width = width, height = height)
    #generate the trendline using OLS regression
    trendline_fig = px.scatter(df_filtered_price_mom_mret, x=df_filtered_price_mom_mret.index, y=df_filtered_price_mom_mret.iloc[:, 0], trendline="ols")
    trendline_trace = trendline_fig.data[1]
    trendline_trace.update(line=dict(color="red", dash="dot"))
    fig.add_trace(trendline_trace)
    st.plotly_chart(fig,use_container_width=True)
    return fig, df_filtered_price_mom_mret.columns[0], "historic_price"


def sma_ema_returns(df_filtered_price_mom_mret,df_filtered_columns_sma,df_filtered_columns_ema,width=800, height=300):

    fig = go.Figure()

    #add SMA and EMA
    fig.add_trace(go.Scatter(x=df_filtered_columns_sma.index, y=df_filtered_columns_sma.iloc[:, 0], mode='lines', name='SMA'))
    fig.add_trace(go.Scatter(x=df_filtered_columns_ema.index, y=df_filtered_columns_ema.iloc[:, 0], mode='lines', name='EMA'))
    #add returns
    fig.add_trace(go.Scatter(x=df_filtered_price_mom_mret.index, y=df_filtered_price_mom_mret.iloc[:, 1], mode='lines', name='Returns (COP)'))
    fig.update_layout(title=f"20Y Returns: {df_filtered_columns_sma.columns[0]} MOM", yaxis_title='$ USD', width=width, height=height)
    

    st.plotly_chart(fig,use_container_width=True)
    return fig, "sma_ema_returns"


def auto_correlation(df_filtered_columns_autcorr,width=650, height=300):
    
    #set colors
    # Define a smoother color scale
    colors = []
    for val in df_filtered_columns_autcorr.iloc[:, 0]:
        if val < 0:
            intensity = int((1 + val) * 127 + 128)  # Scale negative values to 128-255 for blue shades
            colors.append(f'rgb(0, 0, {intensity})')
        else:
            intensity = int(val * 127 + 128)  # Scale positive values to 128-255 for red shades
            colors.append(f'rgb({intensity}, 0, 0)')

    #define figure
    fig = go.Figure()
    

    #add the autocorrelation data with values displayed above each bar and conditional formatting
    fig.add_trace(go.Bar(
        x=df_filtered_columns_autcorr.index,
        y=df_filtered_columns_autcorr.iloc[:, 0],
        name='Autocorrelation',
        text=df_filtered_columns_autcorr.iloc[:, 0].round(2),  
        textposition='outside', 
        marker_color=colors  
    ))

    #update layout
    fig.update_layout(
        title=f"{df_filtered_columns_autcorr.columns[0]}",
        xaxis_title='Months Lag',
        xaxis=dict(autorange='reversed'),#reverse x
        yaxis_title='Autocorrelation (0 < p < 1)',
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    return fig, "auto_correlation"


def std_dev_area(df_filtered_columns_ann_stddev,width=650, height=300):
    
    #ensure index fitting
    df_filtered_columns_ann_stddev.reset_index(inplace=True)  
    df_filtered_columns_ann_stddev['year'] = df_filtered_columns_ann_stddev['index'].astype(str)  
    column_name = df_filtered_columns_ann_stddev.columns[1]  

    peaks, _ = find_peaks(df_filtered_columns_ann_stddev[column_name])
    df_peaks = df_filtered_columns_ann_stddev.iloc[peaks]

    fig = px.area(df_filtered_columns_ann_stddev, x='year', y=column_name,
                  labels={'year': 'Year', 'volatility': 'Volatility'},
                  title=f'{column_name} - Harmonic',  
                  color_discrete_sequence=['red'])  

    fig.add_scatter(x=df_peaks['year'], y=df_peaks[column_name], mode='markers', marker=dict(color='blue', size=10),
                    name='Peaks')


    fig.update_layout(xaxis_title='Year',
                      yaxis_title='Volatility',
                      xaxis={'tickmode': 'linear'},
                      yaxis=dict(tickformat=".2%"))  

    st.plotly_chart(fig)
    return fig, "std_dev_area"

def lm_test_pred_vis(df_filtered_columns_test_pred):
    #create line plot
    fig = px.line(df_filtered_columns_test_pred,
                x=df_filtered_columns_test_pred.index,
                y=[df_filtered_columns_test_pred.columns[2], df_filtered_columns_test_pred.columns[3]])

    #update legend
    fig.data[0].name = 'Predicted Test Data (ŷ)' 
    fig.data[1].name = 'Actual Test Data (y)'  

    #tick spacing
    tick_spacing = 5
    tick_values = df_filtered_columns_test_pred.index[::tick_spacing]
    tick_labels = list(range(80, -1, -tick_spacing))  #ticks between 80 and 0

    #title and axis
    fig.update_layout(
        title=f"{df_filtered_columns_test_pred['commodity'].iloc[0].split(' ')[0]} RNN-LSTM Test Data to Predicted Data - Last 80 Months",
        yaxis_title='Return (dec.)',
        xaxis_title='Months Lag',
        xaxis=dict(
            tickmode='array',
            tickvals=tick_values,
            ticktext=tick_labels  #reverse ticks
        )
    )

    # fig.add_annotation(
    # text=f"Adj R Squared - Test: {round(df_filtered_columns_test_pred.iloc[0,4],2)}",
    # xref="paper", yref="paper",
    # x=1.20, y=0.75,  # Adjust x and y to place the annotation where you want
    # showarrow=False,
    # font=dict(size=12, color="white")
    # )

    # fig.add_annotation(
    # text=f"RMSE - Test: {round(df_filtered_columns_test_pred.iloc[0,5],3)}",
    # xref="paper", yref="paper",
    # x=1.20, y=0.70,  # Adjust x and y to place the annotation where you want
    # showarrow=False,
    # font=dict(size=12, color="white")
    # )


    # Plot the chart in Streamlit
    st.plotly_chart(fig)
    st.markdown(f"<p style='color:green;'>Adj R Squared - Test: {round(df_filtered_columns_test_pred.iloc[0,4],2)}</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:green;'>RMSE - Test: {round(df_filtered_columns_test_pred.iloc[0,5],3)}</p>", unsafe_allow_html=True)
    return fig, "lm_test_pred_vis"



#concat minmax
def min_max_combine(df_filtered_columns_min,df_filtered_columns_max):

    min_max_final_filtered_df = pd.concat([df_filtered_columns_max, df_filtered_columns_min], ignore_index=True)
    min_max_final_filtered_df = min_max_final_filtered_df.rename(columns={'Minimum Return':'Returns', 'Corresponding Min Index':'Date'})
    min_max_final_filtered_df.index = ['Max', 'Min']

    return min_max_final_filtered_df

def forcasted_format(df_filtered_columns_forecast):
    df_filtered_columns_forecast = df_filtered_columns_forecast.drop(columns={'index'})

    return df_filtered_columns_forecast

    

def main():

    target_hour = 20  # 10 PM in 24-hour format
    target_minute = 00
    target_second = 0
    
    # Get the current time in MST
    mst = pytz.timezone('US/Mountain')
    current_time = datetime.datetime.now(mst)
    
    # Calculate the next target time
    next_target_time = current_time.replace(hour=target_hour, minute=target_minute, second=target_second, microsecond=0)
    if current_time >= next_target_time:
        next_target_time += datetime.timedelta(days=1)
    
    # Calculate the interval until the next target time in milliseconds
    interval = (next_target_time - current_time).total_seconds() * 1000
    
    # Set the st_autorefresh
    st_autorefresh(interval=interval, limit=365, key="fizzbuzzcounter")


    #establish connection
    conn = st.connection('s3', type=FilesConnection)
    #fomratting - colors and text
    delete_commentary(conn)

    formatting_color_text()
    #data source call
    (df_price_and_returns_monthly, df_geom_cumm_return_monthly, df_annual_geo_returns, 
            df_arith_avg_annual_return, df_windsored_avg_return_annual, df_windsored_avg_return_monthly, df_std_dev_sample_annual, 
            df_min, df_max, df_final_sma, df_final_ema, df_auto_corr, df_corr_matrix, df_metric_table,df_ml_traintest,df_ml_forecast)   = data_store(conn)
    #filter 
    (df_filtered_price_mom_mret,df_filtered_columns_month_geo,df_filtered_columns_ann_geo,df_filtered_columns_ann_arith,
     df_filtered_columns_ann_wind,df_filtered_columns_wind_mret,df_filtered_columns_ann_stddev,df_filtered_columns_min,
     df_filtered_columns_max,df_filtered_columns_sma,df_filtered_columns_ema,df_filtered_columns_autcorr,df_filtered_columns_corrmatrix,
            df_filtered_columns_metric,df_filtered_columns_test_pred,df_filtered_columns_forecast,selected_commodity) = filter_aggregation(df_price_and_returns_monthly, df_geom_cumm_return_monthly, df_annual_geo_returns, 
            df_arith_avg_annual_return, df_windsored_avg_return_annual, df_windsored_avg_return_monthly, df_std_dev_sample_annual, 
            df_min, df_max, df_final_sma, df_final_ema, df_auto_corr, df_corr_matrix, df_metric_table,df_ml_traintest,df_ml_forecast)
    #concat minmax 
    min_max_final_filtered_df = min_max_combine(df_filtered_columns_min,df_filtered_columns_max)
    #fix forcast format
    df_filtered_columns_forecast = forcasted_format(df_filtered_columns_forecast)


    #title
    st.title(f"Active CMDTY: {df_filtered_price_mom_mret.columns[0]}")
    #subheader
    st.header('Analytical: Historical and Forecasted')
    #Page structure
    col1, col2 = st.columns([3, 1]) 
    with col1:
        with st.container():
            #subscript
            st.write("20Y Price Chart:")
            #chart
            fig, title, type = historic_price(df_filtered_price_mom_mret)
            get_commentary(fig, title, type)
            # Subscript
            st.write("Returns and SMA, EMA Chart:")
            # SMA, EMA, returns
            fig, type = sma_ema_returns(df_filtered_price_mom_mret, df_filtered_columns_sma, df_filtered_columns_ema)
            get_commentary(fig, title, type)

            # Subscript:
            st.write("30 Month Lag Autocorrelation: ")
            #chart
            fig, type = auto_correlation(df_filtered_columns_autcorr,width=650, height=300)
            get_commentary(fig, title, type)

            #subscript
            st.write("Historical Annual Harmonic Std Deviation: ")
            #chart
            fig, type = std_dev_area(df_filtered_columns_ann_stddev,width=650, height=300)
            get_commentary(fig, title, type)

            #subheader
            st.header('Analytical: Predictive')
            #subscript
            st.write('LSTM - Predictive Model - Test Data to Test Predictions')
            #chart
            if selected_commodity == 'GLOBCOMINDX':
                st.write('There is no predictive model for the market. Bad Practice.')
            else:
                fig, type = lm_test_pred_vis(df_filtered_columns_test_pred)
                get_commentary(fig, title, type)


    with col2:
        with st.container():
            #subscipt
            st.write("Common Metrics: ")
            #dataframe
            st.dataframe(df_filtered_columns_metric,width=500)
            #subscript
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write("Monthly Geometric Compound Returns: ")
            #dataframe
            st.dataframe(df_filtered_columns_month_geo, width=700, height=310)
            #subscript
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write("Correlation Matrix: ")
            #dataframe
            st.dataframe(df_filtered_columns_corrmatrix,height=280)
            #subscript
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write("Min and Max Values: ")
            #dataframe
            st.dataframe(min_max_final_filtered_df)
            #subscript
    
            #dataframe
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            st.write('Forecasted Returns - Future 6 Months')
            st.dataframe(df_filtered_columns_forecast)
   
            





if __name__ == "__main__":
    
    main()

