# ppe_ctgov_intervention_webapp.py

from random import random
import math

from datetime import date
from random import randint

import pandas as pd

from bokeh.layouts import column, row
from bokeh.models import Button, ColumnDataSource, DataTable, TableColumn, DateFormatter
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc
from bokeh.models.widgets import Div, Paragraph, Slider, RadioGroup, CheckboxGroup, TextInput

import functions_ppe_int_webapp as fppe

import logging
from logging import debug as dbg

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - \n%(message)s')
#logging.disable(logging.CRITICAL)

def temp_data_callback(attrname, old, new):
    global df_ctdata
    ptext.text= 'Loading'
    df_ctdata = fppe.load_temp_data()
    #ptext.text= str(df_ctdata.head())
    topic_list, df_ctdata = fppe.run_topic_modeling(df_ctdata, no_topics = num_topics.value)
    #ptext.text= str(topic_list)
    #str_topic_list = [str(topic_list[i]) for i in range(len(topic_list))]
    str_topic_list = []
    for topic_idx in range(len(topic_list)):
        tmp_topics = topic_list[topic_idx]
        tmp_str = ""
        for idx in range(len(tmp_topics)):
            tmp_str = tmp_str + tmp_topics[idx] + ", "
        str_topic_list.append(tmp_str)

    temp_topic_list.update(labels=str_topic_list)

def num_topic_callback(attrname, old, new):
    global df_ctdata
    ptext.text= str(num_topics.value)
    ptext.text= str(df_ctdata.head())


def select_topic_callback(attrname, old, new):
    global df_ctdata
    topic_encode = [0 for x in range(len(temp_topic_list.labels))]
    for idx in range(len(temp_topic_list.active)):
        cind = temp_topic_list.active[idx]
        topic_encode[cind] = 1
    interv_list = fppe.get_interventions(df_ctdata, topic_encode)
    #ptext.text = str(interv_list)
    #ptext.text = str(df_ctdata)
    
    source.data = {
        'interv_type'             : interv_list['interv_type'],
        'interv_name'             : interv_list['interv_name'],
        'interv_conditions'             : interv_list['interv_condition'],
        'interv_trial'             : interv_list['interv_trial'],
        'interv_trial_name'             : interv_list['interv_trial_name'],
        'interv_sponsor'             : interv_list['interv_sponsor'],
        'interv_last_update'             : interv_list['interv_last_update']
    }
    
    #source = ColumnDataSource(df_ctdata.head(1))
    #ptext.text = str(source.data)
    #columns = [TableColumn(field=col_name, title=col_name) for col_name in source.column_names]
    #Columns = [TableColumn(field=Ci, title=Ci) for Ci in df_ctdata.columns] # bokeh columns
    #data_table.update(columns=Columns, source=ColumnDataSource(df_ctdata), height = 500) # bokeh table   #data_table.update(source=source, columns=source.column_names)
    #data_table.update(source=source, columns=columns)


df_ctdata=pd.DataFrame([])
#df_ctdata=fppe.search_ct_gov()

search_terms_input = TextInput(value="default", title="Pathogen Search Terms:")

temp_pathogen_list = CheckboxGroup(
        labels=[
            "SARS", 
            "Zika", 
            "Malaria"]
        )

topic_method = RadioGroup(
        labels=[
            "LDA",
            "NMF"]
        )

num_topics = Slider(start=0, end=15, value=2, step=1, title="Number of Topics/Clusters")

temp_topic_list = CheckboxGroup(
        labels=[
            "Topic 1", 
            "Topic 2", 
            "Topic 3"]
        )


sample_string = """Loading"""

ptext = Div(text=sample_string,width=200, height=100)

source = ColumnDataSource(data=dict())

columns = [
    TableColumn(field="interv_type", title="Intervention Type"),
    TableColumn(field="interv_name", title="Intervention Name"),
    TableColumn(field="interv_conditions", title="Conditions"),
    TableColumn(field="interv_trial", title="Trial Number (NCT)"),
    TableColumn(field="interv_trial_name", title="Trial Title"),
    TableColumn(field="interv_sponsor", title="Sponsor"),
    TableColumn(field="interv_last_update", title="Last Update Date")
    #TableColumn(field="interv_last_update", title="Last Update Date", formatter=DateFormatter())
]

data_table = DataTable(source=source, columns=columns, width=800, fit_columns=False)


#source = ColumnDataSource(df_ctdata)
#data_table = DataTable(source=source, columns=columns, width=400, height=280)
#data_table = DataTable(source=source, width=400, height=2800)
'''
data = dict(
        dates=[date(2014, 3, i+1) for i in range(10)],
        downloads=[randint(0, 100) for i in range(10)],
    )
source = ColumnDataSource(data)

columns = [
        TableColumn(field="dates", title="Date", formatter=DateFormatter()),
        TableColumn(field="downloads", title="Downloads"),
    ]
data_table = DataTable(source=source, columns=columns, width=400, height=280)
'''
layout3 = column(search_terms_input, temp_pathogen_list, topic_method, num_topics, temp_topic_list)

num_topics.on_change('value',num_topic_callback)
temp_pathogen_list.on_change('active',temp_data_callback)
temp_topic_list.on_change('active',select_topic_callback)

layout2 = row(layout3, ptext, data_table)
#layout2 = column(layout3, data_table)
curdoc().add_root(layout2)
