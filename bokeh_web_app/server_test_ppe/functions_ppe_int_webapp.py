# Sample Code copied from https://medium.com/mlreview/topic-modeling-with-scikit-learn-e80d33668730

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import accuracy_score

from datetime import datetime

import logging
from logging import debug as dbg
import numpy as np
import pandas as pd
import os
import argparse
import sys

from io import StringIO

import requests

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - \n%(message)s')
#logging.disable(logging.CRITICAL)

def search_ct_gov(search_term='sars'):

    max_res_pull = 100
    # preprocess search string
    # -- Replace space with '+'
    search_term = search_term.replace(" ", "+")
    # -- Replace & with '%26'
    search_term = search_term.replace("&", "%26")

    # Build Search URL
    # -- example: search_url = 'https://clinicaltrials.gov/api/query/study_fields?fmt=CSV&expr=heart+attack&fields=NCTId,Condition,InterventionName'
    search_url = 'https://clinicaltrials.gov/api/query/study_fields?fmt=CSV&expr=' + search_term + '&fields=NCTId,Condition,InterventionName,InterventionType,OfficialTitle'

    # Execute get request
    response = requests.get(search_url)
    tmp_search_res = response.text
    dbg(response.text)

    # Splitting on newline delimiter
    res_list = tmp_search_res.splitlines()
    dbg(res_list)
    
    # This code is written for clinicaltrials.gov API version "1.01.02"
    # Ensure that implemented API verson has not changed
    api_ver_str = res_list[0].replace('\"', '')
    api_ver_num_str = api_ver_str.split(':')[1].strip()

    dbg(api_ver_num_str)

    api_ver_check = ("1.01.02" == api_ver_num_str)
    dbg(api_ver_check)

    # Extract total number of trials available for search query
    num_trials_res_str = res_list[4].replace('\"', '')
    num_trials_res_value = int(num_trials_res_str.split(':')[1].strip())

    dbg(num_trials_res_value)

    # Calculate number of get requests needed to grab all results
    # -- max_res_pull is the max number of results that can be requested in each get request
    # -- this limit is set by clinicaltrials.gov
    total_num_pulls = (num_trials_res_value // max_res_pull) + 1

    dbg(total_num_pulls)

    all_search_results_list = []
    all_search_results_str = ''

    all_search_results_list.append(res_list[10].replace('\"', '').split(','))
    all_search_results_str = all_search_results_str.join(res_list[10]) + '\n'

    dbg(all_search_results_list)
    dbg(all_search_results_str)

    #&min_rnk=1198&max_rnk=1249&fmt=xml

    for pull_idx in range(total_num_pulls):
        min_rank = pull_idx*max_res_pull + 1
        max_rank = (pull_idx+1)*max_res_pull + 1
        
        search_url = 'https://clinicaltrials.gov/api/query/study_fields?fmt=CSV&expr=' + search_term + '&fields=NCTId,Condition,InterventionName&min_rnk=' + str(min_rank) + '&max_rnk=' + str(max_rank)
        
        response = requests.get(search_url)
        #tmp_search_res = response.text.replace('\"', '')
        tmp_search_res = response.text
        # Splitting on newline delimiter
        res_list = tmp_search_res.splitlines()
        res_list_comma_split = [x.split(',') for x in res_list]
        dbg(res_list_comma_split[11:])

        all_search_results_list.extend(res_list_comma_split[11:])
        temp_results_str = '\n'.join(str(x) for x in res_list[11:])
        all_search_results_str = all_search_results_str + temp_results_str + '\n'


    dbg(all_search_results_list)
    
    all_search_results_str_IO = StringIO(all_search_results_str)
    df_search_results = pd.read_csv(all_search_results_str_IO) 
    dbg(df_search_results.head())
    dbg(df_search_results.tail())
    dbg(df_search_results.columns)

    return df_search_results

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
    print('\n\n')

def save_topics(model, feature_names, no_top_words):
    topic_list= []
    for topic_idx, topic in enumerate(model.components_):
        tmp_list = []
        for i in topic.argsort()[:-no_top_words - 1:-1]:
            tmp_list.append(feature_names[i])
        topic_list.append(tmp_list)

    return topic_list

def load_csv_ct_data(trial_file_list):
    tot_df = pd.DataFrame()
    for count, fn in enumerate(trial_file_list):
        dbg('fn')
        dbg(fn)

        tmp_df = pd.read_csv(fn, encoding='ANSI')
        tmp_df['class'] = count
        dbg(tmp_df.head())

        tot_df = pd.concat([tot_df, tmp_df.head(n=50)], ignore_index=True, sort=False)
        #tot_df = pd.merge(tot_df, tmp_df, on='NCT Number')

    dbg(tot_df)
    dbg(tot_df.head())
    dbg(tot_df.tail())

    return tot_df

def load_temp_data():
    # eventually use this to read csvs based on temp sars/zika/malaria checkboxes
    trial_file_list = []
    #trial_file_list.append('data/ct_zika.csv')
    #trial_file_list.append('data/ct_malaria.csv')
    trial_file_list.append('../../data/ct_sars.csv')
    #trial_file_list.append('data/ct_zika.csv')

    tot_df = load_csv_ct_data(trial_file_list)

    return tot_df

def run_topic_modeling(df_data, no_topics):
    documents = df_data['Title']
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(documents)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    nmf = NMF(n_components=no_topics, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    trans_nmf = nmf.transform(tfidf)
    max_trans_nmf = np.argmax(trans_nmf, axis=1)
    df_data['pred'] = max_trans_nmf

    no_top_words = 10

    topic_list = save_topics(nmf, tfidf_feature_names, no_top_words)

    return topic_list, df_data

def get_interventions(df_data, topic_encode):
    topic_mask = [False for x in df_data['Interventions']]
    for idx, topic_enc in enumerate(topic_encode):
        if topic_enc == 1:
            tmp_mask = (df_data['pred'] == idx)
            tmp_mask = tmp_mask.to_list()
            topic_mask = [a or b for a, b in zip(topic_mask, tmp_mask)]

    df_data_masked = df_data[topic_mask]
    #gen_interv_list = df_data_masked['Interventions'].dropna()
    gen_interv_index = df_data_masked['Interventions'].dropna().index

    interv_master_list = {}
    interv_master_list['interv_type'] = []
    interv_master_list['interv_name'] = []
    interv_master_list['interv_condition'] = []
    interv_master_list['interv_trial'] = []
    interv_master_list['interv_trial_name'] = []
    interv_master_list['interv_sponsor'] = []
    interv_master_list['interv_last_update'] = []

    df_data_interv = df_data_masked[df_data_masked.Interventions.notnull()]

    #for temp_interv in gen_interv_list:
    #for temp_trial in df_data_interv:
    for index, row in df_data_interv.iterrows(): # apparently iterrows is bad. I should find better way to process this data
        temp_interv = row['Interventions']
        #return temp_interv
        interv_split = temp_interv.split('|')

        for tmp_str in interv_split:
            tmp_split = tmp_str.split(':')
            interv_type = tmp_split[0].strip()
            interv_name = tmp_split[1].strip()

            interv_master_list['interv_type'].append(interv_type)
            interv_master_list['interv_name'].append(interv_name)
            interv_master_list['interv_condition'].append(row['Conditions'])
            interv_master_list['interv_trial'].append(row['NCT Number'])
            interv_master_list['interv_trial_name'].append(row['Title'])
            interv_master_list['interv_sponsor'].append(row['Sponsor/Collaborators'])
            tmp_date = datetime.strptime(row['Last Update Posted'], '%B %d, %Y')
            #interv_master_list['interv_last_update'].append(row['Last Update Posted'])
            interv_master_list['interv_last_update'].append(tmp_date.date())
    return interv_master_list

if __name__ == "__main__":
    search_ct_gov()

'''
#if __name__ == '__main__':

trial_file_list = []
#trial_file_list.append('data/ct_zika.csv')
#trial_file_list.append('data/ct_malaria.csv')
trial_file_list.append('data/ct_sars.csv')
#trial_file_list.append('data/ct_zika.csv')

tot_df = load_csv_ct_data(trial_file_list)

##################
#tot_df = tot_df.head(50)

#dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
#documents = dataset.data

#print(len(documents))

print(tot_df.head())
print(tot_df.columns)
print(tot_df.shape)
print(tot_df['Title'].head())

documents = tot_df['Title']
#documents = tot_df['Conditions']

#sys.exit()

#no_features = 1000
#no_features = 10

# NMF is able to use tf-idf
#tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
#tfidf_cond = tfidf_vectorizer.transform(documents_cond)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
#tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

#no_topics = 20
no_topics = 2 

# Run NMF
#nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
nmf = NMF(n_components=no_topics, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
trans_nmf = nmf.transform(tfidf)
max_trans_nmf = np.argmax(trans_nmf, axis=1)

nmf_acc = accuracy_score(max_trans_nmf, tot_df['class'])

print(trans_nmf)
print(max_trans_nmf)
print(nmf_acc)

# Run LDA
#lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_offset=50.,random_state=0).fit(tf)
trans_lda = lda.transform(tfidf)
max_trans_lda = np.argmax(trans_lda, axis=1)

lda_acc = accuracy_score(max_trans_lda, tot_df['class'])
print(max_trans_lda)
print(lda_acc)

no_top_words = 10
#no_top_words = 3

display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)

#print(lda.components_)

nmf_mask = (max_trans_nmf == 0)

tot_df['nmf_pred'] = max_trans_nmf

df_non_sars = tot_df[nmf_mask]
print(df_non_sars['Title'])
print(df_non_sars['Conditions'])
print(df_non_sars['Interventions'])

interv_master_list = {}

#l = [x for x in df_non_sars['Interventions'] if ~np.isnan(x)]

print(df_non_sars['Interventions'].dropna())
gen_interv_list = df_non_sars['Interventions'].dropna()

temp_interv = df_non_sars['Interventions'][1]

for temp_interv in gen_interv_list:


    print(temp_interv)
    interv_split = temp_interv.split('|')

    for tmp_str in interv_split:
        tmp_split = tmp_str.split(':')
        print(tmp_split)
        interv_type = tmp_split[0].strip()
        interv_name = tmp_split[1].strip()

        print(interv_type)
        print(interv_name)
        if interv_type not in interv_master_list.keys():
            interv_master_list[interv_type] = []

        interv_master_list[interv_type].append(interv_name)

print(interv_master_list)

print('\n')
for tmp_key in interv_master_list.keys():
    print(tmp_key)
    print(interv_master_list[tmp_key])
    print('\n')

#tot_df.to_csv('zika_malaria_pred.csv')
'''
'''
marker_list = Line2D.filled_markers
color_list = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')

color_count = 0

legend_list = []

fig, ax = plt.subplots()

for true_class in range(2):
    for km_class in range(2):
        tmp_pca0 = []
        tmp_pca1 = []
        if (km_class == true_class):
            fill_style = 'full'
        else:
            fill_style = 'none'
        color_count = color_count+1
        dbg(color_count)
        tmp_legend_str = 'True Class %d, Pred Class %d' % (true_class, km_class)
        legend_list.append(tmp_legend_str)
        for count in range(len(max_trans_nmf)):
            if (max_trans_nmf[count] == km_class) and (tot_df['class'][count] == true_class):
                #tmp_enc.append(color_list[color_count])
                tmp_pca0.append(pca_enc[count,2])
                tmp_pca1.append(pca_enc[count,1])
            else:
                #tmp_enc.append(np.nan)
                tmp_pca0.append(np.nan)
                tmp_pca1.append(np.nan)


        #plt.scatter(tmp_pca0,tmp_pca1, c=color_list[color_count], marker=marker_list[color_count], fillstyle=fill_style)
        ax.plot(tmp_pca0,tmp_pca1, c=color_list[color_count%len(color_list)], marker=marker_list[color_count%len(marker_list)], fillstyle=fill_style, lw=0, label=tmp_legend_str)
'''
