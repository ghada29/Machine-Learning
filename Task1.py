# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Pandas is a Python library for data analysis
#Read the dataset, convert it to DataFrame and display some from it.
#Display structure and summary of the data.

import pandas as pd
dataset=pd.read_csv("Wuzzuf_Jobs.csv")
dataset.describe()

#count duplicated rows
dataset.duplicated().sum()
#count nonduplicated rows
(~dataset.duplicated()).sum()

#delete duplicated rows
dataset.drop_duplicates(subset=['Title', 'Company', 'Location','Type', 'Level', 'YearsExp', 'Country', 'Skills']
                                    ,keep="first", inplace=True)
#count null
dataset.isnull().sum()
#Count the jobs for each company and display that in order (What are the most demanding companies for jobs?).
company=dataset['Company'].value_counts()

#pie chart
import matplotlib.pyplot as plt
#for all companies
plt.pie(company)
plt.title("No. of Jobs in each company")
#for first 10 company
plt.pie(company[0:10])

#the most popular job titles
title=dataset['Title'].value_counts()

#bar chart
fig=plt.figure(figsize=(20,5))
f=title.keys()
#for all jops
plt.bar(f,title, color='red', width=0.2)
plt.xlabel("Job titles")
plt.ylabel("No. of Jops")
plt.show()

#for first 10 jops
plt.bar(f[0:10],title[0:10], color='red', width=0.2)

#most popular areas
area=dataset['Location'].value_counts()
#bar chart
fig=plt.figure(figsize=(30,5))
a=area.keys()
#for all location
plt.bar(a,area, color='red', width=0.3)
plt.xlabel("Area")
plt.ylabel("no. of area")
#for first 10 location
plt.bar(a[0:10],area[0:10], color='red', width=0.3)

#print skill one by one ,their counts
s=dataset['Skills'] 
skill=dataset['Skills'].value_counts()

