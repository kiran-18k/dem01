# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:31:17 2022

@author:Kiran
"""

# Importing the required libraries For Data Processing and Feed File Preparation

import pandas as pd
import numpy as np
from functools import reduce
import re
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

#%%
path = 'D:/Pricing Analytics/Price Simulator/FY23Q3/'
qc='D:/Pricing Analytics/Price Simulator/FY23Q3/QC/QC TBA & PS/Brita/'
output='D:/Pricing Analytics/Price Simulator/FY23Q3/Output/'
cust_path = 'D:/Pricing Analytics/Price Simulator/FY23Q3/Custom Aggregates FY21Q1/'
pos_path = 'D:/Pricing Analytics/Price Simulator/FY23Q3/POS/'
bda_path = 'D:/Pricing Analytics/Price Simulator/FY23Q3/BDA Co-Efficient File/'

#%%
# #### Mapping Files

# Assign spreadsheet filename to `file`
mappings = 'Mapping BDA_PPL_FY21Q2.xlsx'

# Load spreadsheet
map_xl = pd.ExcelFile(path+'Mapping Files/'+mappings)

# Print the sheet names
print(map_xl.sheet_names)

# Load a sheet into respective DataFrames
geog_map = map_xl.parse('Geography Map')
time_map = map_xl.parse('Time Periods Map')
prod_map = map_xl.parse('BDA Product Map')
ppl_map = map_xl.parse('PPL')
pos_map = map_xl.parse('POS Mapping')
pos_map = pos_map[pos_map['Comment']=='Include']

#%%
# POS file for Brita
# Reading the Pos file for brita
iri_df = pd.read_excel(pos_path + 'Brita.xlsx', skiprows = 1)
iri_df['Geography'] = iri_df['Geography'].str.upper()
# Considering only Four retailers for T-Bda view
iri_df = iri_df[iri_df['Geography'].isin(['TOTAL US - FOOD','WALMART CORP-RMA - WALMART',
                            'TARGET CORP-RMA - MASS',"SAM'S CORP-RMA - CLUB"])]
#Mapping the geographies with the geog map file for Highler level retailer
iri_df['Geography'] = iri_df['Geography'].str.upper().map(geog_map.set_index('Geography_Name')['Geography'])

#%%
# Preparation of pos file that has to be merged with Coeff db file 
pos_brita = iri_df.copy()
pos_brita = pos_brita[['Geography','Product','Product Key']].drop_duplicates() 

# Map Product for Brand and Sub Brand info:
pos_brita['Sub_Brand'] = pos_brita['Product'].map(pos_map.set_index('Product Name')['Subbrand Elasticity File'])
pos_brita['Brand'] = pos_brita['Product'].map(pos_map.set_index('Product Name')['Brand Elasticity File'])
pos_brita['SBU'] = 'Brita'
pos_brita['Division'] = 'Specialty'

pos_brita = pos_brita[pos_brita['Sub_Brand'].notnull()]
pos_brita.to_csv(qc+'pos_brita.csv')

#%%
#Filtering out required columns for merging from ppl map file.
ppl_map = ppl_map[['Brand Elasticity File','Subbrand Elasticity File']].drop_duplicates()
#Merge with Pos data 
pos_brita = ppl_map.merge(pos_brita,left_on=['Brand Elasticity File','Subbrand Elasticity File'],right_on=['Brand','Sub_Brand'],how='inner')
pos_brita.to_csv(qc+'pos_brita1.csv')

#%%
#Including the info that has the granular level in Standard Heiarchy:
iri_df = iri_df[iri_df['Standard Hierarchy Level'].isin(['SIZE_WATERFILTER_H1_6'])].reset_index(drop=True)
# Finding SCBV for product that does not have SBCV
iri_df['SCBV_new'] = iri_df.apply(lambda x: x['Baseline Volume']/(x['Volume Sales']/x['Stat Case Volume']) if
(pd.isna(x['Stat Case Baseline Volume']) and (~pd.isna(x['Stat Case Volume']))) else x['Stat Case Baseline Volume'],axis=1)
#considering only those rows that have SCBV.  
iri_df.drop(columns=['Stat Case Baseline Volume'],axis=1,inplace=True)
iri_df.rename(columns={'SCBV_new':'Stat Case Baseline Volume'},inplace=True)
iri_df = iri_df[iri_df['Stat Case Baseline Volume'].notnull()].reset_index(drop=True)

#%%
# Product key check :
iri_df['new'] = iri_df['Product Key'].str.split(':')
iri_df['len'] = iri_df['new'].str.len()
iri_df1 = iri_df[iri_df['len'] == iri_df['len'].max()]
iri_df2 = iri_df[iri_df['len'] != iri_df['len'].max()]

#%%
#Distinguishing Clorox vs Competitor Records
clorox_brands = ['BRITA']
iri_df1['Clx_Comp'] = np.where(iri_df1['Clorox Brand Value'].isin(clorox_brands),"Clorox","Competitor")

# Defining Category
iri_df1['Category_Name'] = iri_df1.apply(lambda x: x['Clorox Sub Category Value']+" "+x['Clorox Segment Value'],axis=1 ) 
iri_df1.to_csv(qc+'iri_df1.csv')

#%%
# Brita totals at Sub Category + Segment Level
cat_tot_lowerlevels = iri_df1.groupby(['Category_Name']).agg({'Baseline Dollars':'sum','Baseline Units':'sum'}).rename(columns = {'Baseline Dollars':'Category Dollar','Baseline Units':'Category Units'}).reset_index()

#At subcategory level agg:
cat_tot_subcatlevel = iri_df1.groupby(['Clorox Sub Category Value']).agg({'Baseline Dollars':'sum',
'Baseline Units':'sum'}).reset_index().rename(columns = {'Clorox Sub Category Value':'Category_Name',
'Baseline Dollars':'Category Dollar','Baseline Units':'Category Units'})
                                                         
# At Brand level agg:                                                      
cat_tot_brand = pd.DataFrame({'Category_Name':['BRAND_LEVEL'],'Category Dollar':[cat_tot_subcatlevel['Category Dollar'].sum()],'Category Units':[cat_tot_subcatlevel['Category Units'].sum()]})

# Appending them into one single file
cat_tot_level =[cat_tot_lowerlevels,cat_tot_subcatlevel,cat_tot_brand]
cat_tot = reduce(lambda left,right: left.append(right, ignore_index = True), cat_tot_level)
cat_tot = cat_tot.drop_duplicates()
cat_tot.to_csv(qc+'cat_tot.csv')

#%%
# Segment level
SubCat_remap = {'POUR THROUGH':'PT', 'FAUCET MOUNT':'FM', 'FILTERING BOTTLE':'B'}
Segment_remap = {'FILTERS (PTF)' :'F','FILTRATION SYSTEMS (PTS)':'S','FILTERS (FMF)':'F',
                'FILTRATION SYSTEMS (FMS)':'S', 'FILTERS (FBF)':'F', 'FILTRATION SYSTEMS (FBS)':'S'}
iri_df1['Segment_level'] = iri_df1['Clorox Brand Value'] +" "+iri_df1['Clorox Sub Category Value'].replace(SubCat_remap) + iri_df1['Clorox Segment Value'].replace(Segment_remap)

#%%
# SubBrand level
iri_df1['SubBrand_level'] = iri_df1['Segment_level'] + " " + iri_df1.apply(lambda x: x['Clorox SubBrand Value'].replace(x['Clorox Brand Value'],"").strip(),axis=1)
iri_df1['SubBrand_level'] = iri_df1.apply(lambda x: x['SubBrand_level'].strip(),axis=1)

#%%
# Assigning BU unit:
iri_df1["BU"] = 'BRITA'
iri_df2 = iri_df1[iri_df1['Category_Name'].isnull() == False]
iri_df2['size'] = iri_df2['Clorox Size Value'].str.extract('(\d*\.\d+|\d+)').astype(float)
iri_df2.to_csv(qc+'iri_df2.csv')

#%%
# Preparing Coefficient DB file for Merge:
#--BDA file
# Assign spreadsheet filename to `file`
file = 'CoefDB - All Total US FY19Q4.xlsx'
xl = pd.ExcelFile(bda_path+file)

# Print the sheet names
print(xl.sheet_names)

#%%
# Load a sheet into a DataFrame bda_coeff_raw
bda_coeff_raw = xl.parse('CoefDb_ All Total US FY19Q4')
bda_raw_all = bda_coeff_raw[['model_source','Model_Period_End','catlib','Product_Level','Product_Name_Modeled',
'Product_Name_Current', 'Geography_Name','Geography_Level','Base_Price_Elasticity','Promo_Price_Elasticity',
'Base_Statcase_Volume','iriprod','prodkey','plvl5','plvl2','plvl3']].drop_duplicates().reset_index(drop=True)
bda_raw_all = bda_raw_all.replace('NULL', np.nan, regex=True)
bda_raw_all['Product_Name_Subbrand'] = bda_raw_all['plvl5'].str.upper() + bda_raw_all['plvl2'].str.upper() + bda_raw_all['plvl3'].str.upper()  
bda_raw_all['Product_Name_Modeled']= bda_raw_all['Product_Name_Modeled'].str.upper()
bda_raw_all.to_csv(qc+'bda_raw_all.csv')

#%%
# Brita catlibs separated for automation. Check if new catlib available for Brita
final_db1 = bda_raw_all[bda_raw_all['catlib'].isin(['B2','B4','B6','BB','BF','BS']) & bda_raw_all['Product_Level'].isin(['S','X','K','Z','I'])]
final_db1 = final_db1[final_db1['Model_Period_End']>='18-06-2017']
final_db1 = final_db1.drop_duplicates()
final_db1.to_csv(qc+'final_db1.csv')

#%%
#Mapping BDA to POS Retailers/Channels 
coeff_db_map = pd.read_excel(path+'Mapping Files/'+'Hyperion DB Channels.xlsx','Hyperion DB Channels')
dataf1 = final_db1.merge(coeff_db_map, on = ['Geography_Name', 'Geography_Level', 'model_source'], how = 'left')
dataf2 = dataf1[dataf1['IRI Channels'].isnull() == False]
dataf2.to_csv(qc+'dataf2.csv')

#%%
#New df's created based on iriprod's:
dataf2_w_iriprod = dataf2[dataf2['iriprod'].isnull()==False].reset_index(drop=True)
dataf2_wo_iriprod = dataf2[dataf2['iriprod'].isnull()==True].reset_index(drop=True) 

#%%
#Custom Aggregate Keys Mapping
df_Brita = pd.read_excel(cust_path+'CustAggs_FY22Q2 - Brita.xlsx', 'SKUs_to_Aggregate')
df_Brita = df_Brita[['Catcode','Prodlvl','Prodkey','Custprod','IRI_Product_Key','Product_Name']].drop_duplicates()
df_Brita.to_csv(qc+'df_Brita.csv')

df_cust = df_Brita.copy()

cust_agg_keys = df_cust[(df_cust['Prodlvl']=='S') & (pd.isnull(df_cust['Custprod'])==False)]
cust_agg_keys_w_cust_cnt = cust_agg_keys.groupby(['Custprod'])['Custprod'].count().reset_index(name="count")
cust_agg_keys_w_cust_cnt.to_csv(qc+'cust_agg_keys_w_cust_cnt.csv')

#%%
cust_agg_keys1 = cust_agg_keys.merge(cust_agg_keys_w_cust_cnt, on = ['Custprod'], how = 'left')
cust_agg_keys1.to_csv(qc+'cust_agg_keys1.csv')

#%%
dataf3_1 = dataf2_wo_iriprod.merge(cust_agg_keys1, left_on=['prodkey'], right_on=['Custprod'], how = 'left')
dataf3_1['Base_Statcase_Volume2'] = dataf3_1.apply(lambda x: x['Base_Statcase_Volume'] if pd.isnull(x['Custprod'])==True
                      else x['Base_Statcase_Volume']/x['count'], axis=1)

dataf3_1.drop(['iriprod'],axis=1,inplace=True)
dataf3_1.rename(columns={'IRI_Product_Key':'iriprod'},inplace=True)
dataf3_1.to_csv(qc+'dataf3_1.csv')

#%%
dataf3 = dataf3_1.append([dataf2_w_iriprod])
dataf3['Base_Statcase_Volume'] = dataf3.apply(lambda x: x['Base_Statcase_Volume'] if pd.isnull(x['count'])==True
                      else x['Base_Statcase_Volume2'], axis=1)
dataf3.to_csv(qc+'dataf3.csv')

#%%
dataf4 = dataf3[['model_source', 'Geography_Level', 'Geography_Name', 'IRI Channels', 'Model_Period_End',
    'Product_Level','catlib','Product_Name_Modeled','Product_Name_Current','Product_Name','prodkey',
    'CLOROX VS COMP','iriprod','Base_Price_Elasticity', 'Promo_Price_Elasticity','Base_Statcase_Volume','Product_Name_Subbrand']]
dataf5 = dataf4[dataf4['Base_Statcase_Volume']>0]
dataf5.to_csv(qc+'dataf5.csv')

#%%
def roll_a(x):
    d = {} 
    d['Base_Statcase_Volume'] = x['Base_Statcase_Volume'].sum()
    d['Promo_Price_Elasticity'] = np.average(x['Promo_Price_Elasticity'], weights=x['Base_Statcase_Volume'])
    d['Base_Price_Elasticity'] = np.average(x['Base_Price_Elasticity'], weights=x['Base_Statcase_Volume'])
    return pd.Series(d, index=['Promo_Price_Elasticity','Base_Price_Elasticity','Base_Statcase_Volume'])

#%%
CoefDb_All = dataf5.groupby(['iriprod','IRI Channels', 'CLOROX VS COMP', 'Model_Period_End', 'catlib']).apply(roll_a).reset_index()
CoefDb_All = CoefDb_All.rename(columns={'IRI Channels':'Geography'})
CoefDb_All.to_csv(qc+'CoefDb_All.csv')

#%%
dataf5['Product_Name_Current'] = dataf5['Product_Name_Current'].str.upper()
Pdt = dataf5[['IRI Channels','Product_Name_Current','iriprod']]
Pdt = Pdt.rename(columns={'IRI Channels':'Geography'})
Pdt = Pdt.drop_duplicates()

#%%
#Start of manipulation to determine BDA lite for Brita. Check if there is any new catlib for Brita before proceeding.
Brita = CoefDb_All[CoefDb_All['catlib'].isin(['B2','B4','B6','BB','BF','BS'])]
Brita.to_csv(qc+'Brita.csv')

#%%
Brita_New = Brita[Brita['Model_Period_End']>='2021-03-28']
Brita_Old = Brita[Brita['Model_Period_End'] < '2021-03-28']

#%%
Brita_pivot = pd.pivot_table(Brita_Old, values=['Base_Statcase_Volume'], index=['catlib', 'Model_Period_End','CLOROX VS COMP'],
                    columns =['Geography'], aggfunc = {'Base_Statcase_Volume' : sum})

#%%
Brita_pivot.columns = Brita_pivot.columns.droplevel(0)
Brita_1 = Brita_pivot.reset_index().rename_axis(None, axis=1)
Brita_1.to_csv(qc+'Brita_1.csv')

#%%
#Deleting catlib and model period for which Wal and TUS is absent - BDA Lite
Brita_1.dropna(subset=['Walmart Corp-RMA - Walmart','Total US - Food'],inplace=True)
Brita_2 = Brita_1.melt(['catlib','Model_Period_End','CLOROX VS COMP'], var_name ='Geography')
Brita_2.to_csv(qc+'Brita_2.csv')

#%%
Brita_2 = Brita_2[['catlib','Model_Period_End','CLOROX VS COMP','Geography']].drop_duplicates()
Brita_final = Brita_2.merge(Brita_Old, on=['catlib','Model_Period_End','CLOROX VS COMP','Geography'],how='left')
Brita_final.to_csv(qc+'Brita_final_check.csv')

#%%
# Dropping all such rows for which a catlib is not modelled for a particular retailer in a period. Came as as result of pivot.
Brita_final.dropna(subset=['Base_Statcase_Volume'],inplace=True)

#%%
Brita_final = Brita_final.append([Brita_New])

#%%

#Time period mapping
Brita_final['Time Period'] = Brita_final['Model_Period_End'].map(time_map.set_index('Model_Period_End')['modeling_period'])
Brita_final['Geography'] = Brita_final['Geography'].str.upper().map(geog_map.set_index('Geography_Name')['Geography'])

#%%
Brita_final['new'] = Brita_final['iriprod'].str.split(':')
Brita_final['new_subb'] = Brita_final['new'].apply(lambda x : x[0:5])
Brita_final['New iriprod subb'] = Brita_final['new_subb'].str.join(':')
Brita_final['new_mdl_size'] = Brita_final['new'].apply(lambda x : x[5:8])
Brita_final['New mdl size'] = Brita_final['new_mdl_size'].str.join(':')
Brita_final.rename(columns = {'iriprod':'old iriprod'},inplace=True)

#%%
brita_key_map = pd.read_excel(path+'Mapping Files/'+'Brita_Key_Mapping.xlsx', 'Brita')
Brita_final = Brita_final.merge(brita_key_map, left_on = ['New iriprod subb'], right_on = ['Subb IRI Key'], how ='left') 
Brita_final['iriprod'] = Brita_final['Subb Pdt Key'] + ":" + Brita_final['New mdl size']
Brita_final.drop(['new','new_subb','new_mdl_size'],axis=1,inplace=True)
Brita_final.to_csv(qc+'Brita_iri_corr.csv')

#%%
#New Product Key for Brand, Subbrand Mapping
Brita_final = Brita_final[Brita_final['iriprod'].notna()]
Brita_final['new'] = Brita_final['iriprod'].str.split(':')
Brita_final['new_split_irip'] = Brita_final['new'].apply(lambda x : x[:-3])
Brita_final['New iriprod'] = Brita_final['new_split_irip'].str.join(':')
Brita_final.drop(['new','new_split_irip'],axis=1,inplace=True)
Brita_final.to_csv(qc+'Brita_final.csv')

#End of manipulation to determine BDA lite for Brita

#%%
# For Uncured view - ranking based on subbrand
# =============================================================================
CoefDb_All1 = Brita_final.copy()

# Need to reset index so that Ranks can be assigned later (Avoid duplication of index for ranking)
CoefDb_All1.reset_index(drop=True, inplace=True)
CoefDb_All1.to_csv(qc+'CoefDb_All_check1.csv')

#%%
subb_remap = {'BRITA LONGLASTPOUR THROUGHFILTERS':'BRITA LONGLAST/ELITEPOUR THROUGHFILTERS', 
              'BRITA LONGLASTPOUR THROUGHFILTERS (PTF)':'BRITA LONGLAST/ELITEPOUR THROUGHFILTERS (PTF)', 
              'BRITA LONGLASTPOUR THROUGHFILTRATION SYSTEMS':'BRITA LONGLAST/ELITEPOUR THROUGHFILTRATION SYSTEMS', 
              'BRITA LONGLASTPOUR THROUGHFILTRATION SYSTEMS (PTS)':'BRITA LONGLAS/ELITETPOUR THROUGHFILTRATION SYSTEMS (PTS)'}
dataf5["Product_Name_Subbrand"] = dataf5["Product_Name_Subbrand"].replace(subb_remap)

#%%
bda_brita = dataf5.copy()
bda_brita = bda_brita[['Geography_Name','Product_Name_Subbrand','iriprod']].drop_duplicates() 
bda_brita['Geography'] = bda_brita['Geography_Name'].str.upper().map(geog_map.set_index('Geography_Name')['Geography'])
bda_brita['Product_Name_Subbrand'] = bda_brita['Product_Name_Subbrand'].str.replace(" ", "")

pos_map['Product Name'] = pos_map['Product Name'].str.replace(" ", "")

# Map Product 
bda_brita['Sub_Brand'] = bda_brita['Product_Name_Subbrand'].map(pos_map.set_index('Product Name')['Subbrand Elasticity File'])
bda_brita['Brand'] = bda_brita['Product_Name_Subbrand'].map(pos_map.set_index('Product Name')['Brand Elasticity File'])
bda_brita['SBU'] = 'Brita'
bda_brita['Division'] = 'Specialty'

bda_brita.to_csv(qc+'bda_brita.csv')
bda_brita = bda_brita[bda_brita['Sub_Brand'].notnull()]

#%%
ppl_map = ppl_map[['Brand Elasticity File','Subbrand Elasticity File']].drop_duplicates()
bda_brita = ppl_map.merge(bda_brita, left_on=['Brand Elasticity File','Subbrand Elasticity File'],right_on=['Brand','Sub_Brand'],how='inner')
bda_brita.to_csv(qc+'bda_brita1.csv')

#%%
CoefDb_subb = CoefDb_All1.merge(bda_brita, left_on = ['Geography','iriprod'], right_on = ['Geography','iriprod'], how='left')
CoefDb_subb.to_csv(qc+'CoefDb_subb_Brita.csv')

#%%
CoefDb_subb_mapped = CoefDb_subb[CoefDb_subb['Sub_Brand'].notnull()]
CoefDb_subb_unmapped = CoefDb_subb[CoefDb_subb['Sub_Brand'].isnull()]

#%%
CoefDb_subb_unmapped = CoefDb_subb_unmapped[['catlib', 'Model_Period_End', 'CLOROX VS COMP', 'Geography', 
'iriprod', 'Promo_Price_Elasticity', 'Base_Price_Elasticity', 'Base_Statcase_Volume', 'Time Period', 'New iriprod']]
CoefDb_subb_mapped_1 = CoefDb_subb_unmapped.merge(pos_brita, left_on = ['Geography','New iriprod'], right_on = ['Geography','Product Key'], how='left')
CoefDb_subb_Brita = CoefDb_subb_mapped.append(CoefDb_subb_mapped_1, ignore_index = True)
CoefDb_subb_Brita.to_csv(qc+'CoefDb_subb_Brita1.csv')
CoefDb_subb = CoefDb_subb_Brita[CoefDb_subb_Brita['Sub_Brand'].notnull()]

#%%
# Ranking based on time period 
# Ranking should not include Geography. It'll mess up dashboard view
year_bria = CoefDb_subb[['Sub_Brand','Time Period']].drop_duplicates().reset_index(drop=True)
year_bria['Year'] = year_bria['Time Period'].apply(lambda x : x[2:4]).astype('int')
year_bria['Quarter'] = year_bria['Time Period'].apply(lambda x : x[5:6]).astype('int')
year_bria['rank'] = year_bria.sort_values(['Sub_Brand','Year','Quarter'], ascending = False).groupby(['Sub_Brand']).cumcount()+1
year_bria = year_bria.sort_values(['Sub_Brand','Year','Quarter']).reset_index(drop=True)

# QC evidence
year_bria.to_csv(qc+'year_bria.csv')

bda_raw_brita_all = pd.merge(CoefDb_subb, year_bria, on = ['Sub_Brand','Time Period'], how = 'left')
bda_raw_brita_all.to_csv(qc+'bda_raw_brita_all.csv')

#%%
bda_raw_brita_all['Flag_0.3'] = bda_raw_brita_all['Base_Price_Elasticity'].apply(lambda x: 1 if x==-0.3 else 0)
bda_raw_brita_all['Flag_5'] = bda_raw_brita_all['Base_Price_Elasticity'].apply(lambda x: 1 if x==-5.0 else 0)
bda_raw_brita_all.to_csv(qc+'bda_raw_brita_all1.csv')

#%%
#Differentiating the main file file for the 2 rank periods for z_scores.
D_f = bda_raw_brita_all.copy()
four = D_f[D_f['rank']<=4]
g_four = D_f[D_f['rank']>=5]

#%%
g_four['z_BPE']  = np.nan
g_four['z_PPE']  = np.nan
g_four['z_BSCV'] = np.nan

#%%
four['z_BPE'] = four.groupby(['iriprod','Geography']).Base_Price_Elasticity.transform(lambda x : zscore(x))
four['z_PPE'] = four.groupby(['iriprod','Geography']).Promo_Price_Elasticity.transform(lambda x : zscore(x))
four['z_BSCV'] = four.groupby(['iriprod','Geography']).Base_Statcase_Volume.transform(lambda x : zscore(x))

#%%
four['z_BPE'] = four['z_BPE'].replace(np.nan,0)
four['z_PPE'] = four['z_PPE'].replace(np.nan,0)
four['z_BSCV'] = four['z_BSCV'].replace(np.nan,0)

#%%
Result = four.append(g_four)
Result.to_csv(qc+'Result.csv')

# The result file will be the main feed file for Product and retailer level 

#%%
O_a = Result.copy()
O_a = O_a[ (O_a['Base_Price_Elasticity']==-5) | (O_a['Base_Price_Elasticity']==-0.3) ]
O_a.to_csv(qc+'Proxy_el_Pid.csv')

#%%
two_5 = Result.copy()
two_5 = two_5[(two_5['z_BPE']>=2.5) | (two_5['z_BPE'] <= -2.5)]
two_5.to_csv(qc+'exceed_std_pid.csv')

#%%
def sb_elas(x):
    d = {}
    d['BPE_by_channel'] =np.average(x['Base_Price_Elasticity'], weights=x['Base_Statcase_Volume'])
    d['PPE_by_channel'] =np.average(x['Promo_Price_Elasticity'], weights=x['Base_Statcase_Volume'])
    d['Base_Statcase_Volume'] = x['Base_Statcase_Volume'].sum() 
    d['Flag_0.3'] = x['Flag_0.3'].sum()
    d['Flag_5']   = x['Flag_5'].sum()
    return pd.Series(d, index=['BPE_by_channel','PPE_by_channel','Base_Statcase_Volume','Flag_0.3','Flag_5'])

#%%
sb_feed_channel = bda_raw_brita_all.groupby(['Division','Geography','Time Period','rank','SBU','Brand','Sub_Brand']).apply(sb_elas).reset_index()
sb_feed_channel.to_csv(qc+'sb_feed_channel.csv')

#%%
sb_feed_channel['Flag_0.3'] = sb_feed_channel['Flag_0.3'].apply(lambda x: 0 if x==0 else 1)
sb_feed_channel['Flag_5'] = sb_feed_channel['Flag_5'].apply(lambda x: 0 if x==0 else 1)
sb_feed_channel.to_csv(qc+'sb_feed_channel.csv')

#%%
D_f_sub = sb_feed_channel.copy()
four_sub = D_f_sub[D_f_sub['rank']<=4]
g_four_sub = D_f_sub[D_f_sub['rank']>=5]

#%%
g_four_sub['z_BPE']  = np.nan
g_four_sub['z_PPE']  = np.nan
g_four_sub['z_BSCV'] = np.nan

#%%
four_sub['z_BPE'] = four_sub.groupby(['Sub_Brand','Geography']).BPE_by_channel.transform(lambda x : zscore(x))
four_sub['z_PPE'] = four_sub.groupby(['Sub_Brand','Geography']).PPE_by_channel.transform(lambda x : zscore(x))
four_sub['z_BSCV'] = four_sub.groupby(['Sub_Brand','Geography']).Base_Statcase_Volume.transform(lambda x : zscore(x))

#%%
four_sub['z_BPE'] = four_sub['z_BPE'].replace(np.nan,0)
four_sub['z_PPE'] = four_sub['z_PPE'].replace(np.nan,0)
four_sub['z_BSCV'] = four_sub['z_BSCV'].replace(np.nan,0)

#%%
Result_sub = four_sub.append(g_four_sub)
Result_sub.to_csv(qc+'Result_sub.csv')

# the result_sub file is the main file at subbrand and retailer level

#%%
SB_lvl_qc=Result_sub[(Result_sub['z_BPE'] >= 2.5) | (Result_sub['z_BPE'] <= -2.5)]
SB_lvl_qc2=Result_sub[(Result_sub['BPE_by_channel'] == -5) | (Result_sub['BPE_by_channel'] == -0.3)]

#%%
SB_lvl_qc.to_csv(qc+'exceed_std_sb.csv')
SB_lvl_qc2.to_csv(qc+'Proxy_el_sb.csv')

#%%
# Jishnu - Rank has been removed from groupby for totalUS. This is done BPE_totalUS has same elasticities for a time period.    
sb_feed_totalUS_rank = sb_feed_channel[['Division','SBU','Brand','Sub_Brand','Time Period','rank']].drop_duplicates() 
    
#%%
sb_feed_BPE_totalUS = sb_feed_channel.groupby(['Division','SBU','Brand','Sub_Brand','Time Period']).apply(lambda x: np.average(x['BPE_by_channel'], weights=x['Base_Statcase_Volume'])).reset_index().rename(columns = {0:'BPE_TotalUS'})
sb_feed_PPE_totalUS = sb_feed_channel.groupby(['Division','SBU','Brand','Sub_Brand','Time Period']).apply(lambda x: np.average(x['PPE_by_channel'], weights=x['Base_Statcase_Volume'])).reset_index().rename(columns = {0:'PPE_TotalUS'})

#%%
# Merged with rank level information.    
sb_feed_BPE_totalUS = sb_feed_totalUS_rank.merge(sb_feed_BPE_totalUS, on = ['Division','SBU', 'Brand', 'Sub_Brand',
                                                                            'Time Period'], how = 'left')    
sb_feed_PPE_totalUS = sb_feed_totalUS_rank.merge(sb_feed_PPE_totalUS, on = ['Division','SBU', 'Brand', 'Sub_Brand',
                                                                            'Time Period'], how = 'left')   
    
#%%
sb_bda_BPE = pd.merge(sb_feed_channel,sb_feed_BPE_totalUS, on = ['Division','SBU', 'Brand', 'Sub_Brand','Time Period','rank'], how = 'left' )
sb_bda = pd.merge(sb_bda_BPE,sb_feed_PPE_totalUS, on = ['Division','SBU', 'Brand', 'Sub_Brand','Time Period','rank'], how = 'left' )
sb_bda.to_csv(output+'trended_bda_FY23Q3_Brita_uncured+3cured.csv')

# Uncured view manipulation complete - ranking based on subbrand

# =============================================================================

#%%
# Cured view - latest period
# =============================================================================
#Check if there is any new catlib for Brita before proceeding.
CoefDb_All = Brita_final.copy()

# Need to reset index so that Ranks can be assigned later (Avoid duplication of index for ranking)
CoefDb_All.reset_index(drop=True, inplace=True)
CoefDb_All.to_csv(qc+'CoefDb_All_check.csv')

#%%
# For Cured view - ranking based on geo and ret
#Select Latest 4 periods for all retailers and product keys
CoefDb_All['date'] = pd.to_datetime(CoefDb_All['Model_Period_End'],format='%Y-%m-%d')
CoefDb_All['year'] = pd.DatetimeIndex(CoefDb_All['date']).year
CoefDb_All['month'] = pd.DatetimeIndex(CoefDb_All['date']).month
CoefDb_All['Rank'] = CoefDb_All.sort_values(['Geography','iriprod','CLOROX VS COMP','year','month'], ascending = False).groupby(['Geography','iriprod', 'CLOROX VS COMP']).cumcount()+1
CoefDb_All.to_csv(qc+'CoefDb_All_ranked.csv')

#%%
CoefDb_All_Cl1 = CoefDb_All[CoefDb_All['Rank']<=4]
CoefDb_All_Cl1.to_csv(qc+'CoefDb_latest_4.csv')

#%%
CoefDb_All_Cl2 = CoefDb_All_Cl1.groupby(['iriprod','Geography']).apply(roll_a).reset_index()
CoefDb_All_F = CoefDb_All_Cl2.copy()
CoefDb_All_F.to_csv(qc+'CoefDb_All_F.csv')

#%%
#Latest 4 Period Aggregated
#1. Left Join 
POS_CoefDb_All = iri_df2.merge(CoefDb_All_F,left_on=['Product Key','Geography'],right_on=['iriprod','Geography'], how='left')
POS_CoefDb_All.to_csv(qc+'POS_CoefDb_All.csv')

#%%
#Filtering out mapped POS+BDA after Key-Mapping
POS_CoefDb_All_mapped = POS_CoefDb_All.loc[POS_CoefDb_All['iriprod'].notnull()]

# 1st df to be appended
POS_CoefDb_All_mapped.to_csv(qc+'POS_CoefDb_All_mapped.csv')

#%%
#Filtering out unmapped POS+BDA after Key-Mapping
POS_CoefDb_All_unmapped = POS_CoefDb_All.loc[POS_CoefDb_All['iriprod'].isnull()]
POS_CoefDb_All_unmapped.to_csv(qc+'POS_CoefDb_All_unmapped.csv')

#%%
# > $5000 Baseline Dollar Sales
POS_CoefDb_All_unmapped = POS_CoefDb_All_unmapped[POS_CoefDb_All_unmapped['Dollar Sales'] >= 5000].reset_index(drop=True)
POS_CoefDb_All_unmapped.to_csv(qc+'POS_CoefDb_All_unmapped1.csv')

#%%
#New Product Key = Product Key - 2nd last key
iri_df3 = POS_CoefDb_All_unmapped.drop(['iriprod', 'Promo_Price_Elasticity','Base_Price_Elasticity', 'Base_Statcase_Volume'], axis = 1)
iri_df3['new_split_pk'] = iri_df3['new'].apply(lambda x : [x[index] for index in [0,1,2,3,4,5,7]])
iri_df3['New Product Key'] = iri_df3['new_split_pk'].str.join(':')
iri_df3.drop(['new','new_split_pk'],axis=1,inplace=True)
iri_df3.to_csv(qc+"iri_df3.csv",index=False)

#%%
#New iriprod = iriprod - 2nd last key
CoefDb_All_F['new_split'] = CoefDb_All_F['iriprod'].str.split(':')
CoefDb_All_F['new_split_iri'] = CoefDb_All_F['new_split'].apply(lambda x : [x[index] for index in [0,1,2,3,4,5,7]])
CoefDb_All_F['New iriprod'] = CoefDb_All_F['new_split_iri'].str.join(':')
CoefDb_All_F.drop(['new_split','new_split_iri'],axis=1,inplace=True)
CoefDb_All_F.to_csv(qc+"CoefDb_All_F_new_iri_prod.csv",index=False)

#%%
def proxy_roll_a(x):
    d = {} 
    d['Base_Statcase_Volume'] = x['Base_Statcase_Volume'].mean()
    d['Promo_Price_Elasticity'] = np.average(x['Promo_Price_Elasticity'], weights=x['Base_Statcase_Volume'])
    d['Base_Price_Elasticity'] = np.average(x['Base_Price_Elasticity'], weights=x['Base_Statcase_Volume'])
    return pd.Series(d, index=['Promo_Price_Elasticity','Base_Price_Elasticity','Base_Statcase_Volume'])

#%%
#bda aggregation after Key - 2nd last key
CoefDb_All_F_Agg = CoefDb_All_F.groupby(['New iriprod','Geography']).apply(proxy_roll_a).reset_index()
CoefDb_All_F_Agg.to_csv(qc+"CoefDb_All_F_new_iri_prod1.csv",index=False)

#%%
#1. Left Join unmapped POS with BDA at Key - 2nd last key
POS_CoefDb_All_nw_pdt_key = iri_df3.merge(CoefDb_All_F_Agg, left_on=['New Product Key','Geography'], right_on=['New iriprod','Geography'], how='left')
POS_CoefDb_All_nw_pdt_key.to_csv(qc+"POS+Elasticity_nw_pdt_key.csv",index=False)

#%%
#Rule 1 completed - Appending Key - Mapped data with Key - 2nd last key mapped
POS_CoefDb_All_updated = POS_CoefDb_All_mapped.append(POS_CoefDb_All_nw_pdt_key)
POS_CoefDb_All_updated.to_csv(qc+'POS+Elasticity_RULE1.csv')

#%%
#Filtering out mapped POS+BDA after Key and Key-2nd last key Mapping
POS_CoefDb_All_updated_mapped = POS_CoefDb_All_updated.loc[POS_CoefDb_All_updated['iriprod'].notnull() | POS_CoefDb_All_updated['New iriprod'].notnull()] 
POS_CoefDb_All_updated_mapped.to_csv(qc+'POS_CoefDb_All_updated_mapped.csv')

#%%
#Filtering out unmapped POS+BDA after Key and Key-2nd last key Mapping
POS_CoefDb_All_updated_unmapped = POS_CoefDb_All_updated.loc[pd.isnull(POS_CoefDb_All_updated['iriprod']) & pd.isnull(POS_CoefDb_All_updated['New iriprod'])]
POS_CoefDb_All_updated_unmapped.to_csv(qc+'POS+Elasticity_updated_unmapped.csv')

#%%
#Filtering out unmapped POS+BDA after Key and Key-2nd last key Mapping having only Food and Mass Retailers
POS_CoefDb_All_unmapped_FOMA = POS_CoefDb_All_updated_unmapped[~POS_CoefDb_All_updated_unmapped['Geography'].isin(['TOTAL U.S. GROCERY', 
'Total US - Multi Outlet', 'Total Mass Aggregate', 'Total US - Drug', 'Petco Corp-RMA - Pet', "TOTAL U.S. SAMS CLUB", "BJ's Corp-RMA - Club"])]

#Filtering out unmapped POS+BDA after Key and Key-2nd last key Mapping having all Retailers/Channels except Food and Mass
POS_CoefDb_All_unmapped_TCP = POS_CoefDb_All_updated_unmapped[POS_CoefDb_All_updated_unmapped['Geography'].isin(['TOTAL U.S. GROCERY', 
'Total US - Multi Outlet', 'Total Mass Aggregate', 'Total US - Drug', 'Petco Corp-RMA - Pet', "TOTAL U.S. SAMS CLUB", "BJ's Corp-RMA - Club"])]

#%%
# #POS data for unmapped after Key Mapping
# iri_df4 = POS_CoefDb_All_unmapped_FOMA.drop(['iriprod','New iriprod','Promo_Price_Elasticity','Base_Price_Elasticity', 
#                                 'Base_Statcase_Volume'], axis = 1)
# iri_df4.rename(columns = {'Geography':'Geography_unmapped'},inplace=True)
# iri_df4.to_csv(qc+'iri_df4.csv')
# Geography_unmapped = iri_df4['Geography_unmapped'].unique()
# print(Geography_unmapped)

# #%%
# #Reading the geography proxy file. This file needs to be updated everytime there is a new unmapped Geography in iri_df4  
# geo_pxy  = pd.read_csv(path +'Mapping Files/'+'Geo Proxy Mapping.csv', low_memory=False)

# #%%
# # Iterating through the list of unmapped retailers
# POS_CoefDb_RULE2_0 = pd.DataFrame()
# for geo in Geography_unmapped:
#     print(geo)
#     iri_df4_Geo = iri_df4[iri_df4['Geography_unmapped'] == geo ] 
#     iri_df4_Geo =  iri_df4_Geo.merge(geo_pxy, on = ['Geography_unmapped'], how = 'inner')
#     POS_CoefDb_RULE2_0 = POS_CoefDb_RULE2_0.append([iri_df4_Geo.merge(CoefDb_All_F_Agg, left_on = ['New Product Key','Geography_Proxy'], right_on = 
#     ['New iriprod','Geography'], how = 'left')])

# POS_CoefDb_RULE2_0['Geography Proxy'] = 'Yes'
# POS_CoefDb_RULE2_0.to_csv(qc+'POS_CoefDb_RULE2_0.csv')

#%%
# #Filtering out only the BDA file information from the appended data
# CoefDb_RULE2 = POS_CoefDb_RULE2_0[['New iriprod','Geography_unmapped','Promo_Price_Elasticity','Base_Price_Elasticity', 
#                                    'Base_Statcase_Volume']]

# #Duplicates are formed in the BDA file as each retailer within a channel gets mapped to multiple retailers within a channel
# #Duplicates removed and BDA rolled once again 
# CoefDb_RULE2 = CoefDb_RULE2.drop_duplicates()
# CoefDb_RULE2_rolled = CoefDb_RULE2.groupby(['New iriprod','Geography_unmapped']).apply(proxy_roll_a).reset_index() 

# ######## IMPORTANT ##########
# # Even after the geo map, the iriprods are missng for the retailers so ignoring the below code upto 675

# #%%
# #Dropping Geo, Geo keys and BDA data from the appended dataframe
# POS_CoefDb_RULE2_1 = POS_CoefDb_RULE2_0.drop(['Geography','Geography_Proxy','Promo_Price_Elasticity',
#                                               'Base_Price_Elasticity', 'Base_Statcase_Volume','new'],axis=1)

# #Each retailer does not get mapped to all retailers within a channel. Dropping all such rows.
# POS_CoefDb_RULE2_1.dropna(subset = ["New iriprod"], inplace=True)

# #Duplicates on POS data fromed due to same reason as above. Those being dropped.
# POS_CoefDb_RULE2_1 = POS_CoefDb_RULE2_1.drop_duplicates()
            
# #%%
# #Left Join POS after duplicate removal with rolled up BDA. Completion of Rule 2
# # 3rd df to be appended
# POS_CoefDb_RULE2 = POS_CoefDb_RULE2_1.merge(CoefDb_RULE2_rolled, on=['New iriprod','Geography_unmapped'],how='left')
# POS_CoefDb_RULE2.rename(columns = {'Geography_unmapped':'Geography'},inplace=True)

# #%%
# #Dropping Geo, Geo keys and BDA data from the appended dataframe
# POS_CoefDb_RULE2_1_0 = POS_CoefDb_RULE2_0.drop(['Geography','Geography_Proxy','Promo_Price_Elasticity',
#                                               'Base_Price_Elasticity', 'Base_Statcase_Volume','new'],axis=1)

# #Each retailer does not get mapped to all retailers within a channel. Appending all such rows.
# POS_CoefDb_RULE2_1_0 = POS_CoefDb_RULE2_1_0[POS_CoefDb_RULE2_1_0['New iriprod'].isna()].reset_index(drop=True)

# #Duplicates on POS data fromed due to same reason as above. Those being dropped.
# POS_CoefDb_RULE2_1_0 = POS_CoefDb_RULE2_1_0.drop_duplicates()
# POS_CoefDb_RULE2_1_0.rename(columns = {'Geography_unmapped':'Geography'},inplace=True)

# #%%
# #Appending Unmapped Food data with Rule 2 data
# POS_CoefDb_RULE2 = POS_CoefDb_RULE2.append([POS_CoefDb_RULE2_1_0])

# #%%
# POS_CoefDb_RULE2['is_duplicate'] = POS_CoefDb_RULE2[['Geography','Product Key','Product']].duplicated()
# POS_CoefDb_RULE2_nd = POS_CoefDb_RULE2[POS_CoefDb_RULE2['is_duplicate']== False]
# POS_CoefDb_RULE2_d = POS_CoefDb_RULE2[POS_CoefDb_RULE2['is_duplicate']== True] 
# POS_CoefDb_RULE2_d = POS_CoefDb_RULE2_d[POS_CoefDb_RULE2_d['New iriprod'].notna()]
# POS_CoefDb_RULE2 = POS_CoefDb_RULE2_nd.append([POS_CoefDb_RULE2_d])
# POS_CoefDb_RULE2.to_csv(qc+'POS_CoefDb_RULE2.csv')

#%%
#Appending Unmapped Total, Club data with Rule 2 data
POS_CoefDb_RULE2_All = POS_CoefDb_All_updated_unmapped.copy()

#%%
#Appending mapped Key data with Rule 2 and Unmapped Total, Club data. Completion of Rule 1+2
POS_CoefDb_RULE12 = POS_CoefDb_All_updated_mapped.append([POS_CoefDb_RULE2_All])
POS_CoefDb_RULE12.to_csv(qc+'POS_CoefDb_RULE1+2.csv')

#%%
#--BDA file
# Custom product BDA manipulation
# Assign spreadsheet filename to `file`
file = 'CoefDB - All Total US FY19Q4.xlsx'
# Load spreadsheet
xl = pd.ExcelFile(bda_path+file)

# Load a sheet into a DataFrame bda_coeff_raw
bda_alter = xl.parse('CoefDb_ All Total US FY19Q4')

#extracting columns which are necessary
bda_catlib = bda_alter[['catlib','Model_Period_End','Product_Level','Geography_Name', 'Geography_Level', 
                            'model_source','iriprod','Product_Name_Current','Category_Name','prodkey'
                            ,'Base_Price_Elasticity','Promo_Price_Elasticity','Base_Statcase_Volume']]
bda_catlib = bda_catlib.replace('NULL', np.nan, regex=True)
bda_catlib = bda_catlib[bda_catlib['catlib'].isin(['B2','B4','B6','BB','BF','BS']) & bda_catlib['Product_Level'].isin(['S','X','K','Z','I'])]

#%%
#Mapping BDA to POS Retailers/Channels 
coeff_db_map = pd.read_excel(path+'Mapping Files/'+'Hyperion DB Channels.xlsx','Hyperion DB Channels')
bda_catlib = bda_catlib.merge(coeff_db_map, on = ['Geography_Name', 'Geography_Level', 'model_source'], how = 'left')
bda_catlib = bda_catlib[bda_catlib['IRI Channels'].isnull() == False]
bda_catlib.to_csv(qc+'bda_catlib.csv')

# Jishnu - This step was not there in previous code
bda_catlib['Geography'] = bda_catlib['Geography_Name'].str.upper().map(geog_map.set_index('Geography_Name')['Geography'])

#%%
#Creating Brand column
bda_catlib['Brand']='BRITA'
category_bda = bda_catlib['Category_Name'].unique()
print(category_bda)

#%%
bda_catlib['Product_Name_Current']=bda_catlib['Product_Name_Current'].str.upper()
bda_catlib['Geography']=bda_catlib['Geography'].str.upper()
bda_catlib['Category_Name']=bda_catlib['Category_Name'].str.upper()

#%%
#Creating Sub Brand column
bda_catlib['Sub_brand']=bda_catlib['Brand']+ " " +bda_catlib.apply(lambda x: 'FMF' 
                                                                   if  x['Category_Name']=='BRITA FM FILTER' 
          else 'FMS' if x['Category_Name']=='BRITA FM SYSTEM'
          else 'BF' if x['Category_Name']=='BRITA ON-THE-GO FILT'
          else 'BS' if x['Category_Name']=='BRITA ON-THE-GO BTTL'
          else 'PTF LEGACY'   if ('LEGACY'   in x['Product_Name_Current']) & (x['Category_Name']=='BRITA PT FILTER')
          else 'PTF STREAM'   if ('STREAM'   in x['Product_Name_Current']) & (x['Category_Name']=='BRITA PT FILTER')
          else 'PTF LONGLAST/ELITE' if ('LONGLAST' in x['Product_Name_Current']) & (x['Category_Name']=='BRITA PT FILTER') 
          else 'PTS LEGACY'   if ('LEGACY'   in x['Product_Name_Current']) & (x['Category_Name']=='BRITA PT SYSTEM')
          else 'PTS STREAM'   if ('STREAM'   in x['Product_Name_Current']) & (x['Category_Name']=='BRITA PT SYSTEM')
          else 'PTS LONGLAST/ELITE' if ('LONGLAST' in x['Product_Name_Current']) & (x['Category_Name']=='BRITA PT SYSTEM')
          else 'NA',axis=1)
bda_catlib.to_csv(qc+'bda_catlib1.csv')

#%%
bda_catlib_w_iriprod = bda_catlib[bda_catlib['iriprod'].isnull()==False].reset_index(drop=True)
bda_catlib_wo_iriprod = bda_catlib[bda_catlib['iriprod'].isnull()==True].reset_index(drop=True)

#%%
dataf3_1 = bda_catlib_wo_iriprod.merge(cust_agg_keys1, left_on=['prodkey'], right_on=['Custprod'], how = 'left')
dataf3_1['Base_Statcase_Volume2'] = dataf3_1.apply(lambda x: x['Base_Statcase_Volume'] if pd.isnull(x['Custprod'])==True
                      else x['Base_Statcase_Volume']/x['count'], axis=1)
dataf3_1.drop(['iriprod'],axis=1,inplace=True)
dataf3_1.rename(columns={'IRI_Product_Key':'iriprod'},inplace=True)
dataf3_1.to_csv(qc+'dataf3_1.csv')

#%%
dataf3 = dataf3_1.append([bda_catlib_w_iriprod])
dataf3['Base_Statcase_Volume'] = dataf3.apply(lambda x: x['Base_Statcase_Volume'] if pd.isnull(x['count'])==True
                      else x['Base_Statcase_Volume2'], axis=1)
dataf3.to_csv(qc+'dataf3.csv')

#%%
dataf4 = dataf3[['model_source', 'Geography_Level', 'Geography_Name', 'Geography', 'Model_Period_End',
    'Product_Level','catlib','Product_Name_Current','Product_Name','prodkey','Category_Name','Brand',
    'Sub_brand','CLOROX VS COMP','iriprod','Base_Price_Elasticity', 'Promo_Price_Elasticity','Base_Statcase_Volume']]
dataf6 = dataf4[dataf4['Base_Statcase_Volume']>0]
dataf6.to_csv(qc+'dataf6.csv')

#%%
dataf6['new'] = dataf6['iriprod'].str.split(':')
dataf6['len'] = dataf6['new'].str.len()
dataf6 = dataf6[dataf6['len'] == dataf6['len'].max()]
dataf6['new_subb'] = dataf6['new'].apply(lambda x : x[0:5])
dataf6['New iriprod subb'] = dataf6['new_subb'].str.join(':')
dataf6['new_mdl_size'] = dataf6['new'].apply(lambda x : x[5:8])
dataf6['New mdl size'] = dataf6['new_mdl_size'].str.join(':')
dataf6.rename(columns = {'iriprod':'old iriprod'},inplace=True)

#%%
brita_key_map = pd.read_excel(path+'Mapping Files/'+'Brita_Key_Mapping.xlsx', 'Brita')
bda_catlib_final = dataf6.merge(brita_key_map, left_on = ['New iriprod subb'], right_on = ['Subb IRI Key'], how ='left') 
bda_catlib_final['iriprod'] = bda_catlib_final['Subb Pdt Key'] + ":" + bda_catlib_final['New mdl size']
bda_catlib_final.drop(['new','new_subb','new_mdl_size'],axis=1,inplace=True)
bda_catlib_final.to_csv(qc+'bda_catlib_iri_corr.csv')

#%%
bda_catlib_final = bda_catlib_final[['Category_Name','Brand','Sub_brand','Geography','iriprod','Model_Period_End',
    'Product_Name_Current']].drop_duplicates()

#%%
bda_catlib = CoefDb_All_Cl1.merge(bda_catlib_final, left_on = ['Geography','iriprod','Model_Period_End'], right_on = ['Geography','iriprod','Model_Period_End'])
bda_catlib.to_csv(qc+'bda_catlib2.csv')

#%%
############################ IMPORTANT ################################################
# The category name KEY ACCOUNT RMAS is not present in bda_catalib so ignoring this piece of code.


# Filter Cat name = Key Acc RMAs for Sub_brand defn 
# bda_catlib_rma = bda_catlib[bda_catlib['Category_Name']=='KEY ACCOUNT RMAS'] 
#  #Creating Sub Brand column
# bda_catlib_rma['Sub_brand'] = bda_catlib_rma['Brand']+ " " + bda_catlib_rma.apply(lambda x: 'FMF' 
#                                                                    if  x['Subb Product']=='BRITA FAUCET MOUNT FILTERS (FMF)' 
#           else 'FMS' if x['Subb Product']=='BRITA FAUCET MOUNT FILTRATION SYSTEMS (FMS)'
#           else 'BF' if x['Subb Product']=='BRITA FILTERING BOTTLE FILTERS (FBF)'
#           else 'BS' if x['Subb Product']=='BRITA FILTERING BOTTLE FILTRATION SYSTEMS (FBS)'
#           else 'PTF LEGACY'   if x['Subb Product']=='BRITA LEGACY POUR THROUGH FILTERS (PTF)'
#           else 'PTF STREAM'   if x['Subb Product']=='BRITA STREAM POUR THROUGH FILTERS (PTF)'
#           else 'PTF LONGLAST/ELITE' if x['Subb Product']=='BRITA LONGLAST/ELITE POUR THROUGH FILTERS (PTF)'
#           else 'PTS LEGACY'   if x['Subb Product']=='BRITA LEGACY POUR THROUGH FILTRATION SYSTEMS (PTS)'
#           else 'PTS STREAM'   if x['Subb Product']=='BRITA STREAM POUR THROUGH FILTRATION SYSTEMS (PTS)'
#           else 'PTS LONGLAST/ELITE' if x['Category_Name']=='BRITA LONGLAST/ELITE POUR THROUGH FILTRATION SYSTEMS (PTS)'
#           else 'NA',axis=1)
# bda_catlib_rma.to_csv(qc+'bda_catlib_rma.csv')

#%%
#bda_catlib = bda_catlib[~bda_catlib['Category_Name'].isin(['KEY ACCOUNT RMAS'])] 
#bda_catlib = bda_catlib.append(bda_catlib_rma,ignore_index=True)
#bda_catlib.to_csv(qc+'bda_catlib3.csv')

#%%
#--------Filters
filters = bda_catlib[bda_catlib['Category_Name'].isin(['BRITA FM FILTER','BRITA ON-THE-GO FILT','BRITA PT FILTER'])]

#%%%
def size(x,y):
    #Size extracted from Product Name Current in BDA file
    size= re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',x)
    res = list(size)
    return res[-1]+" "+ "CT" 

#%%%
filters['size'] = filters.apply(lambda x: size(x['Product_Name_Current'],x['Category_Name']),axis=1)
filters.to_csv(qc+'filters.csv')

#%%
#Remove na subrands
filters_1 = filters[filters['Sub_brand']!='BRITA NA']
filters_1.to_csv(qc+'filters_1.csv')

#%%
# Roll bda measures for Filters Subbrand
def elas_agg(x):
        d = {}
        d['Base_Price_Elasticity'] = np.average(x['Base_Price_Elasticity'], weights = x['Base_Statcase_Volume'])
        d['Promo_Price_Elasticity'] = np.average(x['Promo_Price_Elasticity'], weights = x['Base_Statcase_Volume'])
        d['Base_Statcase_Volume']=x['Base_Statcase_Volume'].mean()
        return pd.Series(d, index=['Base_Price_Elasticity','Promo_Price_Elasticity','Base_Statcase_Volume'])

#%%%
bda_filters = filters_1.groupby(['Geography','Sub_brand','size']).apply(elas_agg).reset_index()
bda_filters.to_csv(qc+'bda_filters.csv')

#%%%
#--System---
system = bda_catlib[bda_catlib['Category_Name'].isin(['BRITA PT SYSTEM','BRITA FM SYSTEM','BRITA ON-THE-GO BTTL'])]
system_1 = system[system['Sub_brand']!='BRITA NA']
system_1.to_csv(qc+'system_1.csv')

#%%%
def sys_size(x,y):
    #Size extracted from Product Name Current in BDA file
    size= re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?',x)
    res = list(size)
    print(list(size))
    return res[0]+" "+ "CUPS" 

#%%%
system_1['size']=system_1.apply(lambda x: sys_size(x['Product_Name_Current'],x['Category_Name']),axis=1)

# Why 1? To join with level_0_w_bda 
system_1['size'] = system_1.apply(lambda x: "1" if (x['Category_Name'] in (['BRITA FM SYSTEM','BRITA ON-THE-GO BTTL'])) 
                                      else x['size'],axis=1)
system_1.to_csv(qc+'system_1.csv')

#%%
# Roll bda measures for System Subbrand
bda_system = system_1.groupby(['Geography','Sub_brand','size']).apply(elas_agg).reset_index()
bda_system.to_csv(qc+'bda_system.csv')

#Combine bda filter and system data
bda_final = bda_filters.append(bda_system,ignore_index = True)
bda_final.to_csv(qc+'bda_final.csv')

#%%
#Filtering out mapped POS+BDA after 'GEO-KEY MAP', 'GEO-SIZE MAP', 'GEO PROXY-SIZE MAP'
POS_CoefDb_RULE12_mapped = POS_CoefDb_RULE12.loc[POS_CoefDb_RULE12['Base_Price_Elasticity'].notnull()] 
POS_CoefDb_RULE12_mapped.to_csv(qc+'POS_CoefDb_RULE12_mapped.csv')

#%%
#Filtering out unmapped POS+BDA after 'GEO-KEY MAP', 'GEO-SIZE MAP', 'GEO PROXY-SIZE MAP'
POS_CoefDb_RULE12_unmapped = POS_CoefDb_RULE12.loc[POS_CoefDb_RULE12['Base_Price_Elasticity'].isnull()] 
POS_CoefDb_RULE12_unmapped.to_csv(qc+'POS_CoefDb_RULE12_unmapped.csv')

#%%
#POS data for unmapped after 'GEO-KEY MAP', 'GEO-SIZE MAP', 'GEO PROXY-SIZE MAP'
iri_df5 = POS_CoefDb_RULE12_unmapped.drop(['iriprod','New iriprod','Promo_Price_Elasticity','Base_Price_Elasticity', 
                                'Base_Statcase_Volume'], axis = 1)
# Segment level
SubCat_remap = {'POUR THROUGH':'PT', 'FAUCET MOUNT':'FM', 'FILTERING BOTTLE':'B'}
Segment_remap = {'FILTERS (PTF)' :'F','FILTRATION SYSTEMS (PTS)':'S','FILTERS (FMF)':'F',
                'FILTRATION SYSTEMS (FMS)':'S', 'FILTERS (FBF)':'F', 'FILTRATION SYSTEMS (FBS)':'S'}
iri_df5['Segment_level'] = iri_df5['Clorox Brand Value'] +" "+iri_df5['Clorox Sub Category Value'].replace(SubCat_remap) + iri_df5['Clorox Segment Value'].replace(Segment_remap)
print(iri_df5['Segment_level'].unique())

# SubBrand level
iri_df5['SubBrand_level'] = iri_df5['Segment_level'] + " " + iri_df5.apply(lambda x: x['Clorox SubBrand Value'].replace(x['Clorox Brand Value'],"").strip(),axis=1)
iri_df5['SubBrand_level'] = iri_df5.apply(lambda x: x['SubBrand_level'].strip(),axis=1)

# create dummy size column
# Why 1? To join with level_0_w_bda 
iri_df5['dummy_size'] = iri_df5.apply(lambda x: "1" if (x['Segment_level'] in (['BRITA BS','BRITA FMS'])) 
                                      else x['Clorox Size Value'],axis=1)

# There will be duplications at 'Geography','SubBrand_level','dummy_size' as different products 
iri_df5.to_csv(qc+'iri_df5.csv')

#%%       
# Merge POS and new BDA data
# iri_df5 + bda_final is joined on Geo+Subb+Size
# There will be duplications at 'Geography','SubBrand_level','dummy_size' as different products
POS_CoefDb_GSS = pd.merge(iri_df5, bda_final, left_on =['Geography','SubBrand_level','dummy_size'],
                                  right_on=['Geography','Sub_brand','size'],how = 'left' )
POS_CoefDb_GSS.to_csv(qc+'POS_CoefDb_GSS.csv')

#%% 
POS_CoefDb_RULE12 = POS_CoefDb_RULE12_mapped.append([POS_CoefDb_GSS])
POS_CoefDb_RULE12.to_csv(qc+'POS_CoefDb_RULE1+2_new.csv')

#%%
#Filtering out mapped POS+BDA after Key, Key-2nd last key and Subb Mapping
POS_CoefDb_RULE12_mapped = POS_CoefDb_RULE12.loc[POS_CoefDb_RULE12['Base_Price_Elasticity'].notnull()] 
POS_CoefDb_RULE12_mapped.to_csv(qc+'POS_CoefDb_RULE12_mapped.csv')

#%%
#Filtering out unmapped POS+BDA after Key, Key-2nd last key and Subb Mapping
POS_CoefDb_RULE12_unmapped = POS_CoefDb_RULE12.loc[pd.isnull(POS_CoefDb_RULE12['Base_Price_Elasticity'])]
POS_CoefDb_RULE12_unmapped.to_csv(qc+'POS_CoefDb_RULE12_unmapped.csv')

#%%
#Filtering out unmapped POS+BDA after Key, Key-2nd last key and Subb Mapping having only Food and Mass Retailers
POS_CoefDb_RULE12_unmapped_FOMA = POS_CoefDb_RULE12_unmapped[~POS_CoefDb_RULE12_unmapped['Geography'].isin(['TOTAL U.S. GROCERY', 
'Total US - Multi Outlet', 'Total Mass Aggregate', 'Total US - Drug', 'Petco Corp-RMA - Pet', "TOTAL U.S. SAMS CLUB", "BJ's Corp-RMA - Club"])]

#Filtering out unmapped POS+BDA after Key, Key-2nd last key and Subb Mapping having all Retailers/Channels except Food and Mass
POS_CoefDb_RULE12_unmapped_TCP = POS_CoefDb_RULE12_unmapped[POS_CoefDb_RULE12_unmapped['Geography'].isin(['TOTAL U.S. GROCERY', 
'Total US - Multi Outlet', 'Total Mass Aggregate', 'Total US - Drug', 'Petco Corp-RMA - Pet', "TOTAL U.S. SAMS CLUB", "BJ's Corp-RMA - Club"])]

#%%
# #%%
# #POS data for unmapped after Key, Key-2nd last key and Subb Mapping
# iri_df6 = POS_CoefDb_RULE12_unmapped_FOMA.drop(['iriprod','New iriprod','Promo_Price_Elasticity','Base_Price_Elasticity', 
#                                 'Base_Statcase_Volume','Sub_brand','size'], axis = 1)
# iri_df6.rename(columns = {'Geography':'Geography_unmapped'},inplace=True)
# iri_df6.to_csv(qc+'iri_df6.csv')
# Geography_unmapped = iri_df6['Geography_unmapped'].unique()
# print(Geography_unmapped)

# #%%
# #Reading the geography proxy file. This file needs to be updated everytime there is a new unmapped Geography in iri_df6  
# geo_pxy  = pd.read_csv(path +'Mapping Files/'+'Geo Proxy Mapping.csv', low_memory=False)

# #%%
# # Iterating through the list of unmapped retailers
# POS_CoefDb_RULE12_0 = pd.DataFrame()
# for geo in Geography_unmapped:
#     print(geo)
#     iri_df6_Geo = iri_df6[iri_df6['Geography_unmapped'] == geo ] 
#     iri_df6_Geo =  iri_df6_Geo.merge(geo_pxy, on = ['Geography_unmapped'], how = 'inner')
#     POS_CoefDb_RULE12_0 = POS_CoefDb_RULE12_0.append([iri_df6_Geo.merge(bda_final, left_on =['Geography_Proxy','SubBrand_level','dummy_size'],
#     right_on = ['Geography','Sub_brand','size'], how='left')])

# POS_CoefDb_RULE12_0['Geography Proxy 2.1'] = 'Yes'
# POS_CoefDb_RULE12_0.to_csv(qc+'POS_CoefDb_RULE12_0.csv')

# #%%
# #Filtering out only the BDA file information from the appended data
# CoefDb_RULE12 = POS_CoefDb_RULE12_0[['Geography_unmapped','Sub_brand','size','Promo_Price_Elasticity','Base_Price_Elasticity', 
#                                    'Base_Statcase_Volume']]

# #Duplicates are formed in the BDA file as each retailer within a channel gets mapped to multiple retailers within a channel
# #Duplicates removed and BDA rolled once again 
# CoefDb_RULE12 = CoefDb_RULE12.drop_duplicates()
# CoefDb_RULE12_rolled = CoefDb_RULE12.groupby(['Geography_unmapped','Sub_brand','size']).apply(elas_agg).reset_index() 

# #%%
# #Dropping Geo, Geo keys and BDA data from the appended dataframe
# POS_CoefDb_RULE2_10 = POS_CoefDb_RULE12_0.drop(['Geography','Geography_Proxy','Promo_Price_Elasticity',
#                     'Base_Price_Elasticity','Base_Statcase_Volume'],axis=1)

# #Each retailer does not get mapped to all retailers within a channel. Dropping all such rows.
# POS_CoefDb_RULE2_10.dropna(subset = ['Sub_brand','size'], inplace=True)

# #Duplicates on POS data fromed due to same reason as above. Those being dropped.
# POS_CoefDb_RULE2_10 = POS_CoefDb_RULE2_10.drop_duplicates()
            
# #%%
# #Left Join POS after duplicate removal with rolled up BDA. Completion of Geo Subb size Mapping
# # 3rd df to be appended
# POS_CoefDb_RULE121 = POS_CoefDb_RULE2_10.merge(CoefDb_RULE12_rolled, on=['Geography_unmapped','Sub_brand','size'],how='left')
# POS_CoefDb_RULE121.rename(columns = {'Geography_unmapped':'Geography'},inplace=True)

# #%%
# #Dropping Geo, Geo keys and BDA data from the appended dataframe
# POS_CoefDb_RULE12_1_0 = POS_CoefDb_RULE12_0.drop(['Geography','Geography_Proxy','Promo_Price_Elasticity',
#                                               'Base_Price_Elasticity', 'Base_Statcase_Volume'],axis=1)

# #Each retailer does not get mapped to all retailers within a channel. Appending all such rows.
# POS_CoefDb_RULE12_1_0 = POS_CoefDb_RULE12_1_0[POS_CoefDb_RULE12_1_0['Sub_brand'].isna()].reset_index(drop=True)

# #Duplicates on POS data fromed due to same reason as above. Those being dropped.
# POS_CoefDb_RULE12_1_0 = POS_CoefDb_RULE12_1_0.drop_duplicates()
# POS_CoefDb_RULE12_1_0.rename(columns = {'Geography_unmapped':'Geography'},inplace=True)

# #%%
# #Appending Unmapped Food data with Rule 2.1 data
# POS_CoefDb_RULE12 = POS_CoefDb_RULE121.append([POS_CoefDb_RULE12_1_0])
# POS_CoefDb_RULE12.to_csv(qc+'POS_CoefDb_RULE12.csv')

# #%%
# POS_CoefDb_RULE12['is_duplicate'] = POS_CoefDb_RULE12[['Geography','Product Key','Product']].duplicated()
# POS_CoefDb_RULE12_nd = POS_CoefDb_RULE12[POS_CoefDb_RULE12['is_duplicate']== False]
# POS_CoefDb_RULE12_d = POS_CoefDb_RULE12[POS_CoefDb_RULE12['is_duplicate']== True] 
# POS_CoefDb_RULE12_d = POS_CoefDb_RULE12_d[POS_CoefDb_RULE12_d['Sub_brand'].notna()]
# POS_CoefDb_RULE12 = POS_CoefDb_RULE12_nd.append([POS_CoefDb_RULE12_d])
# POS_CoefDb_RULE12.rename(columns ={'size_x':'size'},inplace=True)
# POS_CoefDb_RULE12.to_csv(qc+'POS_CoefDb_RULE12_GeoSubb.csv')

# #%%
# #Appending Unmapped Total, Club data with Rule 2.1 data
# POS_CoefDb_RULE12_All = POS_CoefDb_RULE12_unmapped_TCP.append([POS_CoefDb_RULE12])

# #REVISIT ABOVE BLOCKS AS CODE FAILS WHEN NOTHING GETS MAPPED. SHOULD BE IN TRY AND EXCEPT BLOCK.

#%%
POS_CoefDb_RULE12_All = POS_CoefDb_RULE12_unmapped.copy()

#%%
#Appending mapped Key data with Rule 2.1 and Unmapped Total, Club data. Completion of Rule 1+2+2.1
POS_CoefDb_RULE12 = POS_CoefDb_RULE12_mapped.append([POS_CoefDb_RULE12_All])
POS_CoefDb_RULE12.to_csv(qc+'POS_CoefDb_RULE1+2.csv')

#%%
# #Creating MAP STAT and MAP TYPE columns
# POS_CoefDb_RULE12['MAP STAT'] = np.where(POS_CoefDb_RULE12['Base_Price_Elasticity'].isnull(), 'UNMAP', 'MAP')

# conditions = [(POS_CoefDb_RULE12['Base_Price_Elasticity'].isnull()),

# (~POS_CoefDb_RULE12['Base_Price_Elasticity'].isnull() & 
# ~POS_CoefDb_RULE12['iriprod'].isnull() &
# POS_CoefDb_RULE12['Geography Proxy'].isnull()),

# (~POS_CoefDb_RULE12['Base_Price_Elasticity'].isnull() & 
# ~POS_CoefDb_RULE12['New iriprod'].isnull() &
# POS_CoefDb_RULE12['Geography Proxy'].isnull()),

# (~POS_CoefDb_RULE12['Base_Price_Elasticity'].isnull() & 
# ~POS_CoefDb_RULE12['New iriprod'].isnull() &
# ~POS_CoefDb_RULE12['Geography Proxy'].isnull()),

# (~POS_CoefDb_RULE12['Base_Price_Elasticity'].isnull() & 
# POS_CoefDb_RULE12['New iriprod'].isnull() &
# ~POS_CoefDb_RULE12['Sub_brand'].isnull() &
# POS_CoefDb_RULE12['Geography Proxy 2.1'].isnull()), 

# (~POS_CoefDb_RULE12['Base_Price_Elasticity'].isnull() & 
# POS_CoefDb_RULE12['New iriprod'].isnull() &
# ~POS_CoefDb_RULE12['Sub_brand'].isnull() &
# ~POS_CoefDb_RULE12['Geography Proxy 2.1'].isnull())]

# choices = ['UNMAP', 'GEO-KEY MAP', 'GEO-SIZE MAP', 'GEO PROXY-SIZE MAP', 'GEO-SUB-SIZE MAP', 'GEO PROXY-SUB-SIZE MAP']

# POS_CoefDb_RULE12['MAP TYPE'] = np.select(conditions, choices, default=np.nan)
# POS_CoefDb_RULE12.to_csv(qc+'POS_CoefDb_Brita.csv')

# #REVISIT ABOVE BLOCKS AS CODE FAILS WHEN NOTHING GETS MAPPED. SHOULD BE IN TRY AND EXCEPT BLOCK.

#%%
#Creating MAP STAT and MAP TYPE columns
POS_CoefDb_RULE12['MAP STAT'] = np.where(POS_CoefDb_RULE12['Base_Price_Elasticity'].isnull(), 'UNMAP', 'MAP')

conditions = [(POS_CoefDb_RULE12['Base_Price_Elasticity'].isnull()),

(~POS_CoefDb_RULE12['Base_Price_Elasticity'].isnull() & 
~POS_CoefDb_RULE12['iriprod'].isnull()),

(~POS_CoefDb_RULE12['Base_Price_Elasticity'].isnull() & 
~POS_CoefDb_RULE12['New iriprod'].isnull()),

(~POS_CoefDb_RULE12['Base_Price_Elasticity'].isnull() & 
~POS_CoefDb_RULE12['New iriprod'].isnull()),

(~POS_CoefDb_RULE12['Base_Price_Elasticity'].isnull() & 
POS_CoefDb_RULE12['New iriprod'].isnull())]

choices = ['UNMAP', 'GEO-KEY MAP', 'GEO-SIZE MAP', 'GEO PROXY-SIZE MAP', 'GEO-SUB-SIZE MAP']

POS_CoefDb_RULE12['MAP TYPE'] = np.select(conditions, choices, default=np.nan)
POS_CoefDb_RULE12.to_csv(qc+'POS_CoefDb_Brita.csv')

#%%
# New Product Key for Brand, Subbrand Mapping
POS_CoefDb_RULE12['new'] = POS_CoefDb_RULE12['Product Key'].str.split(':')
POS_CoefDb_RULE12['new_split_pk'] = POS_CoefDb_RULE12['new'].apply(lambda x : x[:-3])
POS_CoefDb_RULE12['New Product Key'] = POS_CoefDb_RULE12['new_split_pk'].str.join(':')
POS_CoefDb_RULE12.drop(['new','new_split_pk'],axis=1,inplace=True) 
      
#%%
POS_CoefDb_RULE12 = POS_CoefDb_RULE12.merge(pos_brita, left_on=['New Product Key','Geography'], 
                                            right_on = ['Product Key','Geography'], how='left')
POS_CoefDb_RULE12.to_csv(qc+'POS+Elasticity_Brita_w_subb.csv')

#%%
brita_clx = POS_CoefDb_RULE12[POS_CoefDb_RULE12['Clx_Comp'].isin(['Clorox'])
                    & ~ POS_CoefDb_RULE12['Base_Price_Elasticity'].isnull() & ~ POS_CoefDb_RULE12['Sub_Brand'].isnull()]

#%%
def sb_elas_latest(x):
    d = {}
    d['BPE_by_channel_latest'] =np.average(x['Base_Price_Elasticity'], weights=x['Base_Statcase_Volume'])
    d['PPE_by_channel_latest'] =np.average(x['Promo_Price_Elasticity'], weights=x['Base_Statcase_Volume'])
    d['Base_Statcase_Volume_latest'] = x['Base_Statcase_Volume'].sum() 
    return pd.Series(d, index=['BPE_by_channel_latest','PPE_by_channel_latest','Base_Statcase_Volume_latest'])

#%%
sb_bda_latest = sb_bda[sb_bda['rank']==1]
sb_bda_latest = sb_bda_latest[['Division','Time Period','rank','SBU','Brand','Sub_Brand']].drop_duplicates() 
brita_clx = brita_clx.merge(sb_bda_latest, on=['Division','SBU','Brand','Sub_Brand'], how ='left')
brita_clx.to_csv(qc+'brita_clx.csv')

#%%
sb_feed_channel_latest = brita_clx.groupby(['Division','Geography','Time Period','rank','SBU','Brand','Sub_Brand']).apply(sb_elas_latest).reset_index()
sb_feed_channel_latest.to_csv(qc+'sb_feed_channel_latest.csv')

#%%
# Jishnu - Rank has been removed from groupby for totalUS. This is done BPE_totalUS has same elasticities for a time period.    
sb_feed_totalUS_rank_latest = sb_feed_channel_latest[['Division','SBU','Brand','Sub_Brand','Time Period','rank']].drop_duplicates() 
    
#%%
sb_feed_BPE_totalUS_latest = sb_feed_channel_latest.groupby(['Division','SBU','Brand','Sub_Brand','Time Period']).apply(lambda x: np.average(x['BPE_by_channel_latest'], weights=x['Base_Statcase_Volume_latest'])).reset_index().rename(columns = {0:'BPE_TotalUS_latest'})
sb_feed_PPE_totalUS_latest = sb_feed_channel_latest.groupby(['Division','SBU','Brand','Sub_Brand','Time Period']).apply(lambda x: np.average(x['PPE_by_channel_latest'], weights=x['Base_Statcase_Volume_latest'])).reset_index().rename(columns = {0:'PPE_TotalUS_latest'})

#%%
# Merged with rank level information.    
sb_feed_BPE_totalUS_latest = sb_feed_totalUS_rank_latest.merge(sb_feed_BPE_totalUS_latest, on = ['Division','SBU', 'Brand', 'Sub_Brand',
                                                                            'Time Period'], how = 'left')    
sb_feed_PPE_totalUS_latest = sb_feed_totalUS_rank_latest.merge(sb_feed_PPE_totalUS_latest, on = ['Division','SBU', 'Brand', 'Sub_Brand',
                                                                            'Time Period'], how = 'left')   
    
#%%
sb_bda_BPE_latest = pd.merge(sb_feed_channel_latest , sb_feed_BPE_totalUS_latest, on = ['Division','SBU',
                                        'Brand', 'Sub_Brand','Time Period','rank'], how = 'left' )
sb_bda_latest = pd.merge(sb_bda_BPE_latest, sb_feed_PPE_totalUS_latest, on = ['Division','SBU', 
                                        'Brand', 'Sub_Brand','Time Period','rank'], how = 'left' )
sb_bda_latest.to_csv(output+'trended_bda_FY23Q3_Brita_latest_cured.csv')

# Uncured view latest period - complete
# =============================================================================

#%%
sb_bda_all = sb_bda.append(sb_bda_latest, ignore_index = True)
sb_bda_all.to_csv(qc+'sb_bda_all.csv')

#%%
sb_bda_all_3 = sb_bda_all[sb_bda_all['rank'].isin([4,3,2])]
sb_bpe_all_3_sum = sb_bda_all_3.groupby(['Division','Geography','SBU','Brand','Sub_Brand']).apply(lambda x: np.sum(x['BPE_by_channel'])).reset_index().rename(columns = {0:'BPE_by_channel_sum'})
sb_ppe_all_3_sum = sb_bda_all_3.groupby(['Division','Geography','SBU','Brand','Sub_Brand']).apply(lambda x: np.sum(x['PPE_by_channel'])).reset_index().rename(columns = {0:'PPE_by_channel_sum'})

#%%
sb_bpe_tus_3_sum = sb_bda_all_3.groupby(['Division','Geography','SBU','Brand','Sub_Brand']).apply(lambda x: np.sum(x['BPE_TotalUS'])).reset_index().rename(columns = {0:'BPE_TotalUS_sum'})
sb_ppe_tus_3_sum = sb_bda_all_3.groupby(['Division','Geography','SBU','Brand','Sub_Brand']).apply(lambda x: np.sum(x['PPE_TotalUS'])).reset_index().rename(columns = {0:'PPE_TotalUS_sum'})

#%%
sb_bda_all_3 = sb_bda_all_3.merge(sb_bpe_all_3_sum, on = ['Division','Geography','SBU','Brand','Sub_Brand'],how='left')
sb_bda_all_3 = sb_bda_all_3.merge(sb_ppe_all_3_sum, on = ['Division','Geography','SBU','Brand','Sub_Brand'],how='left')

#%%
sb_bda_all_3 = sb_bda_all_3.merge(sb_bpe_tus_3_sum, on = ['Division','Geography','SBU','Brand','Sub_Brand'],how='left')
sb_bda_all_3 = sb_bda_all_3.merge(sb_ppe_tus_3_sum, on = ['Division','Geography','SBU','Brand','Sub_Brand'],how='left')

#%%
sb_bda_all_3 = sb_bda_all_3[['Division', 'Geography','SBU', 'Brand','Sub_Brand','BPE_by_channel_sum','PPE_by_channel_sum',
                             'BPE_TotalUS_sum', 'PPE_TotalUS_sum']].drop_duplicates() 
sb_bda_all_dup = sb_bda_all.merge(sb_bda_all_3, on = ['Division','Geography','SBU','Brand','Sub_Brand'], how='left')

#%%
sb_bda_all_1 = sb_bda_all_dup[sb_bda_all_dup['rank'].isin([2,3,4]) | (sb_bda_all_dup['rank'].isin([1]) & sb_bda_all_dup['BPE_by_channel'].isnull())]
sb_bda_all_1.to_csv(qc+'sb_bda_all_1.csv')

#%%
sb_bda_all_1_unq_rnk = sb_bda_all_1.groupby(['Division','Geography',
'SBU','Brand','Sub_Brand'], sort=False)['rank'].nunique().reset_index().rename(columns={'rank':'unique_rnk'})
sb_bda_all_1_unq_rnk.to_csv(qc+'sb_bda_all_1_unq_rnk.csv')

#%%
sb_bda_all_1 = sb_bda_all_1.merge(sb_bda_all_1_unq_rnk, on=['Division','Geography','SBU','Brand','Sub_Brand'], how ='left')
sb_bda_all_1.to_csv(qc+'sb_bda_all_1_check.csv')

#%%
sb_bda_all_1['BPE_by_channel1'] = sb_bda_all_1.apply(lambda x: x['unique_rnk']*x['BPE_by_channel_latest'] - x['BPE_by_channel_sum']
            if pd.isna(x['BPE_by_channel']) else x['BPE_by_channel'], axis=1)
sb_bda_all_1['PPE_by_channel1'] = sb_bda_all_1.apply(lambda x: x['unique_rnk']*x['PPE_by_channel_latest'] - x['PPE_by_channel_sum']
            if pd.isna(x['PPE_by_channel']) else x['PPE_by_channel'], axis=1)
sb_bda_all_1['BPE_TotalUS1'] = sb_bda_all_1.apply(lambda x: x['unique_rnk']*x['BPE_TotalUS_latest'] - x['BPE_TotalUS_sum']
            if pd.isna(x['BPE_TotalUS']) else x['BPE_TotalUS'], axis=1)
sb_bda_all_1['PPE_TotalUS1'] = sb_bda_all_1.apply(lambda x: x['unique_rnk']*x['PPE_TotalUS_latest'] - x['PPE_TotalUS_sum']
            if pd.isna(x['PPE_TotalUS']) else x['PPE_TotalUS'], axis=1)
sb_bda_all_1.to_csv(qc+'sb_bda_all_1_final.csv')

#%%
sb_bda_all_gr_4 = sb_bda_all[sb_bda_all['rank']>4]
sb_bda_all_new = sb_bda_all[sb_bda_all['rank'].isin([1]) & sb_bda_all['BPE_by_channel'].notnull()]
sb_bda_all_2 = sb_bda_all_gr_4.append(sb_bda_all_new, ignore_index = True)
sb_bda_all = sb_bda_all_1.append(sb_bda_all_2, ignore_index = True)
sb_bda_all.to_csv(qc+'sb_bda_all_final.csv')

#%%
sb_bda_all['BPE_by_channel2'] = sb_bda_all.apply(lambda x: x['BPE_by_channel_latest']
            if pd.isna(x['BPE_by_channel1']) else x['BPE_by_channel1'], axis=1)
sb_bda_all['PPE_by_channel2'] = sb_bda_all.apply(lambda x:  x['PPE_by_channel_latest']
            if pd.isna(x['PPE_by_channel1']) else x['PPE_by_channel1'], axis=1)
sb_bda_all['BPE_TotalUS2'] = sb_bda_all.apply(lambda x: x['BPE_TotalUS_latest']
            if pd.isna(x['BPE_TotalUS1']) else x['BPE_TotalUS1'], axis=1)
sb_bda_all['PPE_TotalUS2'] = sb_bda_all.apply(lambda x: x['PPE_TotalUS_latest']
            if pd.isna(x['PPE_TotalUS1']) else x['PPE_TotalUS1'], axis=1)
sb_bda_all.drop(['BPE_by_channel_sum','PPE_by_channel_sum','BPE_TotalUS_sum', 'PPE_TotalUS_sum',
'Base_Statcase_Volume_latest', 'BPE_by_channel_sum', 'PPE_by_channel_sum', 'BPE_TotalUS_sum',
'PPE_TotalUS_sum','BPE_by_channel1','PPE_by_channel1','BPE_TotalUS1','PPE_TotalUS1','unique_rnk'],axis=1,inplace=True)
sb_bda_all.to_csv(output+'trended_bda_FY23Q3_Brita.csv')

#%%
sub_brand_bda = sb_bda_latest.copy()
sub_brand_bda.rename(columns={'BPE_by_channel_latest':'BPE_by_channel', 'PPE_by_channel_latest':'PPE_by_channel', 
                              'Base_Statcase_Volume_latest':'Base_Statcase_Volume', 
                              'BPE_TotalUS_latest':'BPE_TotalUS', 'PPE_TotalUS_latest':'PPE_TotalUS'},inplace=True)
sub_brand_bda.to_csv(qc+'sub_brand_bda_brita.csv')

#%%
b_feed_channel_latest = brita_clx.groupby(['Division','Geography','Time Period','rank','SBU','Brand']).apply(sb_elas_latest).reset_index()
b_feed_channel_latest.to_csv(qc+'b_feed_channel_latest_brita.csv')

#%%
# Jishnu - Rank has been removed from groupby for totalUS. This is done BPE_totalUS has same elasticities for a time period.    
b_feed_totalUS_rank_latest = b_feed_channel_latest[['Division','SBU','Brand','Time Period','rank']].drop_duplicates() 
    
#%%
b_feed_BPE_totalUS_latest = b_feed_channel_latest.groupby(['Division','SBU','Brand','Time Period']).apply(lambda x: np.average(x['BPE_by_channel_latest'], weights=x['Base_Statcase_Volume_latest'])).reset_index().rename(columns = {0:'BPE_TotalUS_latest'})
b_feed_PPE_totalUS_latest = b_feed_channel_latest.groupby(['Division','SBU','Brand','Time Period']).apply(lambda x: np.average(x['PPE_by_channel_latest'], weights=x['Base_Statcase_Volume_latest'])).reset_index().rename(columns = {0:'PPE_TotalUS_latest'})

#%%
# Merged with rank level information.    
b_feed_BPE_totalUS_latest = b_feed_totalUS_rank_latest.merge(b_feed_BPE_totalUS_latest, on = ['Division','SBU', 'Brand',
                                                                            'Time Period'], how = 'left')    
b_feed_PPE_totalUS_latest = b_feed_totalUS_rank_latest.merge(b_feed_PPE_totalUS_latest, on = ['Division','SBU', 'Brand',
                                                                            'Time Period'], how = 'left')   
    
#%%
b_bda_BPE_latest = pd.merge(b_feed_channel_latest , b_feed_BPE_totalUS_latest, on = ['Division','SBU',
                                        'Brand','Time Period','rank'], how = 'left' )
brand_bda = pd.merge(b_bda_BPE_latest, b_feed_PPE_totalUS_latest, on = ['Division','SBU', 
                                        'Brand','Time Period','rank'], how = 'left' )
brand_bda.rename(columns={'BPE_by_channel_latest':'BPE_by_channel', 'PPE_by_channel_latest':'PPE_by_channel', 
                          'Base_Statcase_Volume_latest':'Base_Statcase_Volume', 
                          'BPE_TotalUS_latest':'BPE_TotalUS', 'PPE_TotalUS_latest':'PPE_TotalUS'},inplace=True)
brand_bda.to_csv(qc+'brand_bda_brita.csv')

#%%
def pos_agg(x):
    d = {}
    d['Stat Case Baseline Volume'] = x['Stat Case Baseline Volume'].sum()
    d['Stat Case Volume'] = x['Stat Case Volume'].sum()
    d['Dollar Sales'] = x['Dollar Sales'].sum()
    d['Baseline Dollars'] = x['Baseline Dollars'].sum()
    d['Baseline Units'] = x['Baseline Units'].sum()
    d['Baseline Volume'] = x['Baseline Volume'].sum()
    return pd.Series(d, index=['Stat Case Baseline Volume','Stat Case Volume','Dollar Sales','Baseline Dollars',
    'Baseline Units','Baseline Volume'])

#%%
sub_brand_pos = POS_CoefDb_RULE12.groupby(['Division','SBU','Brand','Sub_Brand']).apply(pos_agg).reset_index()
brand_pos = POS_CoefDb_RULE12.groupby(['Division','SBU','Brand']).apply(pos_agg).reset_index()

sub_brand_pos['Retail Price'] = sub_brand_pos['Dollar Sales']/sub_brand_pos['Stat Case Volume']
brand_pos['Retail Price'] = brand_pos['Dollar Sales']/brand_pos['Stat Case Volume']

sub_brand_pos.to_csv(qc+'sub_brand_pos_brita.csv')
brand_pos.to_csv(qc+'brand_pos_brita.csv')

#%%
############# PPL Calculations ####################
# Sub brand Level PPL aggregation
def ppl_agg(x):
    d = {}
    d['Vol'] = x['Vol MSC'].sum()
    d['BCS'] = np.average(x['BCS'], weights=x['Vol MSC'])
    d['Net Real'] = np.average(x['Net Real'], weights=x['Vol MSC'])
    d['CPF'] = np.average(x['CPF'], weights=x['Vol MSC'])
    d['NCS'] = np.average(x['NCS'], weights=x['Vol MSC'])
    d['Contrib'] = np.average(x['Contrib'], weights=x['Vol MSC'])
    d['Gross Profit'] = np.average(x['Gross Profit'], weights=x['Vol MSC'])
    return pd.Series(d, index=['Vol','BCS','Net Real', 'CPF', 'NCS','Contrib', 'Gross Profit'])

#%%
ppl_map = map_xl.parse('PPL')
sub_brand_ppl = ppl_map.groupby(['Division', 'BU', 'Brand Elasticity File','Subbrand Elasticity File']).apply(ppl_agg).reset_index()
brand_ppl = ppl_map.groupby(['Division', 'BU', 'Brand Elasticity File']).apply(ppl_agg).reset_index()

# QC for PPLs
sub_brand_ppl.to_csv(qc+'sub_brand_ppl_brita.csv')
brand_ppl.to_csv(qc+'brand_ppl_brita.csv')

#%%
############# MERGING POS data AND PPL data #################
# Imp step. Check for duplications, blanks
sub_brand_pos_ppl = pd.merge(sub_brand_ppl, sub_brand_pos, left_on = ['Division', 'BU', 'Brand Elasticity File'
,'Subbrand Elasticity File'], right_on = ['Division','SBU','Brand','Sub_Brand'], how = 'left') 
brand_pos_ppl = pd.merge(brand_ppl, brand_pos, left_on = ['Division', 'BU', 'Brand Elasticity File'], 
                              right_on = ['Division','SBU','Brand'], how = 'left') 

# QC for PPLs with #NA POS data
sub_brand_pos_ppl.to_csv(qc+'sub_brand_pos_ppl_brita.csv')
brand_pos_ppl.to_csv(qc+'brand_pos_ppl_brita.csv')

#%%
sub_brand_pos_ppl.drop(['SBU','Brand','Sub_Brand'],axis=1,inplace=True)
brand_pos_ppl.drop(['SBU','Brand'],axis=1,inplace=True)
sub_brand_pos_ppl = sub_brand_pos_ppl[~sub_brand_pos_ppl['Stat Case Baseline Volume'].isnull()]
brand_pos_ppl = brand_pos_ppl[~brand_pos_ppl['Stat Case Baseline Volume'].isnull()]

sub_brand_pos_ppl.to_csv(qc+'sub_brand_pos_ppl_brita_1.csv')
brand_pos_ppl.to_csv(qc+'brand_pos_ppl_brita_1.csv')

#%%
#### Merging BDA, PPL, POS data for Brand and Sub Brands #######
# Imp step. Check for duplications, blanks
sub_brand_final = pd.merge(sub_brand_bda, sub_brand_pos_ppl, left_on=['Division','SBU','Brand','Sub_Brand'],
                           right_on = ['Division','BU','Brand Elasticity File','Subbrand Elasticity File'], how = 'right')
sub_brand_final.to_csv(qc+'sub_brand_final_brita.csv')

brand_final = pd.merge(brand_bda, brand_pos_ppl, left_on=['Division','SBU','Brand'], 
                       right_on = ['Division','BU','Brand Elasticity File'], how = 'right')
brand_final.to_csv(qc+'brand_final_brita.csv')

#%%
sub_brand_final['Product_Name'] = sub_brand_final['Subbrand Elasticity File']
sub_brand_final.drop(['BU','Brand Elasticity File','Subbrand Elasticity File'],axis=1, inplace = True)
brand_final['Product_Name'] = brand_final['Brand Elasticity File']
brand_final.drop(['BU','Brand Elasticity File'],axis=1, inplace = True)

#%%
#appending Brand and sub brand tables
ps_final = sub_brand_final.append(brand_final,ignore_index = True)
ps_final.to_csv(qc+'ps_final_brita.csv')