import pandas as pd
import numpy as np
import itertools 

def probab_when_two_nulls(df,attribute,value, sample_neck, sample_sleeve, sample_pattern):

    if(attribute=='neck'):
        product=list(itertools.product(sample_sleeve,sample_pattern))
        probablist=[]
        for i in product:
            probability = len(df[(df.neck==value) & (df.sleeve_length==i[0]) & (df.pattern==i[1])])/len(df[df.neck==value])
            probablist.append([value,i[0],i[1],probability])
        tempdf = pd.DataFrame(probablist,columns=['neck','sleeve_length','pattern','probability'])
        return(tempdf[tempdf['probability'].max()==tempdf.probability])

    elif(attribute=='sleeve_length'):
        product=list(itertools.product(sample_neck,sample_pattern))
        probablist=[]
        for i in product:
            probability = len(df[(df.neck==i[0]) & (df.sleeve_length==value) & (df.pattern==i[1])])/len(df[df.sleeve_length==value])
            probablist.append([i[0],value,i[1],probability])
        tempdf = pd.DataFrame(probablist,columns=['neck','sleeve_length','pattern','probability'])
        return(tempdf[tempdf['probability'].max()==tempdf.probability])
    else:
        product = list(itertools.product(sample_neck,sample_sleeve))
        probablist=[]
        for i in product:
            probability = len(df[(df.neck==i[0]) & (df.sleeve_length==i[1]) & (df.pattern==value)])/len(df[df.pattern==value])
            probablist.append([i[0],i[1],value,probability])
        tempdf = pd.DataFrame(probablist,columns=['neck','sleeve_length','pattern','probability'])
        return(tempdf[tempdf['probability'].max()==tempdf.probability])

def fill_one_missing_value(df,attribute,value1,value2):

    if (attribute=='neck'):
        sleeve=value1
        pattern=value2
        prob_list=[]
        for i in sample_neck:
            probability = len(df[(df.sleeve_length==sleeve)&(df.neck==i)&(df.pattern==value2)])/len(df[(df.sleeve_length==sleeve)&(df.pattern==pattern)])
            prob_list.append([i,sleeve,pattern,probability])
        tempdf = pd.DataFrame(prob_list,columns=['neck','sleeve_length','pattern','probability'])

        return(tempdf[tempdf['probability'].max()==tempdf.probability])

    if (attribute=='sleeve_length'):
        neck=value1
        pattern=value2
        prob_list=[]
        for i in sample_sleeve:
            probability=len(df[(df.sleeve_length==i)&(df.neck==value1)&(df.pattern==value2)])/len(df[(df.neck==neck)&(df.pattern==pattern)])
            prob_list.append([neck,i,pattern,probability])
        tempdf=pd.DataFrame(prob_list,columns=['neck','sleeve_length','pattern','probability'])

        return(tempdf[tempdf['probability'].max()==tempdf.probability])

    if (attribute=='pattern'):
        neck=value1
        sleeve=value2
        prob_list=[]
        for i in sample_pattern:
            probability=len(df[(df.sleeve_length==value2)&(df.neck==value1)&(df.pattern==i)])/len(df[(df.sleeve_length==sleeve)&(df.neck==neck)])
            prob_list.append([neck,sleeve,i,probability])
        tempdf=pd.DataFrame(prob_list,columns=['neck','sleeve_length','pattern','probability'])

        return(tempdf[tempdf['probability'].max()==tempdf.probability])

def fill_triple_na(df):
    """ Handles cases where all three attributes are null."""

    for i in df.index:
        row=pd.Series(df.loc[i,['neck','sleeve_length','pattern']])
        df.loc[i,'no_of_missing']=row.isnull().sum()
        df.loc[i,'all_fields_str']=str(df.loc[i,'neck'])+'-'+str(df.loc[i,'sleeve_length'])+'-'+str(df.loc[i,'pattern'])
        
    most_common=pd.DataFrame.from_dict(df['all_fields_str'].value_counts().to_dict(),
                                        orient='index',columns=['frequency_of_tshirt'],dtype='int')
    most_common['pattern_type']=most_common.index
    most_common.reset_index(inplace=True,drop=True)
    most_common_combination = most_common[most_common.frequency_of_tshirt==most_common['frequency_of_tshirt'].max()]
    values = most_common_combination.pattern_type[0].split("-")
    for i in df[df.no_of_missing==3.0].index:
            df.loc[i,'neck']=values[0]
            df.loc[i,'sleeve_length']=values[1]
            df.loc[i,'pattern']=values[2]
            df.loc[i,'no_of_missing']=0.0
    return df
    
def fill_double_na(df, neck_fillers, sleeve_fillers, pattern_fillers):
    """ Handles cases where two attribute values are null."""

    for i in df[df.no_of_missing==2.0].index:
        if(str(df.loc[i,'neck']).lower()!='nan'):
            a=df.loc[i,'neck']
            df.loc[i,'pattern']=neck_fillers[neck_fillers.neck==df.loc[i,'neck']]['pattern'].tolist()[0]
            df.loc[i,'sleeve_length']=neck_fillers[neck_fillers.neck==df.loc[i,'neck']]['sleeve_length'].tolist()[0]
            df.loc[i,'no_of_missing']=0.0
        if(str(df.loc[i,'sleeve_length']).lower()!='nan'):
            df.loc[i,'neck']=sleeve_fillers[sleeve_fillers.sleeve_length==df.loc[i,'sleeve_length']]['neck'].tolist()[0]
            df.loc[i,'pattern']=sleeve_fillers[sleeve_fillers.sleeve_length==df.loc[i,'sleeve_length']]['pattern'].tolist()[0]
        if(str(df.loc[i,'pattern']).lower()!='nan'):
            df.loc[i,'neck']=pattern_fillers[pattern_fillers.pattern==df.loc[i,'pattern']]['neck'].tolist()[0]
            df.loc[i,'sleeve_length']=pattern_fillers[pattern_fillers.pattern==df.loc[i,'pattern']]['sleeve_length'].tolist()[0]
    return df

def fill_single_na(df):
    """ Handles cases where single attribute value is null."""
    
    for i in df[df.no_of_missing==1].index:
        if(str(df.loc[i,'neck']).lower()=='nan'):
            temp=fill_one_missing_value(df,'neck',df.loc[i,'sleeve_length'],df.loc[i,'pattern'])
            temp.reset_index(inplace=True,drop=True)
            df.loc[i,'neck']=temp.loc[0,'neck']
        if(str(df.loc[i,'sleeve_length']).lower()=='nan'):
            temp=fill_one_missing_value(df,'sleeve_length',df.loc[i,'neck'],df.loc[i,'pattern'])
            temp.reset_index(inplace=True,drop=True)
            df.loc[i,'sleeve_length']=temp.loc[0,'sleeve_length']
        if(str(df.loc[i,'pattern']).lower()=='nan'):
            temp=fill_one_missing_value(df,'pattern',df.loc[i,'neck'],df.loc[i,'sleeve_length'])
            temp.reset_index(inplace=True,drop=True)
            df.loc[i,'pattern']=temp.loc[0,'pattern'] 
    return df

def impute_data(df):
    
    sample_neck=df.neck.dropna().unique().tolist()
    sample_neck_length=len(sample_neck)
    sample_sleeve=df.sleeve_length.dropna().unique().tolist()
    sample_sleeve_length=len(sample_sleeve)
    sample_pattern=df.pattern.dropna().unique().tolist()
    sample_pattern_length=len(sample_pattern)

    # All three missing.
    df = fill_triple_na(df)

    # Two attributes missing.
    neck_fillers = pd.DataFrame(dtype=float)
    sleeve_fillers = pd.DataFrame(dtype=float)
    pattern_fillers = pd.DataFrame(dtype=float)

    for i in sample_neck:
        prob = probab_when_two_nulls(df, 'neck', i, sample_neck, sample_sleeve, sample_pattern)
        neck_fillers = neck_fillers.append(prob)
    for i in sample_sleeve:
        prob = probab_when_two_nulls(df, 'sleeve_length', i, sample_neck, sample_sleeve, sample_pattern)
        sleeve_fillers = sleeve_fillers.append(prob)
    for i in sample_pattern:
        prob = probab_when_two_nulls(df, 'pattern', i, sample_neck, sample_sleeve, sample_pattern)
        pattern_fillers = pattern_fillers.append(prob)

    df = fill_double_na(df, neck_fillers, sleeve_fillers, pattern_fillers)

    # One attribute missing.
    df = fill_single_na(df)

    final_df = df[['filename', 'neck', 'sleeve_length', 'pattern']].copy()
    return final_df
