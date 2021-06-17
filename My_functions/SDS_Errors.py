#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:31:57 2020

@author: paolasouto
"""

####### IMPORT THE NECESSARY PACKAGES

import pandas as pd
import numpy as np
from datetime import datetime # MIRAR DONDE HACE FALTA
import pickle 
import os
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.interpolate import interp1d
import shutil # MIRAR DONDE HACE FALTA
import glob
from shapely.geometry import Polygon, LineString, mapping, MultiPoint
from shapely.ops import unary_union, polygonize, cascaded_union
import skimage.transform as transform # MIRAR DONDE HACE FALTA
#from mpl_toolkits.basemap import Basemap # MIARRA DONDE HACE FALTA
from osgeo import gdal, gdalconst, osr, ogr # MIRAR DONDE HACE FALTA
from osgeo.gdalconst import * # MIRAR DONDE HACE FALTA
import math # MIRAR DONDE HACE FALTA
import pickle
import datetime as dt
import pytz
import scipy

## TENGO QUE DEFINIR UNA FUNCIÓN QUE AUTOMÁTICAMENTE ME BUSCA CUALES SON LAS LINEAS SATELITARES 
## MAS PROXIMAS EN EL TIEMPO

def Error_by_inteporlation(RTK_shoreline,CoastSat_SH):
    
    # df = CoastSat_shoreline
    
    f = interp1d(RTK_shoreline['Lat'], RTK_shoreline['Lon'], kind = 'linear')
    
    X_interp = f(CoastSat_SH['y'])
    
    rmse = np.sqrt(np.sum((X_interp - CoastSat_SH['x'])**2)/len(CoastSat_SH))
    
    return rmse, X_interp, f

def Error_NearestPoints(RTK_SH, CoastSat_SH):
    
    lonn = []
    latt = []
    
    for x, y in zip(CoastSat_SH['x'], CoastSat_SH['y']):
        
        distt = []
        
        for xx, yy in zip(RTK_SH['Lon'], RTK_SH['Lat']):
            
            dist = np.sqrt((xx-x)**2 + (yy-y)**2)
            distt.append(dist)
            
        ind = distt.index(np.min(distt))
        lonn.append(RTK_SH['Lon'][ind])
        latt.append(RTK_SH['Lat'][ind])
        
        
    X_interp = pd.DataFrame({'x':lonn, 'y':latt})
    rmse = np.sqrt(np.sum((X_interp['x'] - CoastSat_SH['x'])**2)/len(CoastSat_SH))
    
    return rmse, X_interp


def AreaBetweenShorelines(SH1, SH2, opt,plot=False): # Añadir opc si quieres el plot o no
    
    df2 = SH2.iloc[::-1]
    polygon_shores = SH1.copy()
    polygon_shores = polygon_shores.append(df2)
    polygon_shores.reset_index(inplace = True, drop = True)
    polygon_shores = polygon_shores.append(polygon_shores.iloc[0])
    polygon_shores.reset_index(inplace=True, drop = True)
    polygon_shores = np.asanyarray(polygon_shores)
    
    # Compute the areas
    
    polygon = Polygon(polygon_shores)
    #area = polygon.area
    
    # Extract polygon x,y coordinates
    
    x,y = polygon.exterior.coords.xy #coords
    
    ## Added (In order to calculate the area in case the shorelines interact)
    
    ls = LineString(np.c_[x, y])
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    lr.is_simple # False
    mls = cascaded_union(lr) #unary_union
    mls.geom_type
    
    Area_cal = []
    
    for polygon in polygonize(mls):
        Area_cal.append(polygon.area)
        Area_poly = (np.asarray(Area_cal).sum())
        
    if plot:
        
        plt.figure(figsize=(7,4))
        plt.fill(x,y, alpha = 0.2, color='purple')
        #plt.plot(X_interp - Difference_shore, CoastSat_shoreline['y'], 'y.', label='Difference shoreline')
        plt.plot(SH1['x'], SH1['y'],'b.', label= opt[1]) #X_interp,CoastSat_shoreline['y']
        plt.plot(SH2['x'], SH2['y'], 'k.', label=opt[2]) #coastSat_shorelineDate['x'], coastSat_shorelineDate['y']
        #plt.plot(SH2['x'].iloc[::-1], SH2['y'].iloc[::-1], 'k.', label=opt[2]) #coastSat_shorelineDate['x'], coastSat_shorelineDate['y']

        plt.legend()
        #plt.text(max(SH1['x'])/1 -40 ,max(SH1['y'])/1 - 70, 'Area = '+ Area_poly.round(3).astype(str) + ' m$^2$', color='red',fontsize=12, bbox={'alpha': 0.1, 'pad': 10})
        plt.axis('off')
        
    return Area_poly
    

def GeneralizationFactor(SH_predicted, SH_accurate):
    
    """
        Generalization factor:
        A value of 1 will indicate no generalization, and a value smaller than 1 will indicate the amount of generalization.
        Lower resolution shoreline length / Higher resolution shoreline length

        Paola Souto Ceccon
        
        Arguments:
    -----------
        SH_accurate: Pandas DataFrame with the accurate shoreline
        SH_predicted: Pandas DataFrame with the predicted shoreline
        NOTE : The fields of lat lon has to be named as 'x' and 'y'
    
    Returns:
    -----------
        
        GF -----> Return a geopandas.DataFrame with the inf of the forecast 
            
    """
    
    shorelines = [SH_predicted, SH_accurate]
    GF_vec = []
    
    for shoreline in shorelines:
        
        if shoreline.columns.contains('Lon'):  #len(shoreline.columns) = 2
        
            shoreline = pd.DataFrame({'x' : shoreline['Lon'], 'y':shoreline['Lat']})

        shoreline.reset_index(drop=True, inplace=True) # for security

        SumDist = 0
        
        for i in range(0, len(shoreline['x'])-1):
            paso1 = [shoreline['x'][i], shoreline['y'][i]]
            paso2 = [shoreline['x'][i+1], shoreline['y'][i+1]]
            dist = np.sqrt((paso2[0]-paso1[0])**2 + (paso2[1]-paso1[1])**2)
            SumDist = SumDist + dist
            del dist
        GF_vec.append(SumDist)
        print(SumDist)
        
    print(GF_vec)
    GF = GF_vec[0]/GF_vec[1]
        
    return GF


def Read_CoastSat_SDS(dir_CoastSat, sitename):
        
    
    file_CoastSat_SH = sitename + '_output_points.geojson'
    
    dir_CoastSat_SH = os.path.join(dir_CoastSat, 'data', sitename ) #file_CoastSat_SH
    
    # Open the information
    
    CoastSat_SDS= gpd.read_file(dir_CoastSat_SH + os.sep + file_CoastSat_SH)
    CoastSat_SDS['day'] = pd.to_datetime(CoastSat_SDS['date'], utc=True).dt.date
    CoastSat_SDS['date'] = pd.to_datetime(CoastSat_SDS['date'], utc=True)
    
    return CoastSat_SDS

    

    
def Read_Measured_SH(dir_Measured, sitename):
    
    ## Generates a dictionary which stores all the measured shorelines in that site
    ## The keys are the date of the measured shoreline
    ## The items are the pandasDataframes
    
    dir_Measured_SH = dir_Measured + os.sep + sitename
    
    paths = []
    
    for file in [item for item in os.listdir(dir_Measured_SH) if sitename in item and item[-3:] == 'csv' or item == 'Digetized']:
        if file == 'Digetized':
            dir_Digetized = dir_Measured_SH + os.sep + 'Digetized'
            for file2 in [item2 for item2 in os.listdir(dir_Digetized) if sitename in item2]:
                paths.append(dir_Digetized + os.sep +file2)
            continue
        paths.append(dir_Measured_SH + os.sep + file)
        
    print(paths)
    
    Measured_shorelines = {pd.Timestamp(p[-14:-4], tz = 'utc').date(): pd.read_csv(p) for p in paths}
    # {p[-12:-4]: pd.read_csv(p) for p in paths}
    
    return Measured_shorelines


def Crop_SH (SH1, SH2): # SH1 = RTK_SH; SH2 = CoastSat_SDS
    
    ### Questa funziona solo per il momento con spiaggie più o meno rettilinee
      # e contando che RTK_shoreline sia più lunga di CoastSat_shoreline
    ### Con spiagge curve bisognerebbe trovare un'altro metodo
    
    min_y = SH1['Lat'].min()
    max_y = SH1['Lat'].max()
    min_x = SH1['Lon'].min()
    max_x = SH1['Lon'].max()
    
    
    SH2 = SH2[(SH2['x']<= max_x) & (SH2['x']>=min_x) & (SH2['y']<max_y) & (SH2['y']>=min_y)]
    SH2.reset_index(drop = True, inplace = True)
    
    return SH2

def Crop_by_Latitude(SH1, SH2):
    
    # SH1 must be the longest "shoreline"
    
    min_y = SH2['Lat'].min()
    max_y = SH2['Lat'].max()
    
    # SH2 has to be cropped to those latitude limits
    
    SH1 = SH1[(SH1['y']<=max_y) & (SH1['y']>=min_y)]
    
    return SH1


    

def Find_Sea_State(dir_waves, CoastSat_SDS, skip):
    
    ## This function works with the csv files downloaded from Dexter (station EMILIA skip=2)
    ## Will append to the geopandas DataFrame where the SDS are append the Hs, Tp, and Dir 
    ## variables registered by the Nautica boa at the time that the Satellite image was taken
    
    
    folders = ['Hs', 'Tp', 'Dir']
    
    def ToDates(filename):
        
        start_t, end_t, suff = filename.split('_')
        start_t = dt.datetime.strptime(start_t, '%Y-%m-%d')
        start_t = pytz.timezone('UTC').localize(start_t)
        end_t = dt.datetime.strptime(end_t, '%Y-%m-%d')
        end_t = pytz.timezone('UTC').localize(end_t)
        return start_t, end_t, suff
    
    wave_data = {f: [] for f in folders}
    
    for day_SDS in CoastSat_SDS['date']:
        
        
        for folder in folders:
            
            dir_variable = os.path.join(dir_waves, folder)
            
            for filename in [item for item in os.listdir(dir_variable) if item[-3:]=='csv']:
                
                start_t, end_t, suff = ToDates(filename)
                
                    
                if start_t.date() <= day_SDS.date() <= end_t.date():
                    
                    
                    vari = pd.read_csv(os.path.join(dir_waves, folder, filename), skiprows = skip, sep=',', usecols=range(0,3)) #3
                    
                    vari['bool'] = vari['Inizio validità (UTC)'].str.find('2')
                    vari = vari[vari['bool'] == 0]
                    vari.drop('bool', axis = 1, inplace = True )
                    
                    vari['Inizio validità (UTC)'] = pd.to_datetime(vari['Inizio validità (UTC)'], utc = True )
                    vari['Fine validità (UTC)'] = pd.to_datetime(vari['Fine validità (UTC)'], utc = True )
                    
                    interval = (vari[(vari['Inizio validità (UTC)']<=day_SDS) & (vari['Fine validità (UTC)']>=day_SDS)]) 
                    
                 
                    wave_data[folder].append(float(interval.iloc[0,2]))


  
                
                                        
    wave_data = pd.DataFrame(wave_data)
    wave_data.index = CoastSat_SDS.index
    CoastSat_SDS = pd.concat([CoastSat_SDS, wave_data], axis = 1)
    
    return CoastSat_SDS

#def FindDates_SDS_Measured(Measured_SH, CoastSat_SDS):
    
    #o = 0
    
    #for day in Measured_SH:
        
        #print('Date of the measured shorenline : ', day)
        
        #for day_SDS in CoastSat_SDS['day']:
            
            #if day == day_SDS: 
                
                #print('There is a SDS for this date: ', day_SDS)
                
                #CoastSat_SDS_day = pd.DataFrame(CoastSat_SDS['geometry'][o].coords, columns = ['x', 'y'])
                
                #o+= 1
                
                #yield CoastSat_SDS_day, Measured_SH[day]
                
                
def FindDates_SDS_Measured(Measured_SH, CoastSat_SDS):
    
    for day in Measured_SH:
        
        print('Date of the measured shoreline: ', day)
        
        if (day == CoastSat_SDS['day']).any():
            
            print('There is at least one SDS for this date')
            
            equivalent_SDS = CoastSat_SDS[CoastSat_SDS['day']==day]
            equivalent_SDS.reset_index( drop = True, inplace = True)

            
   
            for n in range(0, len(equivalent_SDS)):
                                
                CoastSat_SDS_day = pd.DataFrame(equivalent_SDS.loc[n,'geometry'].coords, columns = ['x','y'])
                
                #CoastSat_inf = pd.DataFrame({'day': equivalent_SDS.loc[n,'date'], 'Tp': equivalent_SDS.loc[n, 'Tp']}, index=[0])
                CoastSat_inf = pd.DataFrame({'day': [equivalent_SDS.loc[n,'date']], 
                                             'Hs':[ equivalent_SDS.loc[n, 'Hs']], 
                                             'Tp': [equivalent_SDS.loc[n, 'Tp']], 
                                             'Dir':[equivalent_SDS.loc[n, 'Dir']], 
                                             'Sat': [equivalent_SDS.loc[n, 'satname'] ], 
                                             'cloud_cover':[equivalent_SDS.loc[n,'cloud_cover']]}, 
                                            index=[0]) # 
                
                yield CoastSat_SDS_day, Measured_SH[day], CoastSat_inf
                
            
            

##################################################################################
##################################################################################
############### MINMUM SPANNING TREE ############################################

def minimum_spanning_tree(X, copy_X=True):
    """X are edge weights of fully connected graph"""
    if copy_X:
        X = X.copy()

    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices = X.shape[0]
    spanning_edges = []

    # initialize with node 0:                                                                                        
    visited_vertices = [0]                                                                                           
    num_visited = 1
    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf

    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)
        # 2d encoding of new_edge from flat, get correct indices                                                     
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]                                                      
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf                                                                    
        num_visited += 1
    return np.vstack(spanning_edges)

def define_breaking_points(edge_list, P):
        
    o=0
    index_break = []
    for i in range(0,len(edge_list)-1):
        o+=1
        if edge_list[i][1] == edge_list[i+1][0]:
            continue
        else:
            index_break.append(o)

    # Insertiamo la posizione 0
    index_break.insert(0,0)

    # Insertiamo la posizione finale
    index_break.append(len(edge_list))

    # Troviamo gli indici del path piu lungo

    ini = index_break[int(np.where(np.diff(index_break) == np.diff(index_break).max())[0])]
    fin = index_break[int(np.where(np.diff(index_break) == np.diff(index_break).max())[0])+1]


    edge_list2 = edge_list[ini:fin] 


    edges = []
    for edge in edge_list2:
        i, j = edge
        #plt.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], 'r.')
        edges.append([(P[i, 0], P[j, 0]), (P[i, 1], P[j, 1])])
        #edges.append([(P[i, 0], P[j, 0]), (P[i, 1], P[j, 1])])
    #plt.show()
    
    ## Definiamo il df
    
    x = []
    y = []

    for coor in edges:
        x.append(coor[0][0])
        x.append(coor[0][1])
        y.append(coor[1][0])
        y.append(coor[1][1])

    df = pd.DataFrame({'cross':x, 'along':y})


    return df


def Geographical_MST(idx, CoastSat_SDS):
    
    "This fucking thing it's still not stable, but show must go one (it works in this case)"
    
    P = np.array(CoastSat_SDS['geometry'][idx].coords)
    
    X = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(P, metric='euclidean'))

    edge_list = minimum_spanning_tree(X)
    
    df = define_breaking_points(edge_list, P)
    
    return df
    

     
                
