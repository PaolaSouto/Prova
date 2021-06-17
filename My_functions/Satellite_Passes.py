#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:56:48 2020

@author: paolasouto
"""

# Load modules

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import pytz
from datetime import date
import datetime as dt


# earth engine modules

import ee
from urllib.request import urlretrieve
import copy

# additional modules
from datetime import datetime
import pytz

## Open files from url
import mechanize
from time import sleep
import cgi

## Read kml as geopandasDataFrame and study geometry
import geopandas as gpd
import fiona
import requests

# Read line by line kml
from fastkml import kml

# Study position for the polygon in satellite trajectories

from shapely.geometry import Point, shape, Polygon




##############################################################
######################  FUTURE PASSES ########################
##############################################################




def ESA_S2_AcquisitionPlans(l, br ,df_Acquisitions):
    
    referer = 'https://sentinel.esa.int/web/sentinel/missions/sentinel-2/acquisition-plans'
    
    
    """
    Find all the files contained in a url, and read it. In case it's necessary to download them
    it will be necessary to uncomment the second part
    
    Paola Souto Ceccon
    
    Arguments:
    -----------
    l: mechanize._html.Link
        Each of the links contained on the page
    br: string
        
    df_Acquisitions: empty pandas.DataFrame

    Returns:
    -----------
    
    df_Acquisitions -----> Return a geopandas.DataFrame with the inf of the forecast 
        
    if SECOND PART UNCOMMENT : download the files in the current directory


    """
    
    
    
    # 1. CLICK AND OPEN THE LINKS
    
    r = br.click_link(l) # r is the files link
    r.add_header("Referer", referer) # add a referer header, just in case
    response = br.open(r)
    
    print(r)
    
    # 2. KML INFORMATION TO GEOPANDASDATAFRAME (a new function?)
    
    gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'

    re = requests.get(l.absolute_url)
    #print(re.content)

    # convert the page content to bytes
    f =  fiona.BytesCollection(bytes(re.content))

    # empty GeoDataFrame
    df = gpd.GeoDataFrame()

    # iterate over layers
    for layer in fiona.listlayers(f.path):
        s = gpd.read_file(f.path, driver='KML', layer=layer)
        df = df.append(s, ignore_index=True)
    
        
    # modify df
    
    #with open(l.absolute_url, 'rt', encoding="utf-8") as myfile:
        #doc=myfile.read()
    
    ## Read kml structure
    
    name = []
    timeStart = []
    timeStop =  []
    for line in re.content.decode("utf-8").split('\n'):
        if 'name' in line and len(line)==25 and line.split('<name>')[1].split('</name>')[0][0]=='S':
            continue
        if '<name>' in line and (len(line)==25 or len(line)==38 )and isinstance(int(line.split('<name>')[1].split('</name>')[0][0]), int):
            name.append(line.split('<name>')[1].split('</name>')[0])
        if '<begin>' in line:
            timeStart.append(line.split('<begin>')[1].split('</begin>')[0])
        if '<end>' in line:
            timeStop.append(line.split('<end>')[1].split('</end>')[0])
            
    print(len(name))
            
    if (np.unique(name == df['Name'])) and (len(name)==len(df)):
        df = df.drop(['Description'], axis=1)
        df['timeStart'] = timeStart
        df['timeStop'] = timeStop
    else:
        print('Something has gone wrong')
        

    df_Acquisitions = df_Acquisitions.append(df)
    
    
    
    ##### THIS WILL BE USED FOR DOWNLOAD THE FILES
    
    #filename = linkUrl.attrs[0][1].split('/')[-1]
    #f = open(filename, "wb") #TODO: perhaps ensure that file doesn't already exist?
    #f.write(response.read()) # write the response content to disk
    #print(filename," has been downloaded")
    br.back()
    
    
    print('The process has finished')

    return df_Acquisitions

def ReadInformation_AcquisitionPlans():
    
    
    url_passes = 'https://sentinel.esa.int/web/sentinel/missions/sentinel-2/acquisition-plans'
    br = mechanize.Browser()
    br.open(url_passes)
    
    # Broser options 
    br.set_handle_equiv( True ) 
    br.set_handle_gzip( True ) 
    br.set_handle_redirect( True ) 
    br.set_handle_referer( True ) 
    br.set_handle_robots( False ) 
    br.set_handle_refresh( mechanize._http.HTTPRefreshProcessor(), max_time = 1 ) 
    br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')] # masquerade as a real browser. this is not nice to do though.
        
        
    #print "Get all kml links\n and take only the one whith dates over the actual one"
    filetypes=["kml"] # the kind of file could be PDF, pdf, csv etc
    myfiles = []
    for l in br.links():
        if filetypes[0] in l.url and l.url[-3:]==filetypes[0]:
            arr_str = l.text.split(' ')
            date_pre_str = ' '.join(arr_str[-3:])
            dt_pre = datetime.strptime(date_pre_str, '%d %B %Y')
            #myfiles.append(l)
            if dt_pre > datetime.today():
                myfiles.append(l)
        #check if this link has the file extension or text we want
        #myfiles.extend([l for t in filetypes if (t in l.url and l.url[-3:]==t)]) # or t in l.text
        
    # initializes empty DataFrame df_Acquisitions
    
    df_Acquisitions = pd.DataFrame()
    
    for l in myfiles:
    # for index, l in zip(range(100), myfiles): # <--- uncomment this line (and coment the one above) to download 100 links.
        #sleep(1) # uncomment to throttle downloads, so you dont hammer the site
        df_Acquisitions = ESA_S2_AcquisitionPlans(l, br ,df_Acquisitions)
            
    
    return df_Acquisitions

def DaysSentinelArea(polygon):
    
    """
    Gives the dates in which Sentinel2 will pass across the desidered area according to the ESA plans acquisition
    
    Paola Souto Ceccon
    
    Arguments:
    -----------
    polygon: list with the bounding box coordinates of the area of interest (SEE IF COULD BE DONE BY READING THE POLYGONS WITH THE COASTSAT FUNCTION)

    Returns:
    -----------
    
    Sentinel_In_Area -----> pandas DataFrame  
        
    if SECOND PART UNCOMMENT : download the files in the current directory


    """
    
    polygon2 = [tuple(t) for t in polygon[0]]
    poly_beach = Polygon(polygon2)
    
    df_Acquisitions = ReadInformation_AcquisitionPlans()
    df_Acquisitions['BeachBoolean'] = df_Acquisitions['geometry'].contains(poly_beach)
    Sentinel_In_Area = df_Acquisitions[df_Acquisitions['BeachBoolean']].copy()
    Sentinel_In_Area.drop(['geometry'], axis=1, inplace=True)
    Sentinel_In_Area ['timeStart'] = pd.to_datetime(Sentinel_In_Area['timeStart'])
    Sentinel_In_Area['timeStop'] = pd.to_datetime(Sentinel_In_Area['timeStop'])
    Sentinel_In_Area.drop_duplicates(subset=['Name','timeStart', 'timeStop'],inplace=True)
    Sentinel_In_Area.sort_values(by=['timeStart'], inplace=True)
    Sentinel_In_Area.drop(['BeachBoolean'], axis=1, inplace=True)
    Sentinel_In_Area=Sentinel_In_Area[Sentinel_In_Area.timeStart>datetime.now().astimezone()]
    #Sentinel_In_Area.index = range(0,len(Sentinel_In_Area))
    #Sentinel_In_Area.index = pd.RangeIndex(start=0,stop=len(Sentinel_In_Area), step=1)
    Sentinel_In_Area.reset_index(inplace=True, drop=True)
    
    Sentinel_In_Area['timeStart'] = Sentinel_In_Area['timeStart'].dt.tz_convert('Europe/Rome')
    Sentinel_In_Area['timeStop'] = Sentinel_In_Area['timeStop'].dt.tz_convert('Europe/Rome')
    
    return Sentinel_In_Area


    

##############################################################
######################  OLD PASSES ########################
##############################################################
    
def Retrive_images(inputs):
    """""
    Modify from CoastSat (KV WRL 2018) by PSC
    
    Find the dates in which the satellites Landsat 5, Landsat 7, Landsat 8 and Sentinel-2 
    has passed over one specific and acquired between specific dates.
    
    ############### BUG IN THE STEP WHERE WE PASS FROM im_all to im_all_upadated (check)
    
        Arguments:
    -----------
        inputs: dict 
            dictionnary that contains the following fields:
        'sitename': str
            String containig the name of the site
        'polygon': list
            polygon containing the lon/lat coordinates to be extracted,
            longitudes in the first column and latitudes in the second column,
            there are 5 pairs of lat/lon with the fifth point equal to the first point.
            e.g. [[[151.3, -33.7],[151.4, -33.7],[151.4, -33.8],[151.3, -33.8],
            [151.3, -33.7]]]
        'dates': list of str
            list that contains 2 strings with the initial and final dates in format 'yyyy-mm-dd'
            e.g. ['1987-01-01', '2018-01-01']
        'sat_list': list of str
            list that contains the names of the satellite missions to include 
            e.g. ['L5', 'L7', 'L8', 'S2']
        'filepath_data': str
            Filepath to the directory where the file with dates will be stored
    
    Returns:
    -----------
        metadata: dict
            contains the information about the satellite images that were downloaded: filename, 
            georeferencing accuracy and image coordinate reference system 
    """
    
    
    # Initialise the connection with GEE server
    
    ee.Initialize()
    
    # Read inputs dictionary
    
    polygon = inputs['polygon']
    dates = inputs['dates']
    sat_list = inputs['sat_list']   
    

        
    Satellite_old_passes = pd.DataFrame({'Date':[], 'Cloud_cover':[], 'Sat':[]}) #, 'ID':[]
        
    for n in sat_list:
    
    #=============================================================================================#
    # Searching L5 images
    #=============================================================================================#
    
        if 'L5' in n or 'Landsat5' in n:
            
            
            satname = 'L5'
            
            # Landsat 5 collection
            count_loop = 0
            while count_loop < 1:
                try:
                    input_col = ee.ImageCollection('LANDSAT/LT05/C01/T1_TOA')
                    # filter by location and dates
                    flt_col = input_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(dates[0],dates[1])
                    # get all images in the filtered collection
                    im_all = flt_col.getInfo().get('features')
                    count_loop = 1
                except:
                    count_loop = 0
                    
            # get cloud percentage
            cloud_cover = [_['properties']['CLOUD_COVER'] for _ in im_all]
    
            
            im_col = im_all
            n_img = len(im_all)
            
            #print('%s: %d images'%(satname,n_img)) 
            
            # loop trough images
            timestamps = []
            acc_georef = []
            filenames = []
            all_names = []
            im_epsg = []
            for i in range(n_img):
                count_loop = 0
                while count_loop < 1:
                    try:
                        # find each image in ee database
                        im = ee.Image(im_col[i]['id'])
                        count_loop = 1
                    except: 
                        count_loop = 0 
                
                        
                # read metadata
                im_dic = im_col[i]
                
                        
                t = im_dic['properties']['system:time_start']
                
                # convert to datetime
                im_timestamp = datetime.fromtimestamp(t/1000, tz=pytz.utc)
                timestamps.append(im_timestamp)
                im_date = im_timestamp.strftime('%Y-%m-%d-%H-%M-%S')
            sat_list = ['L5'] * len(timestamps)
                
            #print(len(timestamps), len(cloud_cover), len(sat_list))    
            kk = pd.DataFrame({'Date':timestamps, 'Cloud_cover':cloud_cover, 'Sat':sat_list})
            Satellite_old_passes = Satellite_old_passes.append(kk)
            del kk
            
            
        #=============================================================================================#
        # Searching L7 images
        #=============================================================================================#
        
        
        if 'L7' in n or 'Landsat7' in n:
            
            
            satname = 'L7'
            
            # Landsat 7 collection
            
            count_loop = 0
            
            while count_loop < 1:
                try:
                    input_col = ee.ImageCollection('LANDSAT/LE07/C01/T1_TOA') # TOP OF THE ATMOSPHERE
                    #input_col = ee.ImageCollection('LANDSAT/LE07/C01/T1')
                    #filter by location and date
                    flt_col = input_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(dates[0],dates[1])
                    # get all the images in the filtered collection
                    im_all = flt_col.getInfo().get('features')
                    count_loop = 1
                except:
                    count_loop = 0
                    
            # cloud cover
            cloud_cover = [_['properties']['CLOUD_COVER'] for _ in im_all]
            
            im_col = im_all
            n_img = len(im_all)
            
            #print('%s: %d images'%(satname,n_img)) 
            
            # loop trough images
            timestamps = []
            acc_georef = []
            filenames = []
            all_names = []
            im_epsg = []
            for i in range(n_img):
                
                count_loop = 0
                while count_loop < 1:
                    try:
                        # find each image in ee database
                        im = ee.Image(im_col[i]['id'])
                        count_loop = 1
                    except: 
                        count_loop = 0            
                # read metadata
                im_dic = im_col[i]
                
                            # get time of acquisition (UNIX time)
                t = im_dic['properties']['system:time_start']
                # convert to datetime
                im_timestamp = datetime.fromtimestamp(t/1000, tz=pytz.utc)
                timestamps.append(im_timestamp)
                im_date = im_timestamp.strftime('%Y-%m-%d-%H-%M-%S')
            sat_list = ['L7'] * len(timestamps)
                
            kk = pd.DataFrame({'Date':timestamps, 'Cloud_cover':cloud_cover, 'Sat':sat_list})
            Satellite_old_passes = Satellite_old_passes.append(kk)
            del kk
            
             
        
        #=============================================================================================#
        # Searching L8 images
        #=============================================================================================#
        

            
########
        if 'L8' in n or 'Landsat8' in n:
            
            
            satname = 'L8'
            
            # Landsat 5 collection
            count_loop = 0
            while count_loop < 1:
                try:
                    input_col = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')
                    # filter by location and dates
                    flt_col = input_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(dates[0],dates[1])
                    # get all images in the filtered collection
                    im_all = flt_col.getInfo().get('features')
                    count_loop = 1
                except:
                    count_loop = 0
                    
            # get cloud percentage
            cloud_cover = [_['properties']['CLOUD_COVER'] for _ in im_all]
    
            
            im_col = im_all
            n_img = len(im_all)
            
            #print('%s: %d images'%(satname,n_img)) 
            
            # loop trough images
            timestamps = []
            acc_georef = []
            filenames = []
            all_names = []
            im_epsg = []
            for i in range(n_img):
                count_loop = 0
                while count_loop < 1:
                    try:
                        # find each image in ee database
                        im = ee.Image(im_col[i]['id'])
                        count_loop = 1
                    except: 
                        count_loop = 0 
                
                        
                # read metadata
                im_dic = im_col[i]
                
                        
                t = im_dic['properties']['system:time_start']
                
                # convert to datetime
                im_timestamp = datetime.fromtimestamp(t/1000, tz=pytz.utc)
                timestamps.append(im_timestamp)
                im_date = im_timestamp.strftime('%Y-%m-%d-%H-%M-%S')
                
            sat_list = ['L8'] * len(timestamps) # hemos quitado la indentacion
                
            kk = pd.DataFrame({'Date':timestamps, 'Cloud_cover':cloud_cover, 'Sat':sat_list})
            Satellite_old_passes = Satellite_old_passes.append(kk)
            del kk
            
            
                
        #=============================================================================================#
        # Searching S2 images
        #=============================================================================================#
        
        if 'S2' in n or 'Sentinel2' in n:
            
            
            satname = 'S2'
            
            # Sentinel2 collection
            count_loop = 0
            while count_loop < 1:
                try:
                    input_col = ee.ImageCollection('COPERNICUS/S2')
                    # filter by location and dates
                    flt_col = input_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(dates[0],dates[1])
                    # get all images in the filtered collection
                    im_all = flt_col.getInfo().get('features')
                    count_loop = 1
                except:
                    count_loop = 0
                    
            n_img = len(im_all)
            
            
            if n_img == 0:
                break
                    
            # remove duplicates in the collection (there are many in S2 collection)
            
            timestamps = [datetime.fromtimestamp(_['properties']['system:time_start']/1000,
                                                 tz=pytz.utc) for _ in im_all]
                    # utm zone projection
            utm_zones = np.array([int(_['bands'][0]['crs'][5:]) for _ in im_all])
            utm_zone_selected =  np.max(np.unique(utm_zones))
            # find the images that were acquired at the same time but have different utm zones
            idx_all = np.arange(0,len(im_all),1)
            idx_covered = np.ones(len(im_all)).astype(bool)
            idx_delete = []
            i = 0
            while 1:
                same_time = np.abs([(timestamps[i]-_).total_seconds() for _ in timestamps]) < 60*60*24
                idx_same_time = np.where(same_time)[0]
                same_utm = utm_zones == utm_zone_selected
                idx_temp = np.where([same_time[j] == True and same_utm[j] == False for j in idx_all])[0]
                idx_keep = idx_same_time[[_ not in idx_temp for _ in idx_same_time ]]
                # if more than 2 images with same date and same utm, drop the last ones
                if len(idx_keep) > 2: 
                   idx_temp = np.append(idx_temp,idx_keep[-(len(idx_keep)-2):])
                for j in idx_temp:
                    idx_delete.append(j)
                idx_covered[idx_same_time] = False
                if np.any(idx_covered):
                    i = np.where(idx_covered)[0][0]
                else:
                    break
                    
            # update the collection by deleting all those images that have same timestamp and different
            # utm projection
            im_all_updated = [x for k,x in enumerate(im_all) if k not in idx_delete]
            
            # remove very cloudy images (>95% cloud)
            cloud_cover = [_['properties']['CLOUDY_PIXEL_PERCENTAGE'] for _ in im_all] #im_all_updated
            #if np.any([_ > 95 for _ in cloud_cover]):
                #idx_delete = np.where([_ > 95 for _ in cloud_cover])[0]
                #im_col = [x for k,x in enumerate(im_all_updated) if k not in idx_delete]
            #else:
                #im_col = im_all_updated
                
            ID = [_['properties']['PRODUCT_ID'] for _ in im_all]
                
                
            n_img = len(im_all) #im_all_updated
            # print how many images there are
            #print('%s: %d images'%(satname,n_img)) 
            
            # loop though images
            
            timestamps = []
            #all_names = []
            
            for i in range(n_img):
                count_loop = 0
                while count_loop < 1:
                    try:
                        # find each image in ee database
                        im = ee.Image(im_all[i]['id']) #im_col
                        count_loop = 1
                    except:
                        count_loop = 0
                        
                # read metadata
                im_dic = im_all[i] #im_col
                # get time of acquisition (UNIX time)
                t = im_dic['properties']['system:time_start']
                # convert to datetime
                im_timestamp = datetime.fromtimestamp(t/1000, tz=pytz.utc)
                timestamps.append(im_timestamp)
                #im_date = im_timestamp.strftime('%Y-%m-%d-%H-%M-%S')
                sat_list = ['S2']*len(timestamps)

            
            kk = pd.DataFrame({'Date':timestamps, 'Cloud_cover':cloud_cover, 'Sat':sat_list}) #"ID"
            Satellite_old_passes = Satellite_old_passes.append(kk)
            del kk
            
    
    print('Total num of images: ', len(Satellite_old_passes))
            
    return Satellite_old_passes


def S2L2A_retrieve_inf(inputs):
    
        # Initialise the connection with GEE server
    
    ee.Initialize()
    
    # Read inputs dictionary
    
    polygon = inputs['polygon']
    dates = inputs['dates']
    #sat_list = inputs['sat_list']   

    

        
    Satellite_S2L2A = pd.DataFrame({'Date':[], 'Cloud_cover':[], 'Sat':[], 'ID':[]})
    
    print('Searching for the SentinelL2A images')
    
    
    satname = 'S2'
    
    count_loop = 0
    while count_loop < 1:
        try:
            input_col = ee.ImageCollection('COPERNICUS/S2_SR')
            # filter by location and dates
            flt_col = input_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(dates[0],dates[1])
            # get all images in the filtered collection
            im_all = flt_col.getInfo().get('features')
            count_loop = 1
        except:
            count_loop = 0
            
    
    #timestamps = [datetime.fromtimestamp(_['properties']['system:time_start']/1000,
                                         #tz=pytz.utc) for _ in im_all]
            # utm zone projection

            

    
    cloud_cover = [_['properties']['CLOUDY_PIXEL_PERCENTAGE'] for _ in im_all] #im_all_updated
    ID = [_['properties']['PRODUCT_ID'] for _ in im_all]
    #timestamps = [datetime.fromtimestamp(_['properties']['GENERATION_TIME']/1000,
                                         #tz=pytz.utc) for _ in im_all]
 
        
        
    n_img = len(im_all) #im_col #im_all_updated
    # print how many images there are
    print('%s: %d images'%(satname,n_img)) 
    
    # loop though images
    
    timestamps = []
    all_names = []
    
    for i in range(n_img):
        count_loop = 0
        while count_loop < 1:
            try:
                # find each image in ee database
                im = ee.Image(im_all[i]['id']) #im_col
                count_loop = 1
            except:
                count_loop = 0
                
        # read metadata
        im_dic = im_all[i] #im_col
        #get time of acquisition (UNIX time)
        t = im_dic['properties']['system:time_start']
        # convert to datetime
        im_timestamp = datetime.fromtimestamp(t/1000, tz=pytz.utc)
        timestamps.append(im_timestamp)
        #im_date = im_timestamp.strftime('%Y-%m-%d-%H-%M-%S')
        sat_list = ['S2']*len(timestamps)

    
    kk = pd.DataFrame({'Date':timestamps, 'Cloud_cover':cloud_cover, 'Sat':sat_list, 'ID':ID})
    Satellite_S2L2A = Satellite_S2L2A.append(kk)
    del kk
    
    return Satellite_S2L2A, im_dic
 
        
def Retrive_Hyperion_im(inputs):
    
    """
    Hyperion images are available for the period 2001-05-01T00:00:00 - 2017-03-13T00:00:00 
    ## ESTO LO TENGO QUE MIRAR EN ALGÚN MOMENTO
    """
    
    ee.Initialize()
    
    # read inputs dictionnary
    sitename = inputs['sitename']
    polygon = inputs['polygon']
    dates = inputs['dates']
    filepath_data = inputs['filepath']
    
    # initialize metadata dictionnary (stores information about each image)       
    metadata = dict([])
    
    # create a new directory for this site
    if not os.path.exists(os.path.join(filepath_data,sitename)):
        os.makedirs(os.path.join(filepath_data,sitename))
        
    print('Downloading images:')
    
    #=============================================================================================#
    # download HYPERION images
    #=============================================================================================#
    
    # Landsat 5 collection
    count_loop = 0
    while count_loop < 1:
        try:
            input_col = ee.ImageCollection('EO1/HYPERION')
            # filter by location and dates
            flt_col = input_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(dates[0],dates[1])
            # get all images in the filtered collection
            im_all = flt_col.getInfo().get('features')
            count_loop = 1
        except: 
            count_loop = 0  
            
    print(len(im_all))
    

def Retrive_S1_im(inputs):
    
    """
    
    Available since 3 Aprile 2014
    
    Inputs is a dictionary that have to contain the following metadata information in order to filter in the database
    
    polygon
    dates
    trasmitterReceiverPolarisation.- 'VV'; 'HH'; ['VV', 'VH']; ['HH', 'HV']
    InstrumentMode # Josep has not specified it
    orbitProperties_pass.- Ascending; Descending
    resolution_meters = 10; 25; 40
    resolution .- 'M' (medium); 'H' (high)
    
    """
    
    ee.Initialize()
    
    # Inicializamos el dataframe
    
    S1_data = gpd.GeoDataFrame({'Date':[] ,'res': [], 'res_m': [],'orbit':[],
                            'polarization': [], 'crs':[]})
    
    # Read inputs polygon 
    
    polygon = inputs['polygon']
    dates = inputs['dates']
    polarization = inputs['trasmitterReceiverPolarisation']
    orbit = inputs['orbitProperties_pass']
    
    ## Filtering by dates and polygon
    
    count_loop = 0
    while count_loop < 1:
        try:
            input_col = ee.ImageCollection('COPERNICUS/S1_GRD')
            # filter by location and dates
            flt_col = input_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(dates[0],dates[1])
            
            # Filter by metadata properties.
            vh = flt_col \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization[0][0])) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization[0][1])) 
            
            hv = flt_col \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization[1][0])) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', polarization[1][1]))
            
            # get all images in the filtered collection
            
            im_all_1= vh.getInfo().get('features')
            im_all_2 = hv.getInfo().get('features')
            
            count_loop = 1
            
        except:
            
            count_loop = 0

            
    if len(im_all_2)>0 and len(im_all_1)>0:
        print('caso1')
        im_all = im_all_1 + im_all_2
        
        
    elif len(im_all_2) == 0:
        print('caso2')
        im_all = im_all_1
        
    elif len(im_all_1) == 0:
        print('caso3')
        im_all = im_all_2
    
        
            
        # Guardamos la informacion que nos interesa en un pandas DataFrame 
        
    resolution = []
    orbit = []
    resolution_m = []
    polare = []
    crs = []
    times = []
    

    for passag in im_all:

        resolution.append(passag['properties']['resolution'])
        orbit.append(passag['properties']['orbitProperties_pass'])
        resolution_m.append(passag['properties']['resolution_meters'])
        polar = passag['properties']['transmitterReceiverPolarisation']
        polar = "-".join(polar)
        polare.append(polar)
        crs.append(passag['bands'][0]['crs'])
        
        t = passag['properties']['system:time_start']
        im_timestamp = datetime.fromtimestamp(t/1000, tz=pytz.utc)
        
        times.append(im_timestamp)
        
    dd = pd.DataFrame({'Date':times, 'res':resolution, 'res_m':resolution_m, 
                       'orbit':orbit, 'polarization': polare, 'crs':crs})
    
    S1_data = S1_data.append(dd)
 
    return S1_data
            
    
    
    
    
       