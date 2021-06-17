#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 09:51:56 2020

@author: paolasouto
"""

import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd
import os
import glob
from dateutil.relativedelta import relativedelta
import geopandas as gpd
import pytz
import gdal

import matplotlib.pyplot as plt
#import skimage.exposure as exposure
#import skimage.morphology as morphology
from osgeo import gdal
from datetime import datetime
import tqdm
import geopandas as gdp

import sys
#sys.path.append('/Users/paolasouto/Desktop/DOTTORATO/python/CoastSat/') 
sys.path.append('/Users/paolasouto/python_toolkits/CoastSat-master')
sys.path.append('/Users/paolasouto/Desktop/DOTTORATO/python/functions/')

from My_functions import Satellite_Passes
from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects, SDS_classify

import fiona

fiona.drvsupport.supported_drivers['kml'] = 'rw' # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw' # enable KML support which is disabled by default



def make_video(dir_ECFAS, framesSec):
    
    
    folders = [f for f in os.listdir(dir_ECFAS) if not ( (f == 'Make_Videos.ipynb') or (f == '.ipynb_checkpoints') or (f.endswith('.csv')) or (f=='.DS_Store'))]
    
    for folder in folders:
        
        AOI_case = folder
        
        dir_png = os.path.join(dir_ECFAS, AOI_case, 'png_im/')
        dir_path = dir_png
        ext = '.jpg'
        output = os.path.join(dir_ECFAS,AOI_case + '.mp4')
        shape = 1195, 640
        fps = framesSec
        
        images_names = [f for f in os.listdir(dir_path) if f.endswith(ext)]
        
        if len(images_names)==0:
            continue
        
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        video = cv2.VideoWriter(output, fourcc, fps, shape, isColor=True)
        
        for image_name in np.sort(images_names):
            image_path = os.path.join(dir_path, image_name)
            image = cv2.imread(image_path)
            resized = cv2.resize(image, shape)
            video.write(resized)
            
        video.release()
        
def generate_png(satname,dir_ECFAS, dir_png, AOI_case, info, date_flood):
    o = 0
    for sat in satname:
        if (sat=='L7') or (sat=='L8'):
            dir_sat = os.path.join(dir_ECFAS, AOI_case,sat, 'ms/')
            
            images_names = [f for f in os.listdir(dir_sat) if f[-4:]=='.tif']
            
        
        
            if len(images_names) == 0:
                continue
    
    
            for image_name in images_names:
                
                o += 1
                
    
                dir_ms = os.path.join(dir_sat, image_name)
                data = gdal.Open(dir_ms, gdal.GA_ReadOnly)
                georef = np.array(data.GetGeoTransform())
                bands = [data.GetRasterBand(k+1).ReadAsArray() for k in range(data.RasterCount)]
                im_ms = np.stack(bands,2)
    
                img_RGB = im_ms[:,:,[2,1,0]]
    
                if datetime.strptime(image_name[0:19], '%Y-%m-%d-%H-%M-%S') < date_flood:
                    cc = 'yellow'
                elif datetime.strptime(image_name[0:19], '%Y-%m-%d-%H-%M-%S') > date_flood:
                    cc ='red'
                    
                print(image_name)
                Cloud = str(np.round(np.unique(info[(info['Date2'] == datetime.strptime(image_name[0:19], '%Y-%m-%d-%H-%M-%S').date()) & (info['Sat']==sat) ].Cloud_cover)[0],2))
                #Cloud = str(np.round(np.unique(info[(info['Date2'] == date_flood.date()) & (info['Sat']==sat) ].Cloud_cover)[0],2))

                fig = plt.figure()
                fig.set_size_inches([18,9])
                fig.set_tight_layout(True)
                ax1 = fig.add_subplot(111)
                ax1.axis('off')
                ax1.text(0.05,0.05,image_name[0:19] + '; sat = ' + sat + '; CC = ' + Cloud, fontsize=30, color =cc,transform=ax1.transAxes)
                ax1.imshow(img_RGB)
    
                fig.savefig(os.path.join(dir_png,image_name[0:19] + '_' + sat + '.jpg'),bbox_inches='tight', dpi=150) #
                plt.close(fig)
                
        if (sat=='L5'):
            dir_sat = os.path.join(dir_ECFAS, AOI_case, sat, '30m/')
            
            

           # if not os.path.exists(dir_sat):
                #continue

            images_names = [f for f in os.listdir(dir_sat) if f[-4:]=='.tif']
            
        
        
            if len(images_names) == 0:
                continue
    
    
            for image_name in images_names:
                
                o += 1
                
    
                dir_ms = os.path.join(dir_sat, image_name)
                data = gdal.Open(dir_ms, gdal.GA_ReadOnly)
                georef = np.array(data.GetGeoTransform())
                bands = [data.GetRasterBand(k+1).ReadAsArray() for k in range(data.RasterCount)]
                im_ms = np.stack(bands,2)
    
                img_RGB = im_ms[:,:,[2,1,0]]
    
                if datetime.strptime(image_name[0:19], '%Y-%m-%d-%H-%M-%S') < date_flood:
                    cc = 'yellow'
                elif datetime.strptime(image_name[0:19], '%Y-%m-%d-%H-%M-%S') > date_flood:
                    cc ='red'
                    
    
                Cloud = str(np.round(np.unique(info[(info['Date2'] == datetime.strptime(image_name[0:19], '%Y-%m-%d-%H-%M-%S').date()) & (info['Sat']==sat) ].Cloud_cover)[0],2))
                #Cloud = str(np.round(np.unique(info[(info['Date2'] == date_flood.date()) & (info['Sat']==sat) ].Cloud_cover)[0],2))

                fig = plt.figure()
                fig.set_size_inches([18,9])
                fig.set_tight_layout(True)
                ax1 = fig.add_subplot(111)
                ax1.axis('off')
                ax1.text(0.05,0.05,image_name[0:19] + '; sat = ' + sat + '; CC = ' + Cloud, fontsize=30, color =cc,transform=ax1.transAxes)
                ax1.imshow(img_RGB)
    
                fig.savefig(os.path.join(dir_png,image_name[0:19] + '_' + sat + '.jpg'),bbox_inches='tight', dpi=150) #
                plt.close(fig)

        if (sat=='S2'):
            o += 1
            
            dir_sat = os.path.join(dir_ECFAS, AOI_case,sat,'10m/')
            
            
            if not os.path.exists(dir_sat):
                continue
            
            images_names = [f for f in os.listdir(dir_sat) if f[-4:]=='.tif']
            
            
            if len(images_names) == 0:
                continue

            for image_name in images_names:
                
                o += 1
                

                dir_ms = os.path.join(dir_sat, image_name)

                data = gdal.Open(dir_ms, gdal.GA_ReadOnly)
                georef = np.array(data.GetGeoTransform())
                bands = [data.GetRasterBand(k+1).ReadAsArray() for k in range(data.RasterCount)]
                im_ms = np.stack(bands,2)
                im_ms = im_ms/10000

                #if sum(sum(sum(im_ms))) < 1:
                    #im_ms = []
                    #georef = []
                    # skip the image by giving it a full cloud_mask
                    #cloud_mask = np.ones((im_ms.shape[0],im_ms.shape[1])).astype('bool')
                    #continue

                img_RGB = im_ms[:,:,[2,1,0]]
                

                if datetime.strptime(image_name[0:19], '%Y-%m-%d-%H-%M-%S') < date_flood:
                    cc = 'yellow'
                elif datetime.strptime(image_name[0:19], '%Y-%m-%d-%H-%M-%S') > date_flood:
                    cc ='red'
                

                Cloud = str(np.round(np.unique(info[(info['Date2'] == datetime.strptime(image_name[0:19], '%Y-%m-%d-%H-%M-%S').date()) & (info['Sat']==sat) ].Cloud_cover)[0],2))
                #Cloud = str(np.round(np.unique(info[(info['Date2'] == date_flood.date()) & (info['Sat']==sat) ].Cloud_cover)[0],2))
                
                fig = plt.figure()
                fig.set_size_inches([18,9])
                fig.set_tight_layout(True)
                ax1 = fig.add_subplot(111)
                ax1.axis('off')
                ax1.text( 0.05, 0.05,image_name[0:19] + '; sat = ' + sat + '; CC = ' + Cloud, fontsize=30, color =cc, transform=ax1.transAxes)
                ax1.imshow(img_RGB)

                fig.savefig(os.path.join(dir_png,image_name[0:19] + '_' + sat + '.jpg'),bbox_inches='tight', dpi=150) #, 
                plt.close(fig)


    
    
def from_ms_to_jpg(allowed_cc, num_months, dir_ECFAS, info_cases):
    
   
    
    satname = ['L5', 'L7', 'L8', 'S2']
    
    # cloud cover allowed
    
    cc = allowed_cc
    
    # Number of months to look after and before the event
    
    n_m = num_months
    
    # Search the subfolders
    
    folders = [f for f in os.listdir(dir_ECFAS) if not ( (f.endswith('.csv')) or (f == '.DS_Store')) ]
    
    for folder in folders:
        AOI_case = folder
        dir_png = os.path.join(dir_ECFAS, AOI_case, 'png_im')
        
        if not os.path.exists(dir_png):
            os.makedirs(dir_png)
            
    for folder in folders:
        sitename = folder
        AOI_case = folder
        dir_png = os.path.join(dir_ECFAS, AOI_case, 'png_im')
        print(dir_png)
        print(folder)
        
        #for row in info_cases.iterrows():
            #if folder.split('_')[2]  in row[1]['Polygon Name']:
                #print(folder.split('_')[2] , row[1]['Polygon Name'], 'He pasado')
                #if folder.split('_')[-1] == row[1]['STORM NAME']:
                    #date_flood = datetime.strptime(row[1]['DATES_STORM'], '%d/%m/%y')
                    #print(date_flood)
        for row in info_cases.iterrows():
            if (folder.split('_')[2]  in row[1]['Polygon Name']) and (folder.split('_')[-1] == row[1]['STORM NAME']):
                date_flood = datetime.strptime(row[1]['DATES_STORM'], '%d/%m/%y')
                print(folder, date_flood)
            else:
                continue
            
        
        # TimeZone to the flood date
        
        flood_utc = pytz.timezone('UTC').localize(date_flood)
        
        # Period at which to look for images
        
        ini_T = (date_flood - relativedelta(months=n_m)).strftime('%Y-%m-%d')
        fini_T = (date_flood + relativedelta(months=n_m)).strftime('%Y-%m-%d')
        
        dates =  ini_T,fini_T 
        print(dates)
        
        # flood date as str
        
        date_flood_str = date_flood.strftime('%Y-%m-%d')
        
        # Load the site polygon
        
        #kml_file = glob.glob(os.path.join(dir_ECFAS, AOI_case, AOI_case) + '*.kml')
        kml_file = [file for file in os.listdir(os.path.join(dir_ECFAS, AOI_case)) if file[-4:] == '.kml']

        
        ff = gdp.read_file(os.path.join(dir_ECFAS, AOI_case,kml_file[0]))
        c = ff['geometry'][0]
        shell_coords = np.array(c.exterior)
        polygon = []
        
        for x,y in zip(shell_coords[:,:1], shell_coords[:,1:2]):
            polygon.append([list(x)[0], list(y)[0]])
            
        filepath = os.path.join(dir_ECFAS, AOI_case)
        
        inputs = {'polygon': polygon, 'dates': dates, 'sat_list': satname, 'sitename': sitename, 'filepath':filepath}
        
        info = Satellite_Passes.Retrive_images(inputs)
        info = info.reset_index(drop=True)
        info['Date2'] = info['Date'].dt.date
        
        print(info)

        
        generate_png(satname, dir_ECFAS, dir_png, AOI_case, info, date_flood)
            
    