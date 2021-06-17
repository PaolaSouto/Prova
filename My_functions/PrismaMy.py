#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 14:54:39 2020

READ PRISMA ON PYTHON 

Paola Souto Ceccon (Like in ENVI I don't delete the duplicated ')

    Inputs:
        dir0 = string with the path where the image is stored
        SCENE = ej : Grosseto, Arborea etc
        image_name = string with the image name
        error_matrix = boolean True/False. True for not taking account of the satured pixels
        save = boolean True/False if the user wants to save the image
        zone = int of the zone where the image is located


@author: paolasouto
"""

import sys


import numpy as np
import os
import h5py
import tqdm
import gdal
from osgeo import osr
import tqdm
from pyproj import Proj
import matplotlib.pyplot as plt
import pandas as pd
import ogr




if 'GDAL_DATA' not in os.environ:
    os.environ['GDAL_DATA'] = r'/path/to/gdal_data'
    
    
def GetImageGeoAccuracy(dir0, SCENE, image_name):
    
    dir_image = os.path.join(dir0, SCENE, image_name)
    
    # Load the image
    
    f = h5py.File(dir_image, 'r')
    
    # Get GeoAccuracy
    
    for ff in f.attrs:
        if ff == 'Geolocation_accuracy':
            geoAcc = f.attrs[ff]
            return geoAcc
        
            
    
def ObtainCW(dir0, SCENE, image_name, error_matrix):
    
    dir_image = os.path.join(dir0, SCENE, image_name )
    
    # Load the image
    
    f = h5py.File(dir_image, 'r')
    
    # Load the CW images
    
    CW = np.concatenate([f.attrs['List_Cw_Vnir'][::-1], f.attrs['List_Cw_Swir'][::-1]])
    
    
    
    # Load the Flags
    
    Flag = np.concatenate([f.attrs['CNM_VNIR_SELECT'][::-1], f.attrs['CNM_SWIR_SELECT'][::-1]])
    
    CW2 = CW[Flag.astype(bool)]
    
    CW3 = CW2[np.argsort(CW2)]
    
    return CW3
    
    

def ReadPrismaImage(dir0, SCENE, image_name, error_matrix, save, save_RGB):
    

    
    dir_image = os.path.join(dir0, SCENE, image_name)
    

    
    # Load the image
    
    f = h5py.File(dir_image, 'r')
    
    
    
    # Generamos Geo with the attributes given in the image
    
    geo = {'proj_code':f.attrs['Projection_Id'], 'proj_name':f.attrs['Projection_Name'],
                          'proj_epsg':f.attrs['Epsg_Code'], 'xmin':np.min([f.attrs['Product_ULcorner_easting'], f.attrs['Product_LLcorner_easting']]),
      'xmax':np.max([f.attrs['Product_LRcorner_easting'], f.attrs['Product_URcorner_easting']]),
      'ymin':np.min([f.attrs['Product_LLcorner_northing'], f.attrs['Product_LRcorner_northing']]),
      'ymax':np.max([f.attrs['Product_ULcorner_northing'], f.attrs['Product_URcorner_northing']])}
    
    
    
    
    # Load the CW images
    
    CW = np.concatenate([f.attrs['List_Cw_Vnir'][::-1], f.attrs['List_Cw_Swir'][::-1]])
    
    # Load the band width
    
    BandWidth = np.concatenate([f.attrs['List_Fwhm_Vnir'][::-1], f.attrs['List_Fwhm_Swir'][::-1]])
    

    
    # Load the Flags
    
    Flag = np.concatenate([f.attrs['CNM_VNIR_SELECT'][::-1], f.attrs['CNM_SWIR_SELECT'][::-1]])
    
    # Load the Geolocation Information
    
    Lat = np.array(f['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Geolocation Fields']['Latitude'])
    Lon = np.array(f['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Geolocation Fields']['Longitude'])
    

    
    print('The image has been loaded')
    
    SWIR_bands =np.array(f['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Data Fields']['SWIR_Cube'])
    VNIR_bands = np.array(f['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Data Fields']['VNIR_Cube'])
    SWIR_bands_C = np.swapaxes(SWIR_bands, 1, 2)
    VNIR_bands_C = np.swapaxes(VNIR_bands, 1, 2)
    VNIR_bands_CC = VNIR_bands_C[:, :, ::-1]
    SWIR_bands_CC = SWIR_bands_C[:, :, ::-1]
    
    
    # Load the parameters for scale the DN
    
    L2ScaleSwirMax = f.attrs['L2ScaleSwirMax']
    L2ScaleSwirMin = f.attrs['L2ScaleSwirMin']
    L2ScaleVnirMax = f.attrs['L2ScaleVnirMax']
    L2ScaleVnirMin = f.attrs['L2ScaleVnirMin']
    
    
    # Aplly the correction
    
    print('Dn to reflectance in the SWIR cube...')
    
    SWIR_bands_R = np.float32(SWIR_bands_CC.copy())
    for n in tqdm.tqdm(range(SWIR_bands_CC.shape[2])):
        SWIR_bands_R[:,:,n] = L2ScaleSwirMin + SWIR_bands_CC[:,:,n]*\
            (L2ScaleSwirMax-L2ScaleSwirMin)/65535
            
            
    print('Dn to reflectance in the VNIR cube...')
    
    
    VNIR_bands_R = np.float32(VNIR_bands_CC.copy())
    for n in tqdm.tqdm(range(VNIR_bands_CC.shape[2])):
        VNIR_bands_R[:,:,n] = L2ScaleVnirMin + VNIR_bands_CC[:,:,n]*\
            (L2ScaleVnirMax - L2ScaleVnirMin)/65535
            
    print('Generating the image ....')
    
    # TENGO QUE COMPROBAR QUE SIEMPRE SON LAS PRIMERAS 3 Y LAS ULTIMAS 2 BANDAS LAS QUE NO SON VALIDAS
    
    img = np.concatenate([VNIR_bands_R,SWIR_bands_R], axis=2)
    
    
    img2 = img[:,:,Flag.astype(bool)]
    CW2 = CW[Flag.astype(bool)]
    BandWidth2 = BandWidth[Flag.astype(bool)] 
    

    # Las tenemos que ordenar
    
    print('Ordering the bands')
    
    CW3 = CW2[np.argsort(CW2)]
    img3 = img2[:,:, np.argsort(CW2)]
    BandWidth3 = BandWidth2[np.argsort(CW2)]
    
            
    if error_matrix:
        
        print('Appliying the error matrix...')
        
        ERR_VNIR = np.array(f['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Data Fields']['VNIR_PIXEL_L2_ERR_MATRIX'])  
        ERR_VNIR_C = np.swapaxes(ERR_VNIR, 1, 2)
        ERR_VNIR_CC = ERR_VNIR_C[:, :, ::-1]
        
        
        ERR_SWIR = np.array(f['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Data Fields']['SWIR_PIXEL_L2_ERR_MATRIX'])  
        ERR_SWIR_C = np.swapaxes(ERR_SWIR, 1, 2)
        ERR_SWIR_CC = ERR_SWIR_C[:, :, ::-1]
        
        ERR = np.concatenate([ERR_VNIR_CC,ERR_SWIR_CC], axis=2)
        ERR_C = ERR[:,:,Flag.astype(bool)]

        
        for n in tqdm.tqdm(range(img2.shape[2])):
            idx = np.where(ERR_C[:,:,n] != 0)
            img2[idx[0], idx[1],n] = -999 # Tb lo podriamos poner a 0 (tengo que explorar)
            #print('He cambiado a nan')
            
         
    res = 30
    
    GeoT, driver, Proje = GetGeoTransform(geo, img2, res)  

    
    if save:
    
        print('Saving the image ...')
        
        dir_save = os.path.join(dir0,SCENE, 'Processed')
        
        if not os.path.exists(dir_save):
            os.makedirs(dir_save, exist_ok=True)
            
        new_name = image_name[:-4] + 'processed.tiff'
        
        output = os.path.join(dir_save, new_name)

        CreateTiff(output, img2, driver, GeoT, Proje)
        
        print('The image ', os.path.join(dir_save, new_name), 'has been created')
        
    if save_RGB:
        
        print('Saving RGB image ...')
        
        dir_save_RGB = os.path.join(dir0, SCENE, 'RGB')
        
        if not os.path.exists(dir_save_RGB):
            os.makedirs(dir_save_RGB, exist_ok = True)
        
        new_name_RGB = image_name[:-4] + 'rgb.tiff'
        
        output_RGB = os.path.join(dir_save_RGB, new_name_RGB)
        
        CreateRGBTiff(output_RGB, img2, driver, GeoT, Proje)
        
        print('The image ', os.path.join(dir_save_RGB, output_RGB), 'has been created')
        
        
        
        #pp = Proj(proj='utm',zone= int(geo['proj_code']),ellps='WGS84', preserve_units=False)
        #LLO, LLA = pp(Lon,Lat)
        
        #Longitude2 = LLO
        #Latitude2 = LLA
        
        #GeoT, driver, Proje = GetGeoTransform(Longitude2, Latitude2, img2)

            
    
    return img3, CW3, BandWidth3


def ReadPanchromatic(dir0, SCENE, image_name, error_matrix, save):
    
    """
    This function reads the panchromatic band from the PRISMA .he5 images
    
    # INPUTS:
        
    dir0 = string with SCENES folder
    SCENE = string indicated the name assigned to the Area of Interes
    image_name = string with the filename of the original PRISMA he5 image
    error_matrix = Boolean indicating if the error_matrix correction has to be applied or not
    save = Boolean indicating if the Panchromatic band has to be saved as GeoTiff 
    
    # OUTPUTS:
        
    pan = Numpy array with the panchromatic band
    
    The saved GeoTiff image in the generated folder "Panchromatic" inside path0
        
    
    """
    
    ## This function reads the panchromatic band from the PRISMA .he5 images
    
    
    
    dir_image = os.path.join(dir0, SCENE, image_name)
    
    # Load the image
    
    f = h5py.File(dir_image, 'r')
    
    # Load the panchromatic band
    
    panchro = np.array(f['HDFEOS']['SWATHS']['PRS_L2D_PCO']['Data Fields']['Cube'])
    
    # Now we have to apply the transformation from unit16 DN to reflectance values
    
    ScalePanMin = f.attrs['L2ScalePanMin']
    ScalePanMax = f.attrs['L2ScalePanMax']
    
    # Transform from DN to reflectance
    
    print('Transforming from DN to reflectance ...')
    
    panchro2 = ScalePanMin + panchro*(ScalePanMax-ScalePanMin)/65535
    
    # Apply the Error matrix
    
    if error_matrix :
    
        print('Applying the error matrix ...')
        
        ERR_Pan = np.array(f['HDFEOS']['SWATHS']['PRS_L2D_PCO']['Data Fields']['PIXEL_L2_ERR_MATRIX'])
        
        idxs = np.where( ERR_Pan != 0)
    
        panchro2[idxs[0], idxs[1]] = 0
    
    # Generamos Geo with the attributes given in the image
    
    geo = {'proj_code':f.attrs['Projection_Id'], 'proj_name':f.attrs['Projection_Name'],
                          'proj_epsg':f.attrs['Epsg_Code'], 'xmin':np.min([f.attrs['Product_ULcorner_easting'], f.attrs['Product_LLcorner_easting']]),
      'xmax':np.max([f.attrs['Product_LRcorner_easting'], f.attrs['Product_URcorner_easting']]),
      'ymin':np.min([f.attrs['Product_LLcorner_northing'], f.attrs['Product_LRcorner_northing']]),
      'ymax':np.max([f.attrs['Product_ULcorner_northing'], f.attrs['Product_URcorner_northing']])}

    res = 5
    
    #panchromatic = True
    
    GeoT, driver, Proje = GetGeoTransform(geo, panchro2, res)  
    

    
    if save:
    
        print('Saving the image ...')
        
        dir_save = os.path.join(dir0,SCENE, 'Panchromatic')
        
        if not os.path.exists(dir_save):
            os.makedirs(dir_save, exist_ok=True)
            
        new_name = image_name[:-4] + 'pan.tiff'
        print(new_name)
        
        output = os.path.join(dir_save, new_name)

        CreateTiffPan(output, panchro2, driver, GeoT, Proje)
        
        print('The image ', os.path.join(dir_save, new_name), 'has been created')

      
    
    return panchro2


## THIS FUNCTION CAN BE DELETED

def GetGeo(dir0, SCENE, image_name):
    
    """
    This function returns only the geo information of the image
    
    """
    
    dir_image = os.path.join(dir0, SCENE, image_name)
    f = h5py.File(dir_image, 'r')
    
    geo = {'proj_code':f.attrs['Projection_Id'], 'proj_name':f.attrs['Projection_Name'],
           'proj_epsg':f.attrs['Epsg_Code'], 'xmin':np.min([f.attrs['Product_ULcorner_easting'], f.attrs['Product_LLcorner_easting']]),
           'xmax':np.max([f.attrs['Product_LRcorner_easting'], f.attrs['Product_URcorner_easting']]),
           'ymin':np.min([f.attrs['Product_LLcorner_northing'], f.attrs['Product_LRcorner_northing']]),
           'ymax':np.max([f.attrs['Product_ULcorner_northing'], f.attrs['Product_URcorner_northing']])}
    
    return geo
    


def VNIR_Cube_Image(dir0, SCENE, image_name, error_matrix, save, zona):
    
    
    dir_image = os.path.join(dir0, SCENE, image_name)
    
    # Load the image
    
    f = h5py.File(dir_image, 'r')
    
    # Load the images
    
    CW = np.float64(f.attrs['List_Cw_Vnir'][::-1])
    
    
    # Load the Flags
    
    Flag = f.attrs['CNM_VNIR_SELECT'][::-1]
    
    # Correct number of bands
    
    CWave = CW[Flag.astype(bool)]
    
    # Load the Geolocation Information
    
    Lat = np.array(f['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Geolocation Fields']['Latitude'])
    Lon = np.array(f['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Geolocation Fields']['Longitude'])
    
    
    print('The image has been loaded')
    
    VNIR_bands = np.float64(np.array(f['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Data Fields']['VNIR_Cube']))
    VNIR_bands_C = np.swapaxes(VNIR_bands, 1, 2)
    VNIR_bands_CC = VNIR_bands_C[:, :, ::-1]
    
    
    # Load the parameters for scale the DN

    L2ScaleVnirMax = f.attrs['L2ScaleVnirMax']
    L2ScaleVnirMin = f.attrs['L2ScaleVnirMin']
    
    
    print('Dn to reflectance in the VNIR cube...')
    
    
    VNIR_bands_R = np.float64(VNIR_bands_CC.copy())
    for n in tqdm.tqdm(range(VNIR_bands_CC.shape[2])):
        VNIR_bands_R[:,:,n] = L2ScaleVnirMin + VNIR_bands_CC[:,:,n]*\
            (L2ScaleVnirMax - L2ScaleVnirMin)/65535
            
    img2 = VNIR_bands_R[:,:,Flag.astype(bool)]
            
    if error_matrix:
    
        print('Appliying the error matrix...')
        
        ERR_VNIR = np.array(f['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Data Fields']['VNIR_PIXEL_L2_ERR_MATRIX'])  
        ERR_VNIR_C = np.swapaxes(ERR_VNIR, 1, 2)
        ERR_VNIR_CC = ERR_VNIR_C[:, :, ::-1]
        
        for n in tqdm.tqdm(range(img2.shape[2])):
            idx = np.where(ERR_VNIR_CC[:,:,n] != 0)
            img2[idx[0], idx[1],n] = 0 # Tb lo podriamos poner a 0 (tengo que explorar)
            
            
    if save:
    
        print('Saving the VNIR image ...')
        
        dir_save = os.path.join(dir0,SCENE, 'Processed_VNIR')
        
        if not os.path.exists(dir_save):
            os.makedir(dir_save, exist_ok=True)
            
        new_name = image_name[:-4] + 'processed_VNIR.tiff'
        output = os.path.join(dir_save, new_name)
        
        
        pp = Proj(proj='utm',zone=zona,ellps='WGS84', preserve_units=False)
        LLO, LLA = pp(Lon,Lat)
        
        Longitude2 = LLO
        Latitude2 = LLA
        
        GeoT, driver, Proje = GetGeoTransform(Longitude2, Latitude2, img2)
        
        panchromatic = True
        
        CreateTiffPan(output, img2, driver, GeoT, Proje)
        
        
        print('The image ', os.path.join(dir_save, new_name), 'has been created')
            
    
    return img2, CWave



def SWIR_Cube_Image(dir0, SCENE, image_name, error_matrix, save, zona):
    
    
    dir_image = os.path.join(dir0, SCENE, image_name)
    
    # Load the image
    
    f = h5py.File(dir_image, 'r')
    
    # Load the images
    
    CW = np.float64(f.attrs['List_Cw_Swir'][::-1])
    
    
    # Load the Flags
    
    Flag = f.attrs['CNM_SWIR_SELECT'][::-1]
    
    # Correct number of bands
    
    CWave = CW[Flag.astype(bool)]
    
    # Load the Geolocation Information
    
    Lat = np.array(f['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Geolocation Fields']['Latitude'])
    Lon = np.array(f['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Geolocation Fields']['Longitude'])
    
    
    print('The image has been loaded')
    
    SWIR_bands = np.float64(np.array(f['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Data Fields']['SWIR_Cube']))
    SWIR_bands_C = np.swapaxes(SWIR_bands, 1, 2)
    SWIR_bands_CC = SWIR_bands_C[:, :, ::-1]
    
    
    # Load the parameters for scale the DN

    L2ScaleSwirMax = f.attrs['L2ScaleSwirMax']
    L2ScaleSwirMin = f.attrs['L2ScaleSwirMin']
    
    
    print('Dn to reflectance in the SWIR cube...')
    
    
    SWIR_bands_R = np.float64(SWIR_bands_CC.copy())
    for n in tqdm.tqdm(range(SWIR_bands_CC.shape[2])):
        SWIR_bands_R[:,:,n] = L2ScaleSwirMin + SWIR_bands_CC[:,:,n]*\
            (L2ScaleSwirMax - L2ScaleSwirMin)/65535
            
    img2 = SWIR_bands_R[:,:,Flag.astype(bool)]
            
    if error_matrix:
    
        print('Appliying the error matrix...')
        
        ERR_SWIR = np.array(f['HDFEOS']['SWATHS']['PRS_L2D_HCO']['Data Fields']['SWIR_PIXEL_L2_ERR_MATRIX'])  
        ERR_SWIR_C = np.swapaxes(ERR_SWIR, 1, 2)
        ERR_SWIR_CC = ERR_SWIR_C[:, :, ::-1]
        
        for n in tqdm.tqdm(range(img2.shape[2])):
            idx = np.where(ERR_SWIR_CC[:,:,n] != 0)
            img2[idx[0], idx[1],n] = 0 # Tb lo podriamos poner a 0 (tengo que explorar)
            
            
    if save:
    
        print('Saving the SWIR image ...')
        
        dir_save = os.path.join(dir0,SCENE, 'Processed_SWIR')
        
        if not os.path.exists(dir_save):
            os.makedirs(dir_save, exist_ok=True)
            
        new_name = image_name[:-4] + 'processed_SWIR.tiff'
        output = os.path.join(dir_save, new_name)
        
        
        pp = Proj(proj='utm',zone=zona,ellps='WGS84', preserve_units=False)
        LLO, LLA = pp(Lon,Lat)
        
        Longitude2 = LLO
        Latitude2 = LLA
        
        GeoT, driver, Proje = GetGeoTransform(Longitude2, Latitude2, img2)
        
        CreateTiff(output, img2, driver, GeoT, Proje)
        
        
        print('The image ', os.path.join(dir_save, new_name), 'has been created')
            
    
    return img2, CWave
        
    
def GetGeoTransform(geo, img2, res):
    """
    This could be changed after compare with PRISMAREAD
    """
    
    #lat_min = np.min(Latitude2)
    #lat_max = np.max(Latitude2)
    #lon_min = np.min(Longitude2)
    #lon_max = np.max(Longitude2)
    
    #xres = (lon_max - lon_min) / float(img2.shape[1]) # Risoluzione del pixel in x
    #yres = (lat_max - lat_min) / float(img2.shape[0]) # Risoluzione del pixel in y
    
    ## x_skew and y_skew
    
    #p1_Geom = [Longitude2[0,0], Latitude2[0,0]] # UpperLeft
    #p2_geom = [Longitude2[img2.shape[0]-1, img2.shape[1]-1], 
                       #Latitude2[img2.shape[0]-1, img2.shape[1]-1]] #LowerRight
    
    #p1_px = [0,0]
    #p2_px = [img2.shape[0]-1, img2.shape[1]-1]
    
    
    #x_skew = np.sqrt((p1_Geom[0] - p2_geom[0])**2 + (p1_Geom[1]- p2_geom[1])**2)/(p1_px[1]-p2_px[1])
    #y_skew = np.sqrt((p1_Geom[0] - p2_geom[0])**2 + (p1_Geom[1]- p2_geom[1])**2)/(p1_px[0]-p2_px[0])
    
    
        # Image extent (based in R code PRISMAREAD)
        
        # En realidad es - 15 a ver que sale poniendo -r/2
    
    ex = {'xmin' : geo['xmin'] - res/2,
          'xmax': geo['xmin'] - res/2 + img2.shape[1] * res,
          'ymin': geo['ymin'] - res/2,
          'ymax': geo['ymin'] - res/2    + img2.shape[0] * res}
    
        
    
    # Si queremos salvar la panchromatica
    

    
    # Set the resolution
    
    
    GeoT = (ex['xmin'], res, 0, ex['ymax'], 0, -res)
    
    driver = gdal.GetDriverByName(str("GTiff"))
    
    #Projj = Proj(proj = geo['proj_name'].lower().decode('UTF-8'),
                 #zone = int(geo['proj_code']),
                 #ellps = 'WGS84', 
                 #preserve_units = False)
    
    Projj = osr.SpatialReference()
    Projj.ImportFromEPSG(int(geo['proj_epsg'])) #4326
    Projj.ExportToPrettyWkt()

    #if Projj != 0:
        #raise RuntimeError(repr(res) + ': could not import from EPSG')

    
    return GeoT, driver, Projj



def CreateRGBTiff(outputRGB, img2, driver, GeoT, Projj):
    
    
    rows = img2.shape[0]
    cols = img2.shape[1]
    
    driver = gdal.GetDriverByName(str('GTiff'))
    
    DataSet = driver.Create(outputRGB, cols, rows, 3, gdal.GDT_Float32)
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection(Projj.ExportToWkt()) #.ExportToWkt()
    

    print('Voy a escribir')    
    for i,o in zip([20,19,9], [0,1,2]):
        image = img2[:,:,i]
        DataSet.GetRasterBand(o+1).WriteArray(image)
    DataSet.FlushCache()
    
    
def CreateTiffPan(output, img2, driver, GeoT, Projj):
    
    """
    This function saves the Panchromatic band of the PRISMA he5 images
    """
    
    rows = img2.shape[0]
    cols = img2.shape[1]
    
    driver = gdal.GetDriverByName(str('GTiff'))
    
    DataSet = driver.Create(output, cols, rows, 1, gdal.GDT_Float32)
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection(Projj.ExportToWkt())
    
    DataSet.GetRasterBand(1).WriteArray(img2)
    DataSet.FlushCache()
    
    
    

def CreateTiff(output, img2, driver, GeoT, Projj):
    
    rows = img2.shape[0]
    cols = img2.shape[1]
    band = img2.shape[2]
    
    driver = gdal.GetDriverByName('GTiff')
    
    DataSet = driver.Create(output, cols, rows, band, gdal.GDT_Float32)
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection(Projj.ExportToWkt())
    
    for i in range(band):
        image = img2[:,:,i]
        DataSet.GetRasterBand(i+1).WriteArray(image)
    DataSet.FlushCache()
    
        
def ComputeStat(img2, plot):
    
    
    """
    This function computes the layer statistic of the input image
    
    Inputs:
        
        img2 = numpy 3D array with axis=2 equal to the number of bands
        plot = boolean (True/False). If True one plot is generated
        
    Outputs:
        
        statis = pandas DataFrame with the Mean, Min, Max, Std per band
        
    """
    
    minn = []
    maxx = []
    mean = []
    std = []
    band_name = []
    
    for band in range(img2.shape[2]):
        
        minn.append(np.min(img2[:,:,band]))
        maxx.append(np.max(img2[:,:,band]))
        mean.append(np.mean(img2[:,:,band]))
        std.append(np.std(img2[:,:,band]))
        band_name.append('Band ' + str(band+1))
        
    statis = pd.DataFrame({'Basic Stats':band_name, 'Min': minn, 'Max':maxx, 'Mean':mean, 'Stdev':std})
    
    if plot:
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12,6))
        
        ax.plot(statis.Min, 'red', label= 'Minimum')
        ax.plot(statis.Mean, 'white', label= 'Mean')
        ax.plot(statis.Mean - statis.Stdev, 'lawngreen', label='Mean-Stdev')
        ax.plot(statis.Mean + statis.Stdev, 'lawngreen', label='Mean + Stdev')
        ax.plot(statis.Max, 'red', label='Maximum')
        
        ax.set_xlabel('Band Number')
        ax.set_ylabel('Value')
        #plt.ylim([-0.2,0.8])
        plt.grid('on')
        plt.legend()
        
    return statis

## ESTA FUNCIÓN TENDRÀ QUE SER MODIFICADA SI EL MÈTODO FUNCIONA

def SavePanSharpened(dir0, SCENE, image_name, img, res ):
    
    """
    At the moment this function saves the pansharpened images generated with Matlab
    
    INPUTS:
        
        img = numpy array with the image to be saved
        resolution = int; image resolution
        
    OUTPUTS:
        
        saved GeoTiff in the PanSharpened fodler
    
    """
    
    # Load the polygon used to crop the beach area (recordemos que he tenido que borrar
    # una columna y una fila para poder obtener la pansharpened)
    
    # Load the epsg from image
    
    dir_image = os.path.join(dir0, SCENE, image_name)

    
    # Load the image
    
    f = h5py.File(dir_image, 'r')
    
    
    
    # Load the polygon used to perform the crop
    
    file_poly = 'Poligon_Square.shp'
    
    path_poly = os.path.join(dir0, SCENE, 'Rubbish', file_poly)
    
    
    ds = ogr.Open(path_poly)
    poly = ds.GetLayer(0)
    
    
    # Get the geo information
    
    geo = dict()
    
    geo['xmin'] = poly.GetExtent()[0]
    geo['xmax'] = poly.GetExtent()[1]
    geo['ymin'] = poly.GetExtent()[2]
    geo['ymax'] = poly.GetExtent()[3]
    geo['proj_epsg'] = f.attrs['Epsg_Code']
        # 
    
    GeoT, driver, Proje = GetGeoTransform(geo, img, res)  
    
    # SAVE THE IMAGE
    
    print('Saving the image ...')
    
    dir_save = os.path.join(dir0,SCENE, 'PanSharpened')
    
    if not os.path.exists(dir_save):
        os.makedirs(dir_save, exist_ok=True)
        
    new_name = image_name[:-4] + 'PanSharpened.tiff'
    print(new_name)
    
    output = os.path.join(dir_save, new_name)

    CreateTiff(output, img, driver, GeoT, Proje)
    
    print('The image ', os.path.join(dir_save, new_name), 'has been created')
    
    

      
def SaveBandsTxt(dir_save, img, CW):
    
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
        
    o = 0
    for n in range(img.shape[2]):
        name = 'Band_' + str(o) + '_CW_' + str(CW[n])
        final_path = os.path.join(dir_save, name)
        np.savetxt(final_path, img[:,:,n], fmt='%.18e', delimiter = '\t', newline = '\n')
        o = o + 1

    print('All the bands have been saved')