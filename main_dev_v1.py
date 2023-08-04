#!/usr/bin/env python3
# coding: utf8
# Written by: Galchenkova M. and Yefanov O., 2022
# source /home/galchenm/P09/om/install/bin/activate

"""
Before using this program, you need to do next steps after loading all necessary files:

g++ -shared -o rings_rejection.so -fPIC rings_rejection.cpp
g++ -shared -o SubLocalBG.so -fPIC SubLocalBG.c
g++ -shared -o peakfinder8.so -fPIC peakfinder8.cpp

Call help function for becoming familiar with required/optional parameters:

python3 main.py --help

Example:
 
python3 main.py -f files.lst -o ./ring_mask_auto -e cbf -m mask_v0.h5 -mh5 /data/data -g HEX-1cyto-predrefine.geom
"""

import os
import numpy as np
import sys
import argparse
import ctypes as ct
import h5py as h5
#from cfelpyutils import crystfel_utils, geometry_utils
from om.utils import crystfel_geometry, exceptions, parameters, zmq_monitor
from om.utils.crystfel_geometry import TypeDetector, TypePixelMaps
import fabio
from numpy.ctypeslib import ndpointer
import glob

import time
from typing import Any, Deque, Dict, List, Tuple, Union
import concurrent.futures
import h5py  # type: ignore
import numpy as np  # type: ignore

from om.algorithms import crystallography as cryst_algs
from om.algorithms import generic as gen_algs
from om.processing_layer import base as process_layer_base
from om.utils import crystfel_geometry, exceptions, parameters, zmq_monitor
from om.utils.crystfel_geometry import TypeDetector, TypePixelMaps
from om.algorithms.crystallography import TypePeakfinder8Info
from dataclasses import dataclass, field

from om.algorithms.crystallography import TypePeakList, Peakfinder8PeakDetection

os.nice(0)

# constant values
MIN_VAL =  -0.5
MAX_VAL = 65000 #Change this parameter if it is needed 

ISTEP=1 #Change this parameter if it is needed 

ASIC_NX=2463 #Change this parameter if it is needed
ASIC_NY=2527 #Change this parameter if it is needed
NASICS_X=1 #Change this parameter if it is needed
NASICS_Y=1 #Change this parameter if it is needed

WIND = 10
DIFFER = 0.05

#hit finder parameter for salt and ice rings masking
HIT_FINDER_MIN_SNR=3.5 #Change this parameter if it is needed 
ADC_THRESH=5 #Change this parameter if it is needed 
HIT_FINDER_MIN_PIX_COUNT=10 #Change this parameter if it is needed 
HIT_FINDER_MAX_PIX_COUNT=500000 #Change this parameter if it is needed 
HIT_FINDER_LOCAL_BG_RADIUS=1 #Change this parameter if it is needed 

#hit finder parameter for non-hits-rejection
HIT_FINDER_MIN_SNR_NON_HIT=8 #Change this parameter if it is needed 
ADC_THRESH_NON_HIT=5 #Change this parameter if it is needed 
HIT_FINDER_MIN_PIX_COUNT_NON_HIT=1 #Change this parameter if it is needed 
HIT_FINDER_MAX_PIX_COUNT_NON_HIT=10 #Change this parameter if it is needed 
HIT_FINDER_LOCAL_BG_RADIUS_NON_HIT=3 #Change this parameter if it is needed 
MIN_RES=0 #Change this parameter if it is needed 
MAX_RES=1200 #Change this parameter if it is needed 

MASK_RAD = 10 #Change this parameter if it is needed 

MIN_PEAKS = 20 #Change this parameter if it is needed 

MAX_NUM_PEAKS = 10000 #Change this parameter if it is needed 

class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                      argparse.ArgumentDefaultsHelpFormatter):
    pass

def parse_cmdline_args():
    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        formatter_class=CustomFormatter)
    parser.add_argument('-p', '--p', type=str, help="The folder with files")
    parser.add_argument('-f', '--f', type=str, help="Files.lst containes all files that it is necessary to copy to the folder")
    parser.add_argument('-o', '--o', type=str, help="Output folder")
    parser.add_argument('-e','--e', default='cbf', type=str, help="Extension of files (cbf, h5 or cxi)")
    parser.add_argument('-h5', '--h5path', default='/entry/data/data', type=str, help="If files have an hdf5 format? you have to specify the path to the data")
    parser.add_argument('-m', '--m', type=str, help="Static mask")
    parser.add_argument('-mh5', '--mh5', default='/data/data', type=str, help="The path to data in mask")
    parser.add_argument('-g', type=str, help="Geometry filename")
    return parser.parse_args()

def _np_ptr(np_array):
    return ct.c_void_p(np_array.ctypes.data)


class PeakFinderStructure(ct.Structure):
    _fields_=[('nPeaks',ct.c_long), ('nHot',ct.c_long), ('peakResolution',ct.c_float), ('peakResolutionA',ct.c_float), ('peakDensity',ct.c_float), ('peakNpix',ct.c_float), 
              ('peakTotal',ct.c_float), ('memoryAllocated',ct.c_int), ('nPeaks_max',ct.c_long), ('peak_maxintensity',ct.POINTER(ct.c_float)), ('peak_totalintensity',ct.POINTER(ct.c_float)), 
              ('peak_sigma',ct.POINTER(ct.c_float)), ('peak_snr',ct.POINTER(ct.c_float)), ('peak_npix',ct.POINTER(ct.c_float)), ('peak_com_x',ct.POINTER(ct.c_float)), ('peak_com_y',ct.POINTER(ct.c_float)), ('peak_com_index',ct.POINTER(ct.c_long)), 
              ('peak_com_x_assembled',ct.POINTER(ct.c_float)), ('peak_com_y_assembled',ct.POINTER(ct.c_float)), ('peak_com_r_assembled',ct.POINTER(ct.c_float)), ('peak_com_q',ct.POINTER(ct.c_float)), ('peak_com_res',ct.POINTER(ct.c_float))]


def PeakFinder8py(peaklist, data, mask, pix_r,
                  ASIC_NX, ASIC_NY, NASICS_X, NASICS_Y,
                  ADC_THRESH, HIT_FINDER_MIN_SNR,
                  HIT_FINDER_MIN_PIX_COUNT, HIT_FINDER_MAX_PIX_COUNT,
                  HIT_FINDER_LOCAL_BG_RADIUS, outliersMask):
    req = PeakFinderStructure()
    req.nPeaks_max = MAX_NUM_PEAKS
    lib = ct.CDLL('/gpfs/cfel/group/cxi/scratch/2020/EXFEL-2019-Schmidt-Mar-p002450/scratch/galchenm/scripts_for_work/ring_mask_auto/peakfinder8.so')
    pfun = lib.peakfinder8
    pfun.restype = ct.c_int
    data = np.array(data, dtype=np.float32)
    pix_r = np.array(pix_r,dtype=np.float32)
    mask = np.array(mask, dtype=np.int8)
    len_outliersMask = len(data)
    outliersMask_buf = np.zeros(len_outliersMask, dtype=np.int8)
    
    pfun.argtypes = (ct.POINTER(PeakFinderStructure),ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_long,ct.c_long,ct.c_long,ct.c_long,ct.c_float,ct.c_float,ct.c_long,ct.c_long,ct.c_long,ct.c_void_p)
    int_flag = pfun(ct.byref(req),_np_ptr(data),_np_ptr(mask),_np_ptr(pix_r),ASIC_NX, ASIC_NY, NASICS_X, NASICS_Y,
                    ADC_THRESH, HIT_FINDER_MIN_SNR,
                    HIT_FINDER_MIN_PIX_COUNT, HIT_FINDER_MAX_PIX_COUNT,
                    HIT_FINDER_LOCAL_BG_RADIUS, _np_ptr(outliersMask_buf))
    if outliersMask is not None:
        outliersMask[:] = outliersMask_buf.copy()
    return outliersMask_buf

def DilateMask(mask, inpMask, ASIC_NX, ASIC_NY, NASICS_X, NASICS_Y, maskRad=MASK_RAD):
    lib = ct.CDLL('/gpfs/cfel/group/cxi/scratch/2020/EXFEL-2019-Schmidt-Mar-p002450/scratch/galchenm/scripts_for_work/ring_mask_auto/peakfinder8.so')
    pfun = lib.DilateMask
    pfun.restype = ct.c_int
    mask = np.array(mask, dtype=np.int8) 
    mask_buf = np.zeros(len(mask), dtype=np.int8)
    inpMask = np.array(inpMask, dtype=np.int8)
    pfun.argtypes = ct.c_void_p, ct.c_void_p, ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_int
    int_flag = pfun(_np_ptr(mask_buf), _np_ptr(inpMask), ASIC_NX, ASIC_NY, NASICS_X, NASICS_Y, maskRad)
    mask[:] = mask_buf.copy()
    
    return mask
    
def pBuildRadiulArray(NxNy, det_x, det_y, ISTEP, pix_r, maxRad, pixelsR):
    det_x = np.array(det_x, dtype=np.float32)
    det_y = np.array(det_y, dtype=np.float32)
    pix_r = np.array([0] * NxNy, dtype=np.int32)
    pixelsR = np.array([0] * NxNy, dtype=np.float32)
    maxRad = ct.c_int32()
    lib = ct.CDLL( '/gpfs/cfel/group/cxi/scratch/2020/EXFEL-2019-Schmidt-Mar-p002450/scratch/galchenm/scripts_for_work/ring_mask_auto/SubLocalBG.so')
    pfun = lib.BuildRadialArray
    pfun.restype = ct.c_bool
    pfun.argtypes = (ct.c_size_t, ct.c_void_p, ct.c_void_p, ct.c_float, ct.c_void_p, ct.POINTER(ct.c_int), ct.c_void_p)
    flag_BRA = pfun(NxNy, _np_ptr(det_x), _np_ptr(det_y), ISTEP, _np_ptr(pix_r), ct.byref(maxRad), _np_ptr(pixelsR))
    return flag_BRA, pix_r, maxRad.value, pixelsR

def MaskRingsSimplepy(tileFs, tileSs, nTilesFs, nTilesSs, data, staticMask, MIN_VAL, MAX_VAL, maxRad, pix_r, WIND, DIFFER, outMask):
    data = np.array(data, dtype=np.float32)
    pix_r = np.array(pix_r,dtype=np.float32)
    staticMask = np.array(staticMask, dtype=np.int8)
    outMask = np.array([0]*len(pix_r), dtype=np.int8)
    lib = ct.CDLL( '/gpfs/cfel/group/cxi/scratch/2020/EXFEL-2019-Schmidt-Mar-p002450/scratch/galchenm/scripts_for_work/ring_mask_auto/rings_rejection.so' )
    pfun = lib.main
    pfun.restype = ct.c_int
    pfun.argtypes = (ct.c_int, ct.c_int, ct.c_int, ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_float, ct.c_float, ct.c_int, ct.c_void_p, ct.c_int, ct.c_float, ct.c_void_p)
    nummasked = pfun(tileFs, tileSs, nTilesFs, nTilesSs, _np_ptr(data), _np_ptr(staticMask), MIN_VAL, MAX_VAL, maxRad, _np_ptr(pix_r), WIND, DIFFER, _np_ptr(outMask))
    return outMask, nummasked 

def copy_hdf5(file, output_folder):
    h5r = h5.File(file, 'r')
    file_copy = os.path.join(output_folder, os.path.basename(file))
    with h5.File(file_copy, 'w') as h5w:
        for obj in h5r.keys():        
            h5r.copy(obj, h5w)       
    h5r.close()

def allkeys(obj, current_key='', results=None):
    """
    Recursively find all keys (paths) in an h5py.Group.
    Returns a list of keys.
    """
    if results is None:
        results = []

    if isinstance(obj, h5py.Group):
        for key, value in obj.items():
            new_key = f"/{current_key}/{key}" if current_key else key
            
            if isinstance(value, h5py.Group):
                allkeys(value, current_key=new_key, results=results)
            else:
                results.append(new_key)

    return results

def is_hit(data):
    global geometry
    @dataclass
    class PF8Info:
        
        max_num_peaks: int
        pf8_detector_info: dict
        adc_threshold: float
        minimum_snr: int
        min_pixel_count: int
        max_pixel_count: int
        local_bg_radius: int
        min_res: float
        max_res: float
        _bad_pixel_map: np.array
        _pixelmaps: np.array = field(init=False)

        def __post_init__(self):
            self._pixelmaps: TypePixelMaps = crystfel_geometry.compute_pix_maps(geometry)

    class PF8:
        def __init__(self, info):
            assert isinstance(
                info, PF8Info
            ), f"Info object expected type PF8Info, found {type(info)}."
            self.pf8_param = info

        def get_peaks_pf8(self, data):
            detector_layout = self.pf8_param.pf8_detector_info

            peak_detection = Peakfinder8PeakDetection(
                self.pf8_param.max_num_peaks,
                self.pf8_param.pf8_detector_info["asic_nx"],
                self.pf8_param.pf8_detector_info["asic_ny"],
                self.pf8_param.pf8_detector_info["nasics_x"],
                self.pf8_param.pf8_detector_info["nasics_y"],
                self.pf8_param.adc_threshold,
                self.pf8_param.minimum_snr,
                self.pf8_param.min_pixel_count,
                self.pf8_param.max_pixel_count,
                self.pf8_param.local_bg_radius,
                self.pf8_param.min_res,
                self.pf8_param.max_res,
                self.pf8_param._bad_pixel_map.astype(np.float32),
                (self.pf8_param._pixelmaps["radius"]).astype(np.float32),
            )
            peaks_list = peak_detection.find_peaks(data)
            return peaks_list

    pf8_info = PF8Info(
        max_num_peaks=MAX_NUM_PEAKS,
        pf8_detector_info=dict(
            asic_nx=ASIC_NX,
            asic_ny=ASIC_NY,
            nasics_x=NASICS_X,
            nasics_y=NASICS_Y,
        ),
        adc_threshold=ADC_THRESH_NON_HIT,
        minimum_snr=HIT_FINDER_MIN_SNR_NON_HIT,
        min_pixel_count=HIT_FINDER_MIN_PIX_COUNT_NON_HIT,
        max_pixel_count=HIT_FINDER_MAX_PIX_COUNT_NON_HIT,
        local_bg_radius=HIT_FINDER_LOCAL_BG_RADIUS_NON_HIT,
        min_res=MIN_RES,
        max_res=MAX_RES,
        _bad_pixel_map=np.ones((ASIC_NY,ASIC_NX), dtype=np.int8),
    )

    pf8 = PF8(pf8_info)

    peak_list = pf8.get_peaks_pf8(data=data)
    num_of_peaks = peak_list['num_peaks']
    print(num_of_peaks, num_of_peaks < MIN_PEAKS)
    if num_of_peaks < MIN_PEAKS:
        return False
    return True

def copy_hdf5_with_non_hits_rejection(file, h5path, init_shape, max_shape, output_folder):
    h5r = h5.File(file, 'r')
    all_h5r_keys = allkeys(h5r)
    add_keys = [key for key in all_h5r_keys if key != h5path]
    
    data = h5r[h5path][()]
    
    num, data_shape = data.shape[0], data.shape[1:]
    
    file_copy = os.path.join(output_folder, f'non-hits-{os.path.basename(file)}')
    h5w = h5.File(file_copy, 'w')
    
    for key in all_h5r_keys:
        _ = h5w.create_dataset(key, init_shape, maxshape=max_shape, chunks=init_shape, compression="gzip", compression_opts=6)  
    
    index = 0
    for i in range(num):
        if is_hit(data[i,]):
            for key in all_h5r_keys:
                appended_data = h5w[key]
                appended_data.resize((index+1,) + mask_shape)
                appended_data[index, ] = h5r[h5path][i, ]
                index += 1
    h5r.close()
    h5w.clsoe()
    return file_copy

def reading_hdf5_files(files, h5path, staticMask_filename, mask_h5path, MIN_VAL, MAX_VAL, maxRad, pix_r, WIND, DIFFER, output_folder):

    for file in files:
        copy_hdf5(file, output_folder)
        
    for file in files:
        h5r = h5.File(file, 'r')
        data = h5r[h5path]
        num, data_shape = data.shape[0], data.shape[1:]
        
        if staticMask_filename is not None:
            staticMask = h5.File(staticMask_filename, 'r')[mask_h5path][()]
            mask_shape = staticMask.shape
            inverted_staticMask = (1 - staticMask).flatten()
            if mask_shape != data_shape:
                print(f'Warning: The problem with {file_copy}.Check the shape of mask ({mask_shape}) and the shape of your data ({data_shape}). They does not match each other.')
                h5r.close()
                break
        else:
            staticMask = np.ones_like(data, dtype=np.int8) #np.zeros_like(data, dtype=np.int8)
            mask_shape = data_shape        
        
        init_shape = (1,) + mask_shape
        max_shape = (num,) + mask_shape
        
        file_copy = copy_hdf5_with_non_hits_rejection(file, h5path, init_shape, max_shape, output_folder)
        
        with open(file_copy, 'a') as f:

            dilate_mask_data = f.create_dataset('/dilate/data', init_shape, maxshape=max_shape, dtype=np.int8, chunks=init_shape, compression="gzip", compression_opts=6)  
            data = f[h5path]
            
            for i in range(num):
                dilate_mask_data.resize((i+1,) + mask_shape)

                outliersMask_buf = np.zeros_like(staticMask.flatten(), dtype=np.int8)
                
                outMask, nummasked = MaskRingsSimplepy(ASIC_NX, ASIC_NY, NASICS_X, NASICS_Y, data_from_cbf.flatten(), staticMask.flatten(), MIN_VAL, MAX_VAL, maxRad, pix_r, WIND, DIFFER, None)
                
                outliersMask_salt = PeakFinder8py(None, data[i,].flatten(), outMask, \
                                        pix_r, ASIC_NX, ASIC_NY, NASICS_X, NASICS_Y, \
                                        ADC_THRESH, HIT_FINDER_MIN_SNR, \
                                        HIT_FINDER_MIN_PIX_COUNT, HIT_FINDER_MAX_PIX_COUNT, HIT_FINDER_LOCAL_BG_RADIUS, outliersMask_buf)
                
                mask_buf = np.zeros_like(staticMask.flatten(), dtype=np.int8)
                        
                outliersMask = DilateMask(mask_buf, outliersMask_salt, ASIC_NX, ASIC_NY, NASICS_X, NASICS_Y, MASK_RAD)
                outliersMask += inverted_staticMask
                outliersMask[outliersMask > 1] = 0
                outliersMask = 1 - outliersMask

                dilate_mask_data[i,] = outliersMask.reshape(mask_shape)
                
        h5r.close()
    else:
        return None
        
def reading_cbf_files(files, staticMask_filename, mask_h5path, MIN_VAL, MAX_VAL, maxRad, pix_r, WIND, DIFFER, output_folder):
    files.sort()
    
    num = len(files)
    data = fabio.open(files[0]).data
    data_shape = data.shape
    
    if staticMask_filename is not None:
        staticMask = h5.File(staticMask_filename, 'r')[mask_h5path][()]
        mask_shape = staticMask.shape
        inverted_staticMask = (1 - staticMask).flatten()
        if mask_shape != data_shape:
            print(f'Warning: check the shape of mask ({mask_shape}) and shape of your data ({data_shape}). They does not match each other.')
            return None
    else:
        staticMask = np.ones_like(data, dtype=np.int8) #np.zeros_like(data, dtype=np.int8)
        mask_shape = data_shape

    
    init_shape = (1,) + mask_shape
    max_shape = (num,) + mask_shape
    
    generated_hdf5_filename = os.path.join(output_folder, f'All-{os.path.basename(output_folder)}.h5')
    
    with h5.File(generated_hdf5_filename, 'w') as output_hdf5:
        data = output_hdf5.create_dataset('/data/data', init_shape, maxshape=max_shape, dtype=np.float32, chunks=init_shape, compression="gzip", compression_opts=6)
        dilate_mask_data = output_hdf5.create_dataset('/dilate/data', init_shape, maxshape=max_shape, dtype=np.int8, chunks=init_shape, compression="gzip", compression_opts=6)  
        salt_mask_data = output_hdf5.create_dataset('/salt/data', init_shape, maxshape=max_shape, dtype=np.int8, chunks=init_shape, compression="gzip", compression_opts=6) 
        
        index = 0
        for i in range(num):
            file = files[i]
            
            img = fabio.open(file)
            data_from_cbf = img.data
            
            if is_hit(data_from_cbf):
                
                data.resize((index+1,) + mask_shape)
                dilate_mask_data.resize((index+1,) + mask_shape)
                salt_mask_data.resize((index+1,) + mask_shape)

                data[index,] = data_from_cbf
                
                outliersMask_buf = np.zeros_like(staticMask.flatten(), dtype=np.int8)
                outMask, nummasked = MaskRingsSimplepy(ASIC_NX, ASIC_NY, NASICS_X, NASICS_Y, data_from_cbf.flatten(), staticMask.flatten(), MIN_VAL, MAX_VAL, maxRad, pix_r, WIND, DIFFER, None)

                outliersMask_salt = PeakFinder8py(None, data[index,].flatten(), outMask, \
                                        pix_r, ASIC_NX, ASIC_NY, NASICS_X, NASICS_Y, \
                                        ADC_THRESH, HIT_FINDER_MIN_SNR, \
                                        HIT_FINDER_MIN_PIX_COUNT, HIT_FINDER_MAX_PIX_COUNT, HIT_FINDER_LOCAL_BG_RADIUS, outliersMask_buf)
                
                mask_buf = np.zeros_like(staticMask.flatten(), dtype=np.int8)
                
                salt_mask_data[index,] = outliersMask_salt.reshape(mask_shape)
                
                outliersMask = DilateMask(mask_buf, outliersMask_salt, ASIC_NX, ASIC_NY, NASICS_X, NASICS_Y, MASK_RAD)
                outliersMask += inverted_staticMask
                outliersMask[outliersMask > 1] = 0
                outliersMask = 1 - outliersMask
                
                dilate_mask_data[index,] = outliersMask.reshape(mask_shape)
                index += 1
            
            
if __name__ == "__main__":
    args = parse_cmdline_args()
    
    geometry_filename = args.g
    staticMask_filename = args.m
    mask_h5path = args.mh5
    
    if args.o is None:
        output_folder = os.getcwd()
    else:
        output_folder = args.o
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print('Create')
        
    print(f"Your results can be found here {output_folder}")
    
    geometry, _, __ = crystfel_geometry.load_crystfel_geometry(geometry_filename)
    pixel_maps = crystfel_geometry.compute_pix_maps(geometry)
    
    
    x_map = pixel_maps['x']
    y_map = pixel_maps['y']
    r_map = pixel_maps['radius']
    
    
    len_x_map = len(x_map.flatten()) 
    flag_BRA, pix_r, maxRad, pixelsR = pBuildRadiulArray(len_x_map, x_map, y_map, ISTEP, None, 0, None)
    
    if args.p is not None:
        files = glob.glob(os.path.join(args.p, f'*{args.e}'))
        if args.e == 'cbf':
            
            reading_cbf_files(files, staticMask_filename, mask_h5path, MIN_VAL, MAX_VAL, maxRad, pix_r, WIND, DIFFER, output_folder)
        elif args.e == 'h5' or args.e == 'cxi':
            h5path = args.h5path
            reading_hdf5_files(files, h5path, staticMask_filename, mask_h5path, MIN_VAL, MAX_VAL, maxRad, pix_r, WIND, DIFFER, output_folder)
    elif args.f is not None:
        files = []
        with open(args.f, 'r') as read:
            for line in read:
                files.append(line.strip())
        if args.e == 'cbf':
            
            reading_cbf_files(files, staticMask_filename, mask_h5path, MIN_VAL, MAX_VAL, maxRad, pixelsR, WIND, DIFFER, output_folder)
        elif args.e == 'h5' or args.e == 'cxi':
            h5path = args.h5path
            reading_hdf5_files(files, h5path, staticMask_filename, mask_h5path, MIN_VAL, MAX_VAL, maxRad, pix_r, WIND, DIFFER, output_folder)              
    else:
        print('Warning: you have to provide path or list of files with absolute path to each file.')
       
       