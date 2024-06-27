#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy

def single_float_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile['count']=1
    profile['dtype']=np.float32
    profile["compress"] = "deflate"
    
    return profile


def single_old_bool_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile['count'] = 1
    profile['dtype'] = bool
    profile["compress"] = "deflate"
    
    return profile
    

def single_bool_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile['count'] = 1
    profile['dtype'] = np.uint8
    profile['nbits'] = 1
    profile["compress"] = "deflate"
    profile["nodata"] = None

    return profile

def single_uint8_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile["count"]= 1
    profile["dtype"]= np.uint8
    profile["compress"] = "deflate"
    profile["nodata"] = 255
    
    return profile

def single_int16_profile(input_profiles: list, map_params):
    profile= input_profiles[0]
    profile["count"]= 1
    profile["dtype"]= np.int16
    profile["nodata"] = 32767
    profile["compress"] = "deflate"
    
    return profile

def single_int32_profile(input_profiles: list, map_params):
    profile= input_profiles[0]
    profile["count"]= 1
    profile["dtype"]= np.int32
    profile["compress"] = "deflate"

    return profile


def three_uint8_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile["count"]= 3
    profile["dtype"]= np.uint8
    profile["compress"] = "lzw"
    profile["nodata"] = 255
    
    return profile

def three_int16_profile(input_profiles: list, map_params):
    profile = input_profiles[0]
    profile["count"]= 3
    profile["dtype"]= np.int16
    profile["compress"] = "deflate"
    profile["nodata"] = 32767
    
    return profile

def three_int32_profile(input_profiles: list, map_params):
    profile= input_profiles[0]
    profile["count"]= 3
    profile["dtype"]= np.int32
    profile["compress"] = "deflate"
    
    return profile
    
def double_int_profile(input_profiles: list, map_params):
    profile1 = input_profiles[0]
    profile1['count']=1
    profile1['dtype']=np.uint8
    profile1['nodata'] = 255
    profile1["compress"] = "deflate"
    
    # avoid to modify profile1
    profile2 = copy.deepcopy(profile1)
       
    return [profile1, profile2] 
