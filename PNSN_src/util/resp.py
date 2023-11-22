"""
:module: util.resp
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: CC-BY-4.0

:purpose:
    This module will contain convenience methods for handling
    instrument response information in various formats.
"""
import os
from obspy.io.sac import attach_paz
def inv_to_paz(inventory, save_path='.', file_fmt='{sta}.{chan}.paz'):
    """
    
    """
    return None

def attach_paz_to_stream(stream, paz_file_list):
    """
    Given a list of *.pz file names (and paths) and a stream
    iterate across traces in the stream and attach
    """
    for _f in paz_file_list:
        
    for _tr in stream:
        sta = _tr.stats.station
        cha = _tr.stats.channel


    return None