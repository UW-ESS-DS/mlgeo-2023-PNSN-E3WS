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
from glob import glob
from obspy.io.sac import attach_paz
from obspy import Stream


def inv_to_paz(inventory, save_path=".", file_fmt="{sta}.{chan}.paz"):
    """ """
    return None


def attach_paz_to_stream(stream, paz_file_dir, **options):
    """
    Given a list of SACPZ file names (and paths) and a stream
    attach SACPZ responses to each valid trace-file combination

    :: INPUTS ::
    :param stream: [obspy.core.stream.Stream]
    :param paz_file_dir: [str]
                string of path/file entries that can be reached from
                the current working directory that has the following
                path/file naming convention:
                `paz_file_dir/station.channel.pz`

    :param **options: collector for key-word-arguments for
                obspy.io.sac.sacpz.attach_pz()

    # TODO
    :param omit_unattached: [bool]
                True  = return stream containing only traces with
                        attached `paz`
                False = return stream containing all traces, both
                        those with successfully attached `paz` and
                        those that did not have matching files


    :: OUTPUT ::
    :return st_out: [obspy.core.stream.Stream]
                Output stream with (varibly) attached poles and zeros
    """
    # Use glob.glob to fetch file list
    paz_file_list = glob(os.path.join(paz_file_dir, "*.*.pz"))
    # Create output stream
    st_out = Stream()
    # Iterate across *.pz files
    for _f in paz_file_list:
        # Get station and channel names from files
        sta, cha, _ = os.path.split(_f)[-1].split(".")
        # Iterate across a copy of the input stream
        for _tr in stream.copy():
            # If station and channel name match...
            if _tr.stats.station == sta and _tr.stats.channel == cha:
                attach_paz(_tr, _f, **options)
                st_out += _tr
            # TODO: handle traces without a matching *.pz file

    return st_out
