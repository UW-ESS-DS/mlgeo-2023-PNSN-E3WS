"""
:module: drivers.build_waveform_dataset.get_RESP_inv_from_bulk
:auth: Nathan T. Stevens
:email: ntsteven (at) uw.edu
:org: Pacific Northwest Seismic Network
:license: CC-BY-4.0

:purpose:
    This driver script loads header information from
    bulkXtsY.mseed files and gets an inventory containing
    instrument responses (RESP_inv) for each NSCL combination
    present in the bluk*.mseed file. 

    It provides options for exporting both StationXML files
    for each bulk file and SAC Poles and Zeros (*.paz) files
    for each station-channel in bluk*.mseed

"""
import os
from glob import glob
from obspy import read
from obspy.clients.fdsn import Client
from tqdm import tqdm

output_xml = True
output_paz = True

# Map relative paths to root directory for process
ROOT = os.path.join("..", "..", "..", "PNSN_data")
# Create glob search string
GSTR = os.path.join(ROOT, "EVID*/*.mseed")
# Get file list
flist = glob(GSTR)
flist.sort()

# Initialize IRIS waveform client
client = Client("IRIS")

for _f in tqdm(flist):
    # Get path from _f
    path, _ = os.path.split(_f)
    ofpn = os.path.join(path, "station.xml")
    # Get waveform header information
    st = read(_f, head_only=True)
    

    if output_xml and not os.path.isfile(ofpn):
        # Create lists for waveform response
        stations = []
        channels = []
        network = "UW"
        sts = []
        ets = []
        for _tr in st:
            if _tr.stats.station not in stations:
                stations.append(_tr.stats.station)
            if _tr.stats.channel not in channels:
                channels.append(_tr.stats.channel)
            sts.append(_tr.stats.starttime)
            ets.append(_tr.stats.endtime)
        sta_str = ",".join(stations)
        cha_str = ",".join(channels)
        st_min = min(sts)
        et_max = max(ets)

        inv = client.get_stations(
            startbefore=st_min,
            endafter=et_max,
            station=sta_str,
            channel=cha_str,
            level="response",
        )
        # Write inventory out as a stationXML file
        inv.write(ofpn, format="STATIONXML")

    if output_paz and not os.path.exists(os.path.join(path,'paz')):
        if not os.path.exists(os.path.join(path, "paz")):
            os.makedirs(os.path.join(path, "paz"))
        for _tr in st:
            keys = ["startbefore", "endafter", "station", "channel", "level"]
            stats = [
                _tr.stats[x] for x in ["starttime", "endtime", "station", "channel"]
            ] + ["response"]
            kw = dict(zip(keys, stats))
            inv = client.get_stations(**kw)
            # Write response out to sacpz by channel:
            inv.write(
                os.path.join(path, "paz", f"{stats[2]}.{stats[3]}.pz"), format="SACPZ"
            )
