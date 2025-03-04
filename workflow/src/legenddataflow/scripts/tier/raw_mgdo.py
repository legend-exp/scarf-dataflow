import argparse
from pathlib import Path

import lgdo
import numpy as np
import ROOT
from dbetto import TextDB
from lgdo import Array, LH5Store, WaveformTable
from lgdo.compression import RadwareSigcompress

from ...log import build_log

ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.StartGUIThread = False
ROOT.PyConfig.DisableRootLogon = True
ROOT.PyConfig.IgnoreCommandLineOptions = True


def build_tier_raw_mgdo() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input", help="input file", type=str)
    argparser.add_argument("output", help="output file", type=str)
    argparser.add_argument("--datatype", help="Datatype", type=str, required=True)
    argparser.add_argument("--timestamp", help="Timestamp", type=str, required=True)
    argparser.add_argument("--configs", help="config file", type=str)
    argparser.add_argument("--chan-maps", help="chan map", type=str)
    argparser.add_argument("--log", help="log file", type=str)
    args = argparser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    config_dict = (
        TextDB(args.configs, lazy=True)
        .on(args.timestamp, system=args.datatype)
        .snakemake_rules.tier_raw_mgdo
    )

    build_log(config_dict, args.log)

    chmap = (
        TextDB(args.chan_maps, lazy=True)
        .channelmaps.on(args.timestamp)
        .map("daq.struckid")
    )

    input_file = ROOT.TFile(args.input)
    tree = input_file.Get("MGTree")

    # this will store the channel tables that will be written to file
    data_dict = {}

    # get info needed to initialize buffer lh5 table
    tree.GetEntry(0)
    event = tree.event

    found_struckids = set()
    for i in range(event.GetNWaveforms()):
        struckid = event.GetDigitizerData(i).GetID()
        found_struckids.add(struckid)

        if event.GetAuxWaveformArrayStatus():
            wf_pre = event.GetWaveform(i)
            wf_win = event.GetAuxWaveform(i)

            tbl = _make_lh5_channel_buffer(
                wf_pre=wf_pre,
                wf_win=wf_win,
            )
        else:
            tbl = _make_lh5_channel_buffer(
                wf=event.GetWaveform(i),
            )

        data_dict[_tblid(chmap[struckid].daq.rawid)] = tbl

    if any(ch not in found_struckids for ch in chmap):
        msg = (
            "could not find data from some channelmap channels "
            f"in the DAQ file: {found_struckids=}, {chmap.keys()=}"
        )
        raise ValueError(msg)

    # TODO: is this needed?
    tree.ResetBranchAddresses()

    store = LH5Store(keep_open=True)

    for _ in tree:
        event = tree.event
        for i in range(event.GetNWaveforms()):
            ddata = event.GetDigitizerData(i)
            struckid = ddata.GetID()
            tbl_name = _tblid(chmap[struckid].daq.rawid)
            tbl = data_dict[tbl_name]

            tbl["timestamp"][tbl.loc] = (
                ddata.GetTimeStamp() + ddata.GetDecimalTimeStamp() * 1e-9
            )
            # tbl[][tbl.loc] = ddata.()
            tbl["daq_energy_sum"][tbl.loc] = event.GetETotal()
            tbl["time_of_first_hit_sec"][tbl.loc] = event.GetTime()
            tbl["event_type"][tbl.loc] = event.GetEventType()

            tbl["daq_crate"][tbl.loc] = ddata.GetCrate()
            tbl["daq_card"][tbl.loc] = ddata.GetCard()
            tbl["daq_channel"][tbl.loc] = ddata.GetChannel()
            tbl["clock_freq_hz"][tbl.loc] = ddata.GetClockFrequency()
            tbl["bit_resolution"][tbl.loc] = ddata.GetBitResolution()
            tbl["nr_of_channels"][tbl.loc] = ddata.GetNChannels()
            tbl["event_number"][tbl.loc] = ddata.GetEventNumber()
            tbl["pre_trigger"][tbl.loc] = ddata.GetPretrigger()
            tbl["trigger_number"][tbl.loc] = ddata.GetTriggerNumber()
            tbl["is_muon_vetoed"][tbl.loc] = ddata.IsMuVetoed()
            tbl["muon_veto_sample"][tbl.loc] = ddata.GetMuVetoSample()
            tbl["waveform_tag"][tbl.loc] = ddata.GetWaveformTag()
            tbl["is_inverted"][tbl.loc] = ddata.IsInverted()

            if event.GetAuxWaveformArrayStatus():
                for name, wf in zip(
                    ["waveform_presummed", "waveform_windowed"],
                    [event.GetWaveform(i), event.GetAuxWaveform(i)],
                ):
                    tbl[name]["dt"][tbl.loc] = wf.GetSamplingPeriod()
                    tbl[name]["t0"][tbl.loc] = wf.GetTOffset()
                    tbl[name]["values"][tbl.loc][:] = np.frombuffer(
                        wf.GetVectorData().data()
                    )
            else:
                wf = event.GetWaveform(i)
                tbl["waveform"]["dt"][tbl.loc] = wf.GetSamplingPeriod()
                tbl["waveform"]["t0"][tbl.loc] = wf.GetTOffset()
                tbl["waveform"]["values"][tbl.loc][:] = np.frombuffer(
                    wf.GetVectorData().data()
                )

            tbl.push_row()

            if tbl.is_full():
                store.write(tbl, tbl_name, args.output, wo_mode="append")
                tbl.clear()


def _tblid(id):
    return f"ch{id:03d}/raw"


def _make_lh5_channel_buffer(wf=None, wf_win=None, wf_pre=None, size=1024):
    col_dict = {
        "timestamp": Array(shape=(size,), dtype="float64", attrs={"units": "s"}),
        "daq_energy_sum": Array(shape=(size,), dtype="float64"),
        "time_of_first_hit_sec": Array(shape=(size,), dtype="float64"),
        "event_type": Array(shape=(size,), dtype="int32"),
        "daq_crate": Array(shape=(size,), dtype="uint32"),
        "daq_card": Array(shape=(size,), dtype="uint32"),
        "daq_channel": Array(shape=(size,), dtype="uint32"),
        "clock_freq_hz": Array(shape=(size,), dtype="float64", attrs={"units": "Hz"}),
        "bit_resolution": Array(shape=(size,), dtype="uint32"),
        "nr_of_channels": Array(shape=(size,), dtype="uint32"),
        "event_number": Array(shape=(size,), dtype="int32"),
        "pre_trigger": Array(shape=(size,), dtype="uint32"),
        "trigger_number": Array(shape=(size,), dtype="uint32"),
        "is_muon_vetoed": Array(shape=(size,), dtype="bool"),
        "muon_veto_sample": Array(shape=(size,), dtype="uint32"),
        "waveform_tag": Array(shape=(size,), dtype="int32"),
        "is_inverted": Array(shape=(size,), dtype="bool"),
    }

    common_kwargs = {
        "size": size,
        "t0": Array(shape=(size,), dtype=int, attrs={"units": "s"}),
        "dt": Array(shape=(size,), dtype=int, attrs={"units": "s"}),
    }

    wf_values_attrs = {"compression": RadwareSigcompress(codec_shift=-32768)}

    if wf is not None:
        col_dict["waveform"] = WaveformTable(
            wf_len=wf.GetLength(),
            dtype="uint16",
            **common_kwargs,
        )
        col_dict["waveform"].values.attrs |= wf_values_attrs
    else:
        for name, wf in zip(
            ["waveform_presummed", "waveform_windowed"], [wf_pre, wf_win]
        ):
            col_dict[name] = WaveformTable(
                wf_len=wf.GetLength(),
                dtype="uint16",
                **common_kwargs,
            )
            col_dict[name].values.attrs |= wf_values_attrs

    return lgdo.Table(size=size, col_dict=col_dict)
