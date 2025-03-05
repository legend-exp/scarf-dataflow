import argparse
from copy import deepcopy
from pathlib import Path

from daq2lh5 import build_raw
from dbetto import TextDB
from dbetto.catalog import Props

from ...log import build_log


def build_tier_raw_llamadaq() -> None:
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
        .snakemake_rules.tier_raw_llamadaq
    )

    log = build_log(config_dict, args.log, fallback=__name__)

    channel_dict = config_dict.inputs.out_spec

    chmap = TextDB(args.chan_maps, lazy=True).channelmaps.on(args.timestamp)

    config = Props.read_from(channel_dict)
    channels = chmap.map("daq.rawid")

    for rawid, chinfo in channels.items():
        cfg_block = deepcopy(config["LLAMAEventDecoder"]["__output_table_name__"])
        cfg_block["key_list"] = [chinfo.daq.struckid]
        config["LLAMAEventDecoder"][f"ch{rawid:03d}/raw"] = cfg_block

    config["LLAMAEventDecoder"].pop("__output_table_name__")

    msg = f"{config=}"
    log.debug(msg)

    build_raw(
        args.input, out_spec=config, filekey=args.output, in_stream_type="LlamaDaq"
    )
