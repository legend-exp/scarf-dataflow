from __future__ import annotations

import argparse
import json
import pickle as pkl
import re
import warnings
from pathlib import Path

import numpy as np
from dbetto import TextDB
from dbetto.catalog import Props
from lgdo.lh5 import ls
from pygama.pargen.data_cleaning import (
    generate_cut_classifiers,
    get_keys,
)
from pygama.pargen.utils import load_data

from .....log import build_log
from ....convert_np import convert_dict_np_to_float
from ....pulser_removal import get_pulser_mask
from ....table_name import get_table_name

warnings.filterwarnings(action="ignore", category=RuntimeWarning)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--cal-files", help="cal_files", nargs="*", type=str)
    argparser.add_argument("--fft-files", help="fft_files", nargs="*", type=str)

    argparser.add_argument(
        "--tcm-filelist", help="tcm_filelist", nargs="*", type=str, required=False
    )
    argparser.add_argument(
        "--pulser-files", help="pulser_file", nargs="*", type=str, required=False
    )
    argparser.add_argument(
        "--overwrite-files", help="overwrite_files", nargs="*", type=str, required=False
    )

    argparser.add_argument("--configs", help="config", type=str, required=True)
    argparser.add_argument("--metadata", help="metadata path", type=str, required=True)
    argparser.add_argument("--log", help="log_file", type=str)

    argparser.add_argument("--datatype", help="Datatype", type=str, required=True)
    argparser.add_argument("--timestamp", help="Timestamp", type=str, required=True)
    argparser.add_argument("--channel", help="Channel", type=str, required=True)

    argparser.add_argument(
        "--plot-path", help="plot_path", type=str, nargs="*", required=False
    )
    argparser.add_argument(
        "--save-path",
        help="save_path",
        type=str,
        nargs="*",
    )
    args = argparser.parse_args()

    configs = TextDB(args.configs, lazy=True).on(args.timestamp, system=args.datatype)
    config_dict = configs["snakemake_rules"]["pars_pht_qc"]

    log = build_log(config_dict, args.log)

    channel = get_table_name(args.metadata, args.timestamp, args.datatype, args.channel)

    # get metadata dictionary
    channel_dict = config_dict["inputs"]["qc_config"][args.channel]
    kwarg_dict = Props.read_from(channel_dict)

    # sort files in dictionary where keys are first timestamp from run
    if isinstance(args.cal_files, list):
        cal_files = []
        for file in args.cal_files:
            with Path(file).open() as f:
                cal_files += f.read().splitlines()
    else:
        with Path(args.cal_files).open() as f:
            cal_files = f.read().splitlines()

    cal_files = sorted(
        np.unique(cal_files)
    )  # need this as sometimes files get double counted as it somehow puts in the p%-* filelist and individual runs also

    if args.overwrite_files:
        overwrite = Props.read_from(args.overwrite_files)
        if channel in overwrite:
            overwrite = overwrite[channel]["pars"]["operations"]
        else:
            overwrite = None
    else:
        overwrite = None

    kwarg_dict_fft = kwarg_dict["fft_fields"]
    if len(args.fft_files) > 0:
        # sort files in dictionary where keys are first timestamp from run
        if isinstance(args.fft_files, list):
            fft_files = []
            for file in args.fft_files:
                with Path(file).open() as f:
                    fft_files += f.read().splitlines()
        else:
            with Path(args.fft_files).open() as f:
                fft_files = f.read().splitlines()

        fft_files = sorted(
            np.unique(fft_files)
        )  # need this as sometimes files get double counted as it somehow puts in the p%-* filelist and individual runs also

        if len(fft_files) > 0:
            fft_fields = get_keys(
                [
                    key.replace(f"{channel}/dsp/", "")
                    for key in ls(fft_files[0], f"{channel}/dsp/")
                ],
                kwarg_dict_fft["cut_parameters"],
            )

            fft_data = load_data(
                fft_files,
                f"{channel}/dsp",
                {},
                [*fft_fields, "timestamp", "trapTmax", "t_sat_lo"],
            )

            discharges = fft_data["t_sat_lo"] > 0
            discharge_timestamps = np.where(fft_data["timestamp"][discharges])[0]
            is_recovering = np.full(len(fft_data), False, dtype=bool)
            for tstamp in discharge_timestamps:
                is_recovering = is_recovering | np.where(
                    (
                        ((fft_data["timestamp"] - tstamp) <= 0.01)
                        & ((fft_data["timestamp"] - tstamp) > 0)
                    ),
                    True,
                    False,
                )
            fft_data["is_recovering"] = is_recovering

            hit_dict_fft = {}
            plot_dict_fft = {}
            cut_data = fft_data.query("is_recovering==0")
            log.debug(f"cut_data shape: {len(cut_data)}")
            for name, cut in kwarg_dict_fft["cut_parameters"].items():
                cut_dict, cut_plots = generate_cut_classifiers(
                    cut_data,
                    {name: cut},
                    kwarg_dict.get("rounding", 4),
                    display=1 if args.plot_path else 0,
                )
                hit_dict_fft.update(cut_dict)
                plot_dict_fft.update(cut_plots)

                log.debug(
                    f"{name} calculated cut_dict is: {json.dumps(cut_dict, indent=2)}"
                )

                ct_mask = np.full(len(cut_data), True, dtype=bool)
                for outname, info in cut_dict.items():
                    # convert to pandas eval
                    exp = info["expression"]
                    for key in info.get("parameters", None):
                        exp = re.sub(
                            f"(?<![a-zA-Z0-9]){key}(?![a-zA-Z0-9])", f"@{key}", exp
                        )
                    cut_data[outname] = cut_data.eval(
                        exp, local_dict=info.get("parameters", None)
                    )
                    if "_classifier" not in outname:
                        ct_mask = ct_mask & cut_data[outname]
                cut_data = cut_data[ct_mask]

            log.debug("fft cuts applied")
            log.debug(f"cut_dict is: {json.dumps(hit_dict_fft, indent=2)}")

        else:
            hit_dict_fft = {}
            plot_dict_fft = {}
    else:
        hit_dict_fft = {}
        plot_dict_fft = {}

    if overwrite is not None:
        for name in kwarg_dict_fft["cut_parameters"]:
            for cut_name, cut_dict in overwrite.items():
                if name in cut_name:
                    hit_dict_fft.update({cut_name: cut_dict})

    kwarg_dict_cal = kwarg_dict["cal_fields"]

    cut_fields = get_keys(
        [
            key.replace(f"{channel}/dsp/", "")
            for key in ls(cal_files[0], f"{channel}/dsp/")
        ],
        kwarg_dict_cal["cut_parameters"],
    )
    if "initial_cal_cuts" in kwarg_dict:
        init_cal = kwarg_dict["initial_cal_cuts"]
        cut_fields += get_keys(
            [
                key.replace(f"{channel}/dsp/", "")
                for key in ls(cal_files[0], f"{channel}/dsp/")
            ],
            init_cal["cut_parameters"],
        )

    # load data in
    data, threshold_mask = load_data(
        cal_files,
        f"{channel}/dsp",
        {},
        [*cut_fields, "timestamp", "trapTmax", "t_sat_lo"],
        threshold=kwarg_dict_cal.get("threshold", 0),
        return_selection_mask=True,
        cal_energy_param="trapTmax",
    )

    total_mask = get_pulser_mask(
        pulser_file=args.pulser_files,
        tcm_filelist=args.tcm_filelist,
        channel=channel,
        pulser_multiplicity_threshold=kwarg_dict.get("pulser_multiplicity_threshold"),
    )
    if "pulser_multiplicity_threshold" in kwarg_dict:
        kwarg_dict.pop("pulser_multiplicity_threshold")

    data["is_pulser"] = total_mask[threshold_mask]

    discharges = data["t_sat_lo"] > 0
    discharge_timestamps = np.where(data["timestamp"][discharges])[0]
    is_recovering = np.full(len(data), False, dtype=bool)
    for tstamp in discharge_timestamps:
        is_recovering = is_recovering | np.where(
            (
                ((data["timestamp"] - tstamp) <= 0.01)
                & ((data["timestamp"] - tstamp) > 0)
            ),
            True,
            False,
        )
    data["is_recovering"] = is_recovering

    rng = np.random.default_rng()
    mask = np.full(len(data.query("~is_pulser & ~is_recovering")), False, dtype=bool)
    mask[
        rng.choice(
            len(data.query("~is_pulser & ~is_recovering")),
            2000 * len(args.cal_files),
            replace=False,
        )
    ] = True

    if "initial_cal_cuts" in kwarg_dict:
        init_cal = kwarg_dict["initial_cal_cuts"]
        hit_dict_init_cal, plot_dict_init_cal = generate_cut_classifiers(
            data.query("~is_pulser&~is_recovering")[mask],
            init_cal["cut_parameters"],
            init_cal.get("rounding", 4),
            display=1 if args.plot_path else 0,
        )
        ct_mask = np.full(len(data), True, dtype=bool)
        for outname, info in hit_dict_init_cal.items():
            # convert to pandas eval
            exp = info["expression"]
            for key in info.get("parameters", None):
                exp = re.sub(f"(?<![a-zA-Z0-9]){key}(?![a-zA-Z0-9])", f"@{key}", exp)
            data[outname] = data.eval(exp, local_dict=info.get("parameters", None))
            if "classifier" not in outname:
                ct_mask = ct_mask & data[outname]

        mask = mask[ct_mask[(~data["is_pulser"] & ~data["is_recovering"]).to_numpy()]]
        data = data[ct_mask]
        log.debug("initial cal cuts applied")
        log.debug(f"cut_dict is: {json.dumps(hit_dict_init_cal, indent=2)}")

    else:
        hit_dict_init_cal = {}
        plot_dict_init_cal = {}

    data = data.query("~is_pulser & ~is_recovering")[mask]

    hit_dict_cal, plot_dict_cal = generate_cut_classifiers(
        data,
        kwarg_dict_cal["cut_parameters"],
        kwarg_dict.get("rounding", 4),
        display=1 if args.plot_path else 0,
    )

    log.debug("initial cuts applied")
    log.debug(f"cut_dict is: {json.dumps(hit_dict_cal, indent=2)}")

    if overwrite is not None:
        for name in kwarg_dict_cal["cut_parameters"]:
            for cut_name, cut_dict in overwrite.items():
                if name in cut_name:
                    hit_dict_cal.update({cut_name: cut_dict})

    hit_dict = {**hit_dict_fft, **hit_dict_init_cal, **hit_dict_cal}
    plot_dict = {**plot_dict_fft, **plot_dict_init_cal, **plot_dict_cal}

    hit_dict = convert_dict_np_to_float(hit_dict)

    for file in args.save_path:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        Props.write_to(file, hit_dict)

    if args.plot_path:
        for file in args.plot_path:
            Path(file).parent.mkdir(parents=True, exist_ok=True)
            with Path(file).open("wb") as f:
                pkl.dump({"qc": plot_dict}, f, protocol=pkl.HIGHEST_PROTOCOL)
