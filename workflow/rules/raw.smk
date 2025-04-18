from legenddataflow.patterns import (
    get_pattern_tier_daq_unsorted,
    get_pattern_tier_oldllamadaq,
    get_pattern_tier_daq,
    get_pattern_tier,
    get_pattern_log,
    get_pattern_tier_raw_blind,
)
from legenddataflow.utils import set_last_rule_name
from legenddataflow.create_pars_keylist import ParsKeyResolve
from legenddataflow.execenv import execenv_pyexe

raw_par_catalog = ParsKeyResolve.get_par_catalog(
    ["-*-*-*-cal"],
    [
        get_pattern_tier_daq_unsorted(config, extension="*"),
        get_pattern_tier_oldllamadaq(config),
        get_pattern_tier_daq(config, extension="*"),
        get_pattern_tier(config, "raw", check_in_cycle=False),
    ],
    {"cal": ["par_raw"]},
)


rule build_raw_orca:
    """
    This rule runs build_raw, it takes in a file.orca and outputs a raw file
    """
    input:
        get_pattern_tier_daq(config, extension="orca"),
    params:
        timestamp="{timestamp}",
        datatype="{datatype}",
        ro_input=lambda _, input: ro(input),
    output:
        get_pattern_tier(config, "raw", check_in_cycle=check_in_cycle),
    log:
        get_pattern_log(config, "tier_raw", time),
    group:
        "tier-raw"
    resources:
        mem_swap=110,
        runtime=300,
    shell:
        execenv_pyexe(config, "build-tier-raw-orca") + "--log {log} "
        f"--configs {ro(configs)} "
        f"--chan-maps {ro(chan_maps)} "
        "--datatype {params.datatype} "
        "--timestamp {params.timestamp} "
        "{params.ro_input} {output}"


rule build_raw_fcio:
    """
    This rule runs build_raw, it takes in a file.fcio and outputs a raw file
    """
    input:
        get_pattern_tier_daq(config, extension="fcio"),
    params:
        timestamp="{timestamp}",
        datatype="{datatype}",
        ro_input=lambda _, input: ro(input),
    output:
        get_pattern_tier(config, "raw", check_in_cycle=check_in_cycle),
    log:
        get_pattern_log(config, "tier_raw", time),
    group:
        "tier-raw"
    resources:
        mem_swap=110,
        runtime=300,
    shell:
        execenv_pyexe(config, "build-tier-raw-fcio") + "--log {log} "
        f"--configs {ro(configs)} "
        f"--chan-maps {ro(chan_maps)} "
        "--datatype {params.datatype} "
        "--timestamp {params.timestamp} "
        "{params.ro_input} {output}"


rule build_raw_llamadaq:
    """
    This rule runs build_raw, it takes in a file.llamadaq and outputs a raw file
    """
    input:
        get_pattern_tier_daq(config, extension="llamadaq"),
    params:
        timestamp="{timestamp}",
        datatype="{datatype}",
        ro_input=lambda _, input: ro(input),
    output:
        get_pattern_tier(config, "raw", check_in_cycle=check_in_cycle),
    log:
        get_pattern_log(config, "tier_raw", time),
    group:
        "tier-raw"
    resources:
        mem_swap=110,
        runtime=300,
    shell:
        execenv_pyexe(config, "build-tier-raw-llamadaq") + "--log {log} "
        f"--configs {ro(configs)} "
        f"--chan-maps {ro(chan_maps)} "
        "--datatype {params.datatype} "
        "--timestamp {params.timestamp} "
        "{params.ro_input} {output}"


rule build_raw_mgdo:
    """
    This rule runs build_raw, it takes in a file.mgdo and outputs a raw file
    """
    input:
        get_pattern_tier_daq(config, extension="mgdo"),
    params:
        timestamp="{timestamp}",
        datatype="{datatype}",
        ro_input=lambda _, input: ro(input),
    output:
        get_pattern_tier(config, "raw", check_in_cycle=check_in_cycle),
    log:
        get_pattern_log(config, "tier_raw", time),
    group:
        "tier-raw"
    shell:
        execenv_pyexe(config, "build-tier-raw-mgdo") + "--log {log} "
        f"--configs {ro(configs)} "
        f"--chan-maps {ro(chan_maps)} "
        "--datatype {params.datatype} "
        "--timestamp {params.timestamp} "
        "{params.ro_input} {output}"


rule build_raw_blind:
    """
    This rule runs the data blinding, it takes in the raw file, calibration curve stored in the overrides
    and runs only if the blinding check file is on disk. Output is just the blinded raw file.
    """
    input:
        tier_file=str(get_pattern_tier(config, "raw", check_in_cycle=False)).replace(
            "{datatype}", "phy"
        ),
        blind_file=lambda wildcards: get_blinding_check_file(wildcards, raw_par_catalog),
    params:
        timestamp="{timestamp}",
        datatype="phy",
        ro_input=lambda _, input: {k: ro(v) for k, v in input.items()},
    output:
        get_pattern_tier_raw_blind(config),
    log:
        str(get_pattern_log(config, "tier_raw_blind", time)).replace(
            "{datatype}", "phy"
        ),
    group:
        "tier-raw"
    resources:
        mem_swap=110,
        runtime=300,
    shell:
        execenv_pyexe(config, "build-tier-raw-blind") + "--log {log} "
        f"--configs {ro(configs)} "
        f"--chan-maps {ro(chan_maps)} "
        f"--metadata {ro(meta)} "
        "--datatype {params.datatype} "
        "--timestamp {params.timestamp} "
        "--blind-curve {params.ro_input[blind_file]} "
        "--input {params.ro_input[tier_file]} "
        "--output {output}"


rule build_daq_mgdo:
    input:
        get_pattern_tier_oldllamadaq(config),
    params:
        timestamp="{timestamp}",
        datatype="{datatype}",
        ro_input=lambda _, input: ro(input),
    output:
        temp(get_pattern_tier_daq(config, extension="mgdo")),
    log:
        get_pattern_log(config, "tier_daq", time),
    group:
        "tier-daq"
    shell:
        "apptainer exec --cleanenv "
        "/mnt/atlas01/projects/scarf/software/containers/gerda-sw-all_v7.0.0.sif "
        "Raw2MGDO -m 1000 -c Struck -f {output} {input} &> {log}"
