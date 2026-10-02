# scarf-dataflow

Implementation of an automatic data processing flow for data from the SCARF
experiment at TUM, based on
[legend-dataflow](https://github.com/legend-exp/legend-dataflow).

## Quick start

```console
$ uv venv --python 3.12
$ source .venv/bin/activate
$ uv pip install '.[runprod]'
$ dataprod -v install -s sator dataflow-config.yaml
$ snakemake --workflow-profile workflow/profiles/sator-build-raw 'all-sp04-*-raw.gen'
```
