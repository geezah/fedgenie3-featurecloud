from os import environ
from pathlib import Path

import synapseclient
import synapseutils
from invoke import task
from invoke.context import Context


# 1. Download the DREAM4 data from Synapse
def _download_synapse(syn_client, syn_id, parent_dir):
    output_dir = Path(parent_dir) / syn_id
    synapseutils.syncFromSynapse(syn_client, entity=syn_id, path=output_dir)


@task
def download(c: Context, synapse_id: str, parent_dir: str = "local_data/raw"):
    syn = synapseclient.Synapse()
    syn.login(
        email=environ.get("SYNAPSE_EMAIL"),
        authToken=environ.get("SYNAPSE_AUTH"),
    )
    _download_synapse(syn, synapse_id, parent_dir)
