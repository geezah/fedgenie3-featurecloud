#!/bin/bash

#For DREAM4 (syn3049714)
DREAM5_SYNAPSE_ID=syn2787209
DREAM5_DIR=data/raw/

source credentials.conf
SYNAPSE_EMAIL=${SYNAPSE_EMAIL} SYNAPSE_AUTH=${SYNAPSE_AUTH} \
    invoke download ${DREAM5_SYNAPSE_ID} --parent-dir=${DREAM5_DIR}