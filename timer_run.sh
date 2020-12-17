#!/bin/bash

source /etc/profile
source ~/.bashrc

/home/jt1/miniconda3/envs/dol2/bin/python app.py --env prod --enable 31,32,34,35 --enable_forward_filter 31,32,34,35 --use_sm 31,32,34,35 --run_direct --send_msg
