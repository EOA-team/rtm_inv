source /mnt/ides/Lukas/venvs/GeoPython/bin/activate

export DB_HOST=id-hdb-psgr-ct5.ethz.ch
export DB_NAME=cs_sat
export DB_PW="KM>k5axMa?KH*:A*"
export DB_USER=cs_sat_admin
export USE_STAC=False

python /home/graflu/git/rtm_inv/scripts/invert_s2_scenes.py
