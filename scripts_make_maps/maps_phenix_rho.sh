#!/bin/sh
PHASE_model=../swissfel_combined_dark_frac_0.01_0.19_ref.pdb

LIGHTMTZ=$1
DARKMTZ=$2
RESO=$3

LIGHT=$(basename "${LIGHTMTZ}" .mtz)
DARK=$(basename "${DARKMTZ}" .mtz)

mkdir ${LIGHT}_${DARK}_${RESO}_temp

cp ${LIGHT}.mtz  ${LIGHT}_${DARK}_${RESO}_temp/light-I.mtz
cp ${DARK}.mtz  ${LIGHT}_${DARK}_${RESO}_temp/dark-I.mtz 

pushd ${LIGHT}_${DARK}_${RESO}_temp
phenix.french_wilson dark-I.mtz 'output_file='dark-F.mtz
phenix.french_wilson light-I.mtz 'output_file='light-F.mtz

phenix.fobs_minus_fobs_map f_obs_1_file=light-F.mtz f_obs_2_file=dark-F.mtz f_obs_1_label=F,SIGF f_obs_2_label=F,SIGF phase_source=${PHASE_model} high_res=${RESO} sigma_cutoff=1 low_res=12.0 multiscale=True output_file=FoFoPHFc_Phenix.mtz

fft hklin FoFoPHFc_Phenix.mtz mapout light--dark.ccp4<<EOF
LABIN F1=FoFo PHI=PHFc
RESOLUTION 12.0 ${RESO}
#EXCLUDE sig1 1.0 
#EXCLUDE sig2 1.0
#GRID SAMPLE 4
SYMMETRY P22121
GRID SAMPLE 8
END
EOF

popd

cp ${LIGHT}_${DARK}_${RESO}_temp/light--dark.ccp4 ${RESO}_${LIGHT}_light--dark_${DARK}.ccp4
rm -r ${LIGHT}_${DARK}_${RESO}_temp

