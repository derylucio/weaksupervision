#!/bin/bash

/usr/local/bin/anaconda << EOF
. /u/at/$USER/.bash_profile
source activate virtualpy
mkdir -p /tmp/$USER
cd /tmp/$USER
echo $PWD
export THEANO_FLAGS="base_compiledir=${PWD}/BatchCompileDir/"
${1}
ls
cp -r /tmp/$USER/model*.h5 $PWD/output
cd /tmp
rm -r /tmp/$USER
EOF
