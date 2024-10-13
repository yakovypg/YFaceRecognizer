#!/bin/bash

test_faces_dir=TestFaces
known_faces_dir=KnownFaces

if [[ -d $test_faces_dir ]]; then
    curr_date=$(date +%Y%m%d%H%M%S)
    test_faces_dir="${test_faces_dir}${curr_date}"
fi

git clone https://github.com/mikeffendii/Celebrity-Face-Recognition
mv "Celebrity-Face-Recognition/Images Dataset" $test_faces_dir

if [[ ! -d $known_faces_dir ]]; then
    mkdir $known_faces_dir
    cp $test_faces_dir/lionel_messi/26622.jpg $known_faces_dir/Lionel_Messi.jpg
fi

rm -rf Celebrity-Face-Recognition
