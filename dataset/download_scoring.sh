#!/bin/bash

download() {
  file=${1}
  url="https://zenodo.org/record/8091220/files/${file}?download=1"
  if [ ! -f tarfiles/${file} ]; then
    wget ${url} -O tarfiles/${file}
  fi
}

untar() {
  file=${1}
  dir=$(basename ${file} .tar.xz | rev | cut -d_ -f1 | rev)

  if [[ "$file" == *"PDBbind-v2020"* ]]; then
    tar_dir=${PWD}/PDBbind-v2020/${dir}
  else
    tar_dir=${PWD}/Benchmark/${dir}
  fi

  mkdir -p ${tar_dir}
  tar -xf ${file} -C ${tar_dir}

  if [ ! -d ${tar_dir}/data ]; then
    if [ -d ${tar_dir}/data_5_sdf ]; then
      ln -s ${tar_dir}/data_5_sdf ${tar_dir}/data
    else
      ln -s ${tar_dir}/data_5 ${tar_dir}/data
    fi
  fi
}

mkdir -p tarfiles
mkdir -p PDBbind-v2020 Benchmark

# Benchmark
download CASF-2016_scoring.tar.xz
untar tarfiles/CASF-2016_scoring.tar.xz

