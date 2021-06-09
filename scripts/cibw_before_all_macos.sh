#!/bin/bash
set -e -x

ln -sf /usr/local/bin/gcc-10 /usr/local/bin/gcc

gcc --version
