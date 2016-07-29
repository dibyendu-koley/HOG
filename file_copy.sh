#!/bin/bash
for file in $(ls -p $1 | grep -v / | tail -100)
do
#mv $file /other/locatio
ls $1$file
done
