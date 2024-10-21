#!/usr/bin/bash -l
#read -s -p 'Press enter to continue...'
# if the script was not launched from a terminal, restart it from a terminal
if [[ ! -t 0 && -x /usr/bin/x-terminal-emulator ]]; then
	echo "not started from terminal, starting again from terminal"
	sleep 3
	/usr/bin/x-terminal-emulator -e "bash -c \"$0 $*; read -s -p 'Press enter to continue...'\""
	exit
fi
#read -s -p 'Press enter to cd to dextra...'

cd ~/Dropbox/GitHub/SensorsINI/dextra-roshambo-python

#read -s -p 'Press enter to set up conda...'
source ~/anaconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh


#read -s -p 'Press enter to activate dextra conda env...'
conda activate dextra || conda activate ./.conda # activate the dextra env

#read -s -p 'Press enter to start dextra...'

python -m roshambo # run the demo
sleep 10
