#!/bin/zsh

pollution_folder="20260329_160648"

rsync -avz /Users/jberndt/Documents/Masterarbeit/data-pollution/data/cleaned/*.csv pella:~/ma/ma-uncertainty-scripts/data/cleaned
rsync -avz /Users/jberndt/Documents/Masterarbeit/data-pollution/data/cleaned/completeness/*.json pella:~/ma/ma-uncertainty-scripts/data/cleaned/completeness
rsync -avz /Users/jberndt/Documents/Masterarbeit/data-pollution/data/polluted/${pollution_folder}/1.25p_EAR/completeness pella:~/ma/ma-uncertainty-scripts/data/polluted/${pollution_folder}/1.25p_EAR --mkpath


rsync -avz pella:~/ma/ma-uncertainty-scripts/data/polluted/${pollution_folder}/1.25p_EAR/results /Users/jberndt/Documents/Masterarbeit/data-pollution/data/polluted/${pollution_folder}/1.25p_EAR/completeness
rsync -avz pella:~/ma/ma-uncertainty-scripts/data/cleaned/completeness/results /Users/jberndt/Documents/Masterarbeit/data-pollution/data/cleaned/completeness
