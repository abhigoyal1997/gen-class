ds=$1
ms=$2
cuda=$3
seed=$4

mkdir -p models/gc$seed-$ds-$ms
echo 'gc\n4 0\nmodels/g'$seed'-'$ms'\nmodels/c'$seed'-'$ms > models/gc$seed-$ds-$ms/config.txt
python main.py -p $cuda -r $seed train -mc models/gc$seed-$ds-$ms/config.txt\
 -d data/blood-cells/train/$seed.csv -dataset bc -ds $ds -ms $ms\
 -s train_specs.txt models/gc$seed-$ds-$ms
