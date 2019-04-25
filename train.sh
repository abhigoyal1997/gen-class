ds=$1
ms=$2
cuda=$3
seed=$4

# ./pretrain.sh $ms $cuda $seed
mkdir -p models/gc$seed-$ds-$ms
echo 'gc\n4 1\nmodels/g'$seed'-'$ms'\nmodels/c'$seed'-'$ms > models/gc$seed-$ds-$ms/config.txt
python main.py -p $cuda -r $seed train -mc models/gc$seed-$ds-$ms/config.txt\
 -d data/ms-celebs/train$seed.csv -ds $ds -ms $ms\
 -s train_specs.txt models/gc$seed-$ds-$ms
