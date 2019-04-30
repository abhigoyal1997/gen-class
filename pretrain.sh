ms=$1
cuda=$2
seed=$3

for model in g c; do
  python main.py -p $cuda -r $seed train -mc $model'.config'\
  -d data/blood-cells/train/$seed.csv -dataset bc -ds $ms\
  -s train_specs.txt models/$model$seed-$ms
done
