ms=$1
cuda=$2
seed=$3

for model in g c; do
  python main.py -p $cuda -r $seed train -mc $model.config\
  -d data/ms-celebs/train$seed.csv -ds $ms\
  -s train_specs.txt models/$model$seed-$ms
done
