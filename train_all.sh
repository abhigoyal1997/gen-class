cuda=$1
seed=$2

ds=468
for ms in 47 94 140 187 234 281; do
  for model in g c; do
    python main.py -p $cuda -r $seed train -mc $model'.config'\
    -d data/ms-celebs/train$seed.csv -ds $ms\
    -s train_specs.txt models/$model$seed-$ms
  done

  mkdir -p models/gc$seed-$ds-$ms
  echo 'gc\n4 1\nmodels/g'$seed'-'$ms'\nmodels/c'$seed'-'$ms > models/gc$seed-$ds-$ms/config.txt
  python main.py -p $cuda -r $seed train -mc models/gc$seed-$ds-$ms/config.txt\
   -d data/ms-celebs/train$seed.csv -ds $ds -ms $ms\
   -s train_specs.txt models/gc$seed-$ds-$ms
done

# ds=500
# for ms in 50 150 250 350 450; do
#   mkdir -p models/gc$seed-$ds-$ms
#   echo 'gc\n4 1\nmodels/g'$seed'-'$ms'\nmodels/c'$seed'-'$ms > models/gc$seed-$ds-$ms/config.txt
#   python main.py -p $cuda -r $seed train -mc models/gc$seed-$ds-$ms/config.txt\
#    -d data/mnist/train.csv -ds $ds -ms $ms\
#    -s train_specs.txt models/gc$seed-$ds-$ms
# done

# ds=1000
# for ms in 100 200 300 400 500 600 700 800 900 1000; do
#   mkdir -p models/gc$seed-$ds-$ms
#   echo 'gc\n4 1\nmodels/g'$seed'-'$ms'\nmodels/c'$seed'-'$ms > models/gc$seed-$ds-$ms/config.txt
#   python main.py -p $cuda -r $seed train -mc models/gc$seed-$ds-$ms/config.txt\
#    -d data/mnist/train.csv -ds $ds -ms $ms\
#    -s train_specs.txt models/gc$seed-$ds-$ms
# done
