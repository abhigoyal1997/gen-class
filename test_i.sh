cuda=$1

ds=468
for seed in 0 1 2 3 4; do
  for ms in 47 94 140 187 234 281 328 374 421 468; do
    python main.py -p $cuda -r $seed test\
     -d data/ms-celebs/test$seed.csv\
     models/gc$seed-$ds-$ms -i
  done
done
