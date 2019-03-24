ds=$1
ms=$2
cuda=$3

for seed in 0 1 2 3 4; do
  python main.py -p $cuda test -d data/test/$seed.csv models/gc-$seed-$ds-$ms
done
