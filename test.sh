cuda=$1

ds=500
for seed in 0 1 2 3 4; do
  for ms in 50 100 150 200 250 300 350 400 450 500; do
    python main.py -p $cuda -r $seed test\
     -d data/mnist/test.csv\
     models/gc$seed-500-$ms
  done
done
