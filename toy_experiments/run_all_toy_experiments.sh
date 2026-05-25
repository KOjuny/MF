echo "==> Dataset: 8gaussians"
python train_flowmatching.py \
  --dataset 8gaussians \
  --steps 20000 \
  --batch-size 512 \
  --save-dir ./results/models

python train_rectifiedflow.py \
  --dataset 8gaussians \
  --steps 20000 \
  --batch-size 512 \
  --save-dir ./results/models

python train_meanflow.py \
  --dataset 8gaussians \
  --steps 20000 \
  --batch-size 512 \
  --save-dir ./results/models

python visualization_sampling.py \
  --dataset 8gaussians \
  --model-dir ./results/models \
  --save-dir ./results/figures/8gaussians \
  --seed 0 \
  --gt-coupling optimal \
  --no-show

echo "==> Dataset: swissroll"
python train_flowmatching.py \
  --dataset swissroll \
  --steps 20000 \
  --batch-size 512 \
  --save-dir ./results/models

python train_rectifiedflow.py \
  --dataset swissroll \
  --steps 20000 \
  --batch-size 512 \
  --save-dir ./results/models

python train_meanflow.py \
  --dataset swissroll \
  --steps 20000 \
  --batch-size 512 \
  --save-dir ./results/models

python visualization_sampling.py \
  --dataset swissroll \
  --model-dir ./results/models \
  --save-dir ./results/figures/swissroll \
  --seed 0 \
  --gt-coupling optimal \
  --no-show

echo "==> Dataset: twomoons"
python train_flowmatching.py \
  --dataset twomoons \
  --steps 20000 \
  --batch-size 512 \
  --save-dir ./results/models

python train_rectifiedflow.py \
  --dataset twomoons \
  --steps 20000 \
  --batch-size 512 \
  --save-dir ./results/models

python train_meanflow.py \
  --dataset twomoons \
  --steps 20000 \
  --batch-size 512 \
  --save-dir ./results/models

python visualization_sampling.py \
  --dataset twomoons \
  --model-dir ./results/models \
  --save-dir ./results/figures/twomoons \
  --seed 0 \
  --gt-coupling optimal \
  --no-show
