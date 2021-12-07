# two source markets
python train_baseline.py --tgt_market t1 --src_markets s1-s2 --tgt_market_valid DATA/t1/valid_run.tsv --tgt_market_test DATA/t1/test_run.tsv --exp_name toytest --num_epoch 5 --cuda
# zero source market(only train on the target data)
conda activate xmrec
CUDA_VISIBLE_DEVICES=0 python train_baseline.py --tgt_market t1 --src_markets none --tgt_market_valid DATA/t1/valid_run.tsv --tgt_market_test DATA/t1/test_run.tsv --exp_name toytest --num_epoch 100 --cuda
CUDA_VISIBLE_DEVICES=1 python train_baseline.py --tgt_market t2 --src_markets none --tgt_market_valid DATA/t2/valid_run.tsv --tgt_market_test DATA/t2/test_run.tsv --exp_name toytest --num_epoch 100 --cuda