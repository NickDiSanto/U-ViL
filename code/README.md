# U-ViL
MIDL 2025 short paper.


Train: 
cd code

python train_fully_supervised_2D.py --model uvil --max_iterations 30000 --batch_size 4 --num_classes 4 --exp ACDC/Fully_Supervised


Test:
cd code
python test_2D_fully.py --model uvil --exp ACDC/Fully_Supervised