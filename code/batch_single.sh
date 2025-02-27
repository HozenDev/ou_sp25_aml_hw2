DATASET=../../datasets/bmi_dataset.pkl

for i in $(seq 0 89)
do
    echo "Running experiment $i"
    
    python hw2.py \
           --hidden 200 100 50 25 12 6 \
           --lrate 0.001 \
           --output_type 'theta' \
           --predict_dim 1 \
           --epochs 1000 \
           --exp_type 'bmi' \
           --exp_index $i \
           --dataset $DATASET \
           --activation_out 'linear' \
           --activation_hidden 'elu' \
           --label "exp" \
           --nowandb \
           --early_stopping \
           &> results/hw2_${i}_stdout.txt
done
