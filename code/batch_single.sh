EXPERIMENT_TYPE='bmi'
DATASET='../../datasets/bmi_dataset.pkl'
NTRAINING_VALUES=(1 2 3 4 6 8 11 14 18)
ROTATION=(0 2 4 6 8 10 12 14 16 18)
NTRAINING_LENGTH=${#NTRAINING_VALUES[@]}
ROTATION_LENGTH=${#ROTATION[@]}

for i in "${ROTATION[@]}"
do
    for j in "${NTRAINING_VALUES[@]}"
    do
        echo "R:$i TV:$j"
        python hw2.py \
               --hidden 200 100 50 25 12 6 \
               --lrate 0.001 \
               --output_type dtheta \
               --predict_dim 1 \
               --epochs 1000 \
               --exp_type 'bmi' \
               --exp_index 0 \
               --dataset $DATASET \
               --Ntraining $j \
               --rotation $i \
               --activation_out 'linear' \
               --activation_hidden 'elu' \
               --label "exp" \
               --nowandb \
    done
done
