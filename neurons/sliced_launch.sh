export MODEL="crate-12L-tenth"

for LAYER in {0..5}
do
    export LAYER=$LAYER

    slice_id=0
    for i in {0..7}
    do
        if [ $i -ne 2 ]
        then
            CUDA_VISIBLE_DEVICES=$i nohup python sliced_eval.py --model $MODEL --layer $LAYER --slice_id $slice_id > nohup_sliced_eval.out &
            slice_id=$((slice_id+1))
            CUDA_VISIBLE_DEVICES=$i nohup python sliced_eval.py --model $MODEL --layer $LAYER --slice_id $slice_id > nohup_sliced_eval.out &
            slice_id=$((slice_id+1))
        fi
    done

    wait

    python merge_sliced_results.py --model $MODEL --layer $LAYER
done
