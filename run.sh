python train.py  --num_epochs 50 \
        --learning_rate 2.5e-2 \
        --data ../data/ \
        --num_classes 19 \
        --cuda 0 \
		--validation_step 10 \
		--num_workers 8 \
        --batch_size 32 \
        --saved_models_path ./checkpoints_101_sgd \
		--model_file_name my_first_bisenet.torch \
        --context_path resnet18 \
        --optimizer sgd

