# change this path to where you clone the repo
# define $RRC_WORK_DIR

# change if you want to use in-context prompts
zeroshot=off # can be zeroshot, fewshot or off
wandb_project="mosaic_b"

# ---
models_dir=$RRC_WORK_DIR"/output"
evals_dir=$RRC_WORK_DIR"/evals"

# scripts
train_path=$RRC_WORK_DIR"/src/finetune.py"
infer_path=$RRC_WORK_DIR"/src/inference.py"

# arguments 
model_tags=("medgemma-4b")

train_datasets=('bd5')
valid_datasets=('bd5')
test_datasets=('bd5')

model_tags="${model_tags[@]}"
train_datasets="${train_datasets[@]}"
valid_datasets="${valid_datasets[@]}"
test_datasets="${test_datasets[@]}"

# Run the main scripts
train_flag=true
test_flag=true
special_tag="" # for debug purposes

for model_tag in $model_tags; do
    if $train_flag; then
        python3 $train_path -m $model_tag -ct "b" -p $wandb_project \
            -tds "$train_datasets" -vds "$valid_datasets" \
            #-et $special_tag # -c $checkpoint
    fi
    if $test_flag; then
        python3 $infer_path -m $model_tag \
            -zs $zeroshot \
            -trds "$train_datasets" -p $wandb_project \
            -d $models_dir -o $evals_dir \
            -teds "$test_datasets" \
            #-et $special_tag            
    fi
done
