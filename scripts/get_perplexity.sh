m=("unsloth/Llama-3.2-3B-Instruct" "unsloth/Llama-3.1-8B-Instruct" "Henrychur/MMed-Llama-3-8B"  "unsloth/gemma-3-4b-it" "unsloth/medgemma-4b-it" "unsloth/gemma-3-12b-it") 
m=("Henrychur/MMed-Llama-3-8B") 


for model_name in "${m[@]}"; do
    output_dir="outputs/ppl_sib"
    python -m mosaic.core.perplexity -d "sib" -m $model_name -o $output_dir --debug True
    output_dir="outputs/ppl_mosaic"
    python -m mosaic.core.perplexity -d "mosaic" -m $model_name -o $output_dir --debug True
done

