python run.py --base_model_name gpt2-xl --n_samples 500 
python run.py --base_model_name gpt2-large --n_samples 500
python run.py --base_model_name gpt2-medium --n_samples 500
python run.py --base_model_name gpt2 --n_samples 500
python run.py --base_model_name EleutherAI/gpt-neo-2.7B --n_samples 500 --batch_size 15
python run.py --base_model_name EleutherAI/gpt-j-6B --n_samples 500 --batch_size 15
python run.py --base_model_name facebook/opt-2.7b --n_samples 500 --batch_size 20
python run.py --openai_model text-davinci-003 --n_samples 500 

python run.py --base_model_name gpt2-xl --n_samples 500  --dataset squad --dataset_key context
python run.py --base_model_name gpt2-large --n_samples 500  --dataset squad --dataset_key context
python run.py --base_model_name gpt2-medium --n_samples 500  --dataset squad --dataset_key context
python run.py --base_model_name gpt2 --n_samples 500  --dataset squad --dataset_key context
python run.py --base_model_name EleutherAI/gpt-neo-2.7B --n_samples 500  --dataset squad --dataset_key context --batch_size 15
python run.py --base_model_name EleutherAI/gpt-j-6B --n_samples 500  --dataset squad --dataset_key context --batch_size 15
python run.py --base_model_name facebook/opt-2.7b --n_samples 500  --dataset squad --dataset_key context --batch_size 15
python run.py --openai_model text-davinci-003 --n_samples 500 --dataset squad --dataset_key context

python run.py --base_model_name gpt2-xl --n_samples 500  --dataset writing
python run.py --base_model_name gpt2-large --n_samples 500  --dataset writing
python run.py --base_model_name gpt2-medium --n_samples 500  --dataset writing
python run.py --base_model_name gpt2 --n_samples 500  --dataset writing
python run.py --base_model_name EleutherAI/gpt-neo-2.7B --n_samples 500  --dataset writing --batch_size 15
python run.py --base_model_name EleutherAI/gpt-j-6B --n_samples 500  --dataset writing --batch_size 15
python run.py --base_model_name facebook/opt-2.7b --n_samples 500  --dataset writing --batch_size 15
python run.py --openai_model text-davinci-003 --n_samples 500 --dataset writing