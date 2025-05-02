#HuBERT Embedding extraction

#imports
import os
import numpy as np
import tqdm
import datasets
from datasets import DatasetDict, Features
from datasets import Dataset
from transformers import AutoProcessor, HubertModel, UniSpeechSatForXVector, Wav2Vec2FeatureExtractor, UniSpeechSatModel, HubertForSequenceClassification, Wav2Vec2Model, WavLMModel
from datasets import load_dataset
import soundfile as sf
import librosa
import torch
import pickle
import argparse
import torchaudio
from torchaudio.models.wav2vec2.utils import import_fairseq_model
import fairseq




#args

parser = argparse.ArgumentParser(description='Specify arguments for pre-trained transformer embedding extraction')

parser.add_argument('-d','--dataset', help='The dataset [all, 25_10, noise]',required=True)
parser.add_argument('-m','--model', help='The pre-trained model name [bio, ...]',required=True)
parser.add_argument('-l','--layer', help='the layer to extract [1:25]', type=int, required=True)

args = parser.parse_args()



if args.model == "bio" :
    #MODEL_DIR = "mount/Gibbon_ID_probing/saved_models/wavlm-large"
    #processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIR)
    #model = WavLMModel.from_pretrained(MODEL_DIR)
    # Load model using fairseq
    model_file = 'mount/Gibbon_ID_probing/saved_models/aves/aves-base-bio.pt'
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
    original = models[0]
    model = import_fairseq_model(original)

elif args.model == "nonbio" :
    # Load model using fairseq
    model_file = 'mount/Gibbon_ID_probing/saved_models/aves/aves-base-nonbio.pt'
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
    original = models[0]
    model = import_fairseq_model(original)

elif args.model == "core" :
    # Load model using fairseq
    model_file = 'mount/Gibbon_ID_probing/saved_models/aves/aves-base-core.pt'
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
    original = models[0]
    model = import_fairseq_model(original)

elif args.model == "all" :
    # Load model using fairseq
    model_file = 'mount/Gibbon_ID_probing/saved_models/aves/aves-base-all.pt'
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
    print(models)
    original = models[0]
    model = import_fairseq_model(original)

source_path = "mount/Gibbon_ID_probing"
sound_example = "audio_data/BalancedSelected_25max_10classes/test/DK_09_031.01.wav"

def sound2embedding(filename, model, layer_num=25) :
    ''' 
    Extracts model's embeddings from a sound file
    returns : (label_name, mean pooled embedding)
    '''
    #extract label name
    label = filename.split("/")[-1]
    label = label.split("_")[:2]
    label = "_".join(label)
    #read sound and downsample to 16kHz
    #sound, sr = librosa.load(os.path.join(source_path, filename), sr=16000)
    waveform, sr = torchaudio.load(os.path.join(source_path, filename))
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    #give sound array to pre-trained model and extract last_hidden_state
    #input_value = processor(sound, return_tensors="pt", sampling_rate=16000, output_hidden_states=True).input_values
    #hidden_state = model(input_value, output_hidden_states=True).hidden_states[layer_num]
    features, _ = model.extract_features(waveform)
    hidden_state = features[layer_num-1]
    #mean pool to get a single embedding
    mean_embedding = torch.mean(hidden_state, dim=1)
    return (label, mean_embedding)

#print(sound2embedding(sound_example, processor, model, 1))

#create a pickled version of collection (necessary to avoid overloading RAM + to save embeddings once)

def pickle_embeddings(sound_dir, emb_dir, model) :
    for sound_file in tqdm.tqdm(os.listdir(os.path.join(source_path, sound_dir))):
        if not sound_file.startswith('.') :
            label, embedding = sound2embedding(os.path.join(sound_dir, sound_file), model, layer_num)
            if os.path.join(str(label+".pkl")) in os.listdir(emb_dir) :
                with open(os.path.join(emb_dir, str(label+".pkl")), 'rb') as f:
                    emb = pickle.load(f)
                    emb[label].append(embedding)
                with open(os.path.join(emb_dir, str(label+".pkl")), 'wb') as f:
                    pickle.dump(emb, f)
            else :
                emb = {label:[embedding]}
                with open(os.path.join(emb_dir, str(label+".pkl")), 'wb') as f:
                    pickle.dump(emb, f)

if args.dataset == "25_10" :
    path = "audio_data/BalancedSelected_25max_10classes/"
elif args.dataset == "noise" :
    path = "audio_data/noise/"
elif args.dataset == "10_50" :
    path = "audio_data/BalancedSelected_10max_50classes/"
elif args.dataset == "pulse" :
    path = "audio_data/pulse/"
elif args.dataset == "pulse_stack" :
    path = "audio_data/pulse_stack/"
elif args.dataset == "denoised" :
    path = "audio_data/denoised/"
layer_num = args.layer




#train
sound_dir = os.path.join(path, "train/")
print("sound_dir : ", sound_dir)
emb_dir = os.path.join(source_path, f"embeddings/aves_{args.model}/layer_{args.layer}/embedding_pickles_{args.dataset}/train/")
print("emb_dir", emb_dir)
pickle_embeddings(sound_dir, emb_dir, model)

if not args.dataset == "10_50" :
#test
    sound_dir = os.path.join(path, "test/")
    emb_dir = os.path.join(source_path, f"embeddings/aves_{args.model}/layer_{args.layer}/embedding_pickles_{args.dataset}/test/")
    pickle_embeddings(sound_dir, emb_dir, model)
