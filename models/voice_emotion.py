"""SpeechMind voice-emotion inference with multi-corpus checkpoint preference."""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoModelForAudioClassification, AutoProcessor

from models.voice_features import POOLING_VERSION, EXPECTED_FEATURE_DIMENSION, extract_wavlm_embedding

BASE_MODEL_NAME="Dpngtm/wav2vec2-emotion-recognition"
MULTICORPUS_MODEL_DIR=Path("checkpoints/voice_emotion_multicorpus_cpu")
WAVLM_CPU_MODEL_DIR=Path("checkpoints/voice_emotion_wavlm_cpu")
FINETUNED_MODEL_DIR=Path("checkpoints/voice_emotion_wavlm")
SAMPLE_RATE=16000
WINDOW_SECONDS=8
WINDOW_STEP_SECONDS=6
EXPECTED_LABELS={"joy","sadness","anger","fear","surprise","disgust","neutral"}


def pool_hidden_states(hidden_states):
    values=np.asarray(hidden_states,dtype=np.float32)
    if values.ndim!=2 or values.shape[0]==0: raise ValueError("Expected non-empty [frames, hidden_size] states.")
    return np.concatenate([values.mean(0),values.std(0),np.quantile(values,.1,axis=0),np.quantile(values,.9,axis=0)]).astype(np.float32,copy=False)


class VoiceEmotionRecognizer:
    def __init__(self):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_metadata=None; self.embedding_model=None; self.feature_extractor=None
        self.svm_pipeline=None; self.model=None; self.processor=None; self.model_source=None
        options=[
            (MULTICORPUS_MODEL_DIR,"wavlm_multicorpus_feature_svc",POOLING_VERSION,EXPECTED_FEATURE_DIMENSION),
            (WAVLM_CPU_MODEL_DIR,"wavlm_frozen_embedding_svc","mean_std",1536),
        ]
        for directory,model_type,pooling,dimension in options:
            meta_path=directory/"speechmind_metadata.json"; head_path=directory/"voice_emotion_head.joblib"
            if not (meta_path.is_file() and head_path.is_file()): continue
            try:
                metadata=json.loads(meta_path.read_text(encoding="utf-8"))
                valid=(metadata.get("model_type")==model_type and metadata.get("backbone")=="microsoft/wavlm-base-plus"
                    and metadata.get("sample_rate")==SAMPLE_RATE and set(metadata.get("labels",[]))==EXPECTED_LABELS
                    and metadata.get("pooling")==pooling and metadata.get("feature_dimension")==dimension
                    and metadata.get("selected_kernel") in {"linear","rbf"})
                if not valid:
                    print(f"[VoiceEmotion] Ignoring incompatible checkpoint: {directory}")
                    continue
                import joblib
                self.feature_extractor=AutoFeatureExtractor.from_pretrained(metadata["backbone"])
                self.embedding_model=AutoModel.from_pretrained(metadata["backbone"]).to(self.device)
                self.embedding_model.eval(); self.svm_pipeline=joblib.load(head_path)
                self.model_metadata=metadata; self.model_source=str(directory); self.pooling_version=pooling
                print(f"[VoiceEmotion] Loaded {model_type} from {directory} on {self.device}.")
                return
            except (OSError,ValueError,TypeError,ImportError,RuntimeError) as error:
                print(f"[VoiceEmotion] Could not load {directory}: {error}")
        meta=FINETUNED_MODEL_DIR/"speechmind_metadata.json"; config=FINETUNED_MODEL_DIR/"config.json"
        if meta.is_file() and config.is_file():
            try:
                metadata=json.loads(meta.read_text(encoding="utf-8"))
                if metadata.get("model_type")=="wavlm_full_finetune" and set(metadata.get("labels",[]))==EXPECTED_LABELS and metadata.get("backbone")=="microsoft/wavlm-base-plus" and metadata.get("sample_rate")==SAMPLE_RATE:
                    self.model_source=str(FINETUNED_MODEL_DIR); self.model_metadata=metadata
                else: self.model_source=BASE_MODEL_NAME
            except (OSError,ValueError,TypeError): self.model_source=BASE_MODEL_NAME
        else: self.model_source=BASE_MODEL_NAME
        print(f"[VoiceEmotion] Loading baseline {self.model_source} on {self.device}...")
        self.processor=AutoProcessor.from_pretrained(self.model_source)
        self.model=AutoModelForAudioClassification.from_pretrained(self.model_source).to(self.device)
        self.model.eval()

    @staticmethod
    def _to_chunks(audio):
        window=WINDOW_SECONDS*SAMPLE_RATE; step=WINDOW_STEP_SECONDS*SAMPLE_RATE
        if len(audio)<=window: return [audio]
        chunks=[audio[start:start+window] for start in range(0,len(audio),step)]
        if len(chunks)>1 and len(chunks[-1])<SAMPLE_RATE: chunks.pop()
        return chunks or [audio]

    @torch.inference_mode()
    def _wavlm_probabilities(self,chunk):
        embedding=extract_wavlm_embedding(self.embedding_model,self.feature_extractor,chunk,self.device).reshape(1,-1)
        probabilities=self.svm_pipeline.predict_proba(embedding)[0]
        classes=self.svm_pipeline.named_steps["classifier"].classes_
        return {str(label):float(prob) for label,prob in zip(classes,probabilities)}

    @torch.inference_mode()
    def predict(self,audio):
        if isinstance(audio,torch.Tensor): audio=audio.detach().cpu().numpy()
        waveform=np.asarray(audio,dtype=np.float32).squeeze()
        if waveform.ndim!=1 or waveform.size==0: raise ValueError("Voice emotion model expects non-empty mono audio.")
        if not np.isfinite(waveform).all(): raise ValueError("Voice emotion model received NaN or infinite audio.")
        peak=float(np.max(np.abs(waveform)))
        if peak>1.0: waveform=waveform/peak
        chunks=self._to_chunks(waveform); weighted={}; total=0.0
        for chunk in chunks:
            if self.svm_pipeline is not None:
                distribution=self._wavlm_probabilities(chunk)
            else:
                inputs=self.processor(chunk,sampling_rate=SAMPLE_RATE,return_tensors="pt",padding=True)
                inputs={k:v.to(self.device) for k,v in inputs.items()}
                logits=self.model(**inputs).logits[0]
                probs=torch.softmax(logits,dim=-1).float().cpu().numpy(); ids=self.model.config.id2label
                distribution={str(ids.get(i,ids.get(str(i),f"label_{i}"))).lower():float(v) for i,v in enumerate(probs)}
            weight=float(len(chunk))
            for label,prob in distribution.items(): weighted[label]=weighted.get(label,0.)+prob*weight
            total+=weight
        distribution={k:v/max(total,1.) for k,v in weighted.items()}
        emotion=max(distribution,key=distribution.get)
        return {"emotion":emotion,"confidence":round(float(distribution[emotion]),3),
            "distribution":distribution,"arousal":self._estimate_arousal(distribution),
            "inference_device":str(self.device),"inference_windows":len(chunks),
            "model_source":self.model_source,"validation_macro_f1":self.model_metadata.get("validation_macro_f1") if self.model_metadata else None}

    @staticmethod
    def _estimate_arousal(distribution):
        high=sum(distribution.get(k,0.) for k in ("angry","anger","fear","fearful","happy","joy","surprise"))
        low=sum(distribution.get(k,0.) for k in ("sad","sadness","neutral","calm"))
        return "High" if high>=.6 else "Low" if low>=.6 else "Moderate"
