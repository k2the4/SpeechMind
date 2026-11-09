"""Multi-corpus CPU-friendly WavLM emotion experiment.

Supports RAVDESS (7 labels) and CREMA-D (6 labels, no surprise). It keeps
speakers disjoint per corpus, uses corpus-balanced/class-balanced SVM training,
selects on validation macro-F1, and reports held-out test metrics per corpus.
Frozen WavLM feature training; not WavLM fine-tuning.
"""
import argparse, csv, hashlib, json, os, random, sys, time
from collections import Counter, defaultdict
from pathlib import Path

# When executed as `python scripts/train_voice_emotion.py`, Python puts the
# scripts directory (not the repository root) at sys.path[0]. Add the project
# root so imports such as `models.voice_features` and `utils.audio_utils` work.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from models.voice_features import POOLING_VERSION, EXPECTED_FEATURE_DIMENSION, extract_wavlm_embedding

LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]
RAVDESS_TO_LABEL = {"01":"neutral","02":"neutral","03":"joy","04":"sadness","05":"anger","06":"fear","07":"disgust","08":"surprise"}
CREMA_D_TO_LABEL = {"ANG":"anger","DIS":"disgust","FEA":"fear","HAP":"joy","NEU":"neutral","SAD":"sadness"}
DEFAULT_DATA_DIR = os.environ.get("SPEECHMIND_RAVDESS_DIR", str(Path(__file__).resolve().parents[1] / "data" / "ravdess"))
DEFAULT_CREMA_D_DIR = os.environ.get("SPEECHMIND_CREMA_D_DIR", str(Path(__file__).resolve().parents[1] / "data" / "crema_d" / "AudioWAV"))
BASE_MODEL_NAME = "microsoft/wavlm-base-plus"
SAMPLE_RATE = 16000
PREPROCESSING_VERSION = "shared_audio_utils_mono16_trim25_peak_normalize_v2"


def parse_ravdess_filename(path):
    parts = Path(path).stem.split("-")
    if len(parts) != 7 or not all(part.isdigit() for part in parts):
        return None
    modality, channel, emotion, intensity, statement, repetition, actor = parts
    if modality != "03" or channel != "01" or emotion not in RAVDESS_TO_LABEL or not 1 <= int(actor) <= 24:
        return None
    return {"path": str(Path(path).resolve()), "dataset":"ravdess",
            "label":RAVDESS_TO_LABEL[emotion], "speaker_id":f"ravdess_actor_{int(actor):02d}",
            "source_emotion_code":emotion}


def parse_crema_d_filename(path):
    parts = Path(path).stem.split("_")
    if len(parts) != 4 or not parts[0].isdigit():
        return None
    actor, sentence, emotion, intensity = parts
    if emotion not in CREMA_D_TO_LABEL:
        return None
    return {"path":str(Path(path).resolve()), "dataset":"crema_d",
            "label":CREMA_D_TO_LABEL[emotion], "speaker_id":f"crema_d_actor_{actor}",
            "source_emotion_code":emotion}


def build_manifest(ravdess_dir, crema_d_dir=None):
    records = []
    root = Path(ravdess_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"RAVDESS folder not found: {root}")
    for path in sorted(root.rglob("*.wav")):
        row = parse_ravdess_filename(path)
        if row: records.append(row)
    if crema_d_dir:
        crema = Path(crema_d_dir).expanduser()
        if crema.is_dir():
            count = 0
            for path in sorted(crema.rglob("*.wav")):
                row = parse_crema_d_filename(path)
                if row:
                    records.append(row); count += 1
            if not count: print(f"[Dataset] No valid CREMA-D WAV files found in {crema}")
        else:
            print(f"[Dataset] CREMA-D not installed; using RAVDESS only: {crema}")
    if not records: raise ValueError("No valid RAVDESS or CREMA-D files found.")
    return records


def assign_speaker_splits(records, seed=42):
    by_dataset = defaultdict(set)
    for row in records: by_dataset[row.get("dataset","unknown")].add(row["speaker_id"])
    assignments = {}
    for dataset, group in sorted(by_dataset.items()):
        actors = sorted(group)
        if len(actors) < 5:
            raise ValueError(f"{dataset} has {len(actors)} speakers; at least 5 required.")
        random.Random(seed + sum(ord(c) for c in dataset)).shuffle(actors)
        n_test, n_val = max(1, round(.2*len(actors))), max(1, round(.2*len(actors)))
        test, val = set(actors[:n_test]), set(actors[n_test:n_test+n_val])
        for actor in actors:
            assignments[(dataset,actor)] = "test" if actor in test else "validation" if actor in val else "train"
    return [{**row,"dataset":row.get("dataset","unknown"),
             "split":assignments[(row.get("dataset","unknown"),row["speaker_id"])]} for row in records]


def summarize(records):
    result={}
    for split in ("train","validation","test"):
        rows=[r for r in records if r.get("split")==split]
        result[split]={"files":len(rows),"speakers":sorted({r["speaker_id"] for r in rows}),
            "datasets":dict(sorted(Counter(r["dataset"] for r in rows).items())),
            "labels":dict(sorted(Counter(r["label"] for r in rows).items()))}
    return result


def write_manifest(records,destination):
    destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    fields=["path","dataset","label","speaker_id","source_emotion_code","split"]
    with destination.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(records)
    return destination


def _cache_fingerprint(records):
    digest=hashlib.sha256()
    for item in (BASE_MODEL_NAME,POOLING_VERSION,PREPROCESSING_VERSION,str(SAMPLE_RATE)):
        digest.update(item.encode())
    for row in records:
        path=Path(row["path"]); stat=path.stat()
        for item in (str(path),str(stat.st_size),str(stat.st_mtime_ns),row["dataset"],row["label"],row["speaker_id"],row["split"]):
            digest.update(item.encode("utf-8",errors="replace"))
    return digest.hexdigest()


def extract_or_load_features(records,cache_path,force_rebuild=False):
    import torch
    from transformers import AutoFeatureExtractor,AutoModel
    from utils.audio_utils import preprocess
    cache=Path(cache_path); cache.parent.mkdir(parents=True,exist_ok=True)
    fingerprint=_cache_fingerprint(records)
    if cache.is_file() and not force_rebuild:
        try:
            with np.load(cache,allow_pickle=False) as saved:
                if str(saved["fingerprint"].item())==fingerprint:
                    arrays={k:saved[k] for k in ("features","labels","speakers","datasets","splits")}
                    if len(arrays["features"])==len(records):
                        print(f"[Features] Reusing cache: {cache}")
                        return (arrays["features"].astype(np.float32,copy=False),
                                arrays["labels"].astype(str),arrays["speakers"].astype(str),
                                arrays["datasets"].astype(str),arrays["splits"].astype(str))
        except (OSError,ValueError,KeyError): print("[Features] Cache incompatible; rebuilding.")
    torch.set_num_threads(max(1,min(4,os.cpu_count() or 2)))
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass
    device=torch.device("cpu")
    print(f"[Features] Loading frozen {BASE_MODEL_NAME} on CPU...")
    extractor=AutoFeatureExtractor.from_pretrained(BASE_MODEL_NAME)
    encoder=AutoModel.from_pretrained(BASE_MODEL_NAME).to(device); encoder.eval()
    vectors=[]; start=time.perf_counter(); total=len(records)
    for i,row in enumerate(records,1):
        waveform=preprocess(row["path"])
        vectors.append(extract_wavlm_embedding(encoder,extractor,waveform,device))
        if i==1 or i%25==0 or i==total:
            elapsed=time.perf_counter()-start; rate=i/max(elapsed,1e-6)
            print(f"[Features] {i}/{total} | {rate:.2f} clips/sec | ETA {(total-i)/max(rate,1e-6)/60:.1f} min")
    features=np.stack(vectors).astype(np.float32,copy=False)
    labels=np.asarray([r["label"] for r in records],dtype=str)
    speakers=np.asarray([r["speaker_id"] for r in records],dtype=str)
    datasets=np.asarray([r["dataset"] for r in records],dtype=str)
    splits=np.asarray([r["split"] for r in records],dtype=str)
    np.savez_compressed(cache,fingerprint=np.asarray(fingerprint),features=features,
        labels=labels,speakers=speakers,datasets=datasets,splits=splits)
    del encoder
    print(f"[Features] Cached {len(features)} vectors, dimension={features.shape[1]}")
    return features,labels,speakers,datasets,splits


def _corpus_weights(dataset_ids):
    counts=Counter(dataset_ids.tolist())
    n_corpora=max(1,len(counts))
    return np.asarray([len(dataset_ids)/(n_corpora*counts[d]) for d in dataset_ids],dtype=np.float64)


def train_classifier(features,labels,speakers,datasets,splits,output_dir,model_dir):
    import joblib
    from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    tr,va,te=(splits=="train"),(splits=="validation"),(splits=="test")
    if not tr.any() or not va.any() or not te.any(): raise ValueError("All data splits must be non-empty.")
    speaker_groups={s:set(speakers[splits==s].tolist()) for s in ("train","validation","test")}
    if speaker_groups["train"]&speaker_groups["validation"] or speaker_groups["train"]&speaker_groups["test"] or speaker_groups["validation"]&speaker_groups["test"]:
        raise ValueError("Speaker leakage detected.")
    Xtr,ytr,Xv,yv,Xte,yte=features[tr],labels[tr],features[va],labels[va],features[te],labels[te]
    weights=_corpus_weights(datasets[tr]); weights/=weights.mean()
    # Four targeted settings keep CPU fitting practical; the prior best
    # linear C=0.1 remains in the candidate set.
    candidate_specs = [
        ("linear", 0.03),
        ("linear", 0.1),
        ("linear", 0.3),
        ("rbf", 1.0),
    ]
    candidates=[]
    for kernel, c in candidate_specs:
        model=Pipeline([("scale",StandardScaler()),("classifier",SVC(
            kernel=kernel,C=c,gamma="scale",class_weight="balanced",probability=True,random_state=42))])
        model.fit(Xtr,ytr,classifier__sample_weight=weights)
        pred=model.predict(Xv)
        score=float(f1_score(yv,pred,labels=LABELS,average="macro",zero_division=0))
        acc=float(accuracy_score(yv,pred))
        candidates.append({"model":model,"kernel":kernel,"C":c,"validation_macro_f1":score,"validation_accuracy":acc})
        print(f"[Classifier] {kernel} C={c:g} validation macro-F1={score:.4f}, accuracy={acc:.4f}")
    best=max(candidates,key=lambda x:(x["validation_macro_f1"],x["validation_accuracy"],-x["C"]))
    pred=best["model"].predict(Xte)
    cm=confusion_matrix(yte,pred,labels=LABELS)
    report=classification_report(yte,pred,labels=LABELS,target_names=LABELS,output_dict=True,zero_division=0)
    per_dataset={}
    for dataset in sorted(set(datasets[te].tolist())):
        mask=datasets[te]==dataset
        per_dataset[dataset]={"files":int(mask.sum()),"accuracy":float(accuracy_score(yte[mask],pred[mask])),
            "macro_f1":float(f1_score(yte[mask],pred[mask],labels=LABELS,average="macro",zero_division=0)),
            "classification_report":classification_report(yte[mask],pred[mask],labels=LABELS,target_names=LABELS,output_dict=True,zero_division=0)}
    model_dir=Path(model_dir); output=Path(output_dir); model_dir.mkdir(parents=True,exist_ok=True); output.mkdir(parents=True,exist_ok=True)
    joblib.dump(best["model"],model_dir/"voice_emotion_head.joblib",compress=3)
    metadata={"model_type":"wavlm_multicorpus_feature_svc","backbone":BASE_MODEL_NAME,"labels":LABELS,
        "class_order":list(best["model"].named_steps["classifier"].classes_),"sample_rate":SAMPLE_RATE,
        "pooling":POOLING_VERSION,"preprocessing":PREPROCESSING_VERSION,"feature_dimension":int(features.shape[1]),
        "selected_kernel":best["kernel"],"selected_C":best["C"],"validation_macro_f1":best["validation_macro_f1"],
        "validation_accuracy":best["validation_accuracy"],"datasets":dict(sorted(Counter(datasets.tolist()).items())),
        "test_speakers":sorted(speaker_groups["test"]),"training_speakers":sorted(speaker_groups["train"]),
        "validation_speakers":sorted(speaker_groups["validation"]),
        "training_method":"Frozen WavLM top-four-layer mean with mean/std/q10/q90 pooling; corpus-balanced class-balanced SVM."}
    report_data={"accuracy":float(accuracy_score(yte,pred)),"macro_f1":float(f1_score(yte,pred,labels=LABELS,average="macro",zero_division=0)),
        "classification_report":report,"confusion_matrix":cm.tolist(),"labels":LABELS,
        "selected_classifier":{"kernel":best["kernel"],"C":best["C"]},
        "validation":{"macro_f1":best["validation_macro_f1"],"accuracy":best["validation_accuracy"],
            "candidate_results":[{k:x[k] for k in ("kernel","C","validation_macro_f1","validation_accuracy")} for x in candidates]},
        "per_dataset_test":per_dataset,"test_files":int(te.sum()),
        "test_speakers_by_dataset":{d:sorted(set(speakers[te&(datasets==d)].tolist())) for d in sorted(set(datasets[te].tolist()))},
        "note":"Speaker-independent held-out tests on acted speech. English RAVDESS/CREMA-D scores do not establish performance on spontaneous or Indian-language speech."}
    (model_dir/"speechmind_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (model_dir/"feature_extractor_config.json").write_text(json.dumps({"backbone":BASE_MODEL_NAME,"sample_rate":SAMPLE_RATE,"pooling":POOLING_VERSION,"preprocessing":PREPROCESSING_VERSION},indent=2),encoding="utf-8")
    report_path=output/"test_metrics.json"; report_path.write_text(json.dumps(report_data,indent=2),encoding="utf-8")
    print(f"[Result] Accuracy={report_data['accuracy']:.4f}, macro-F1={report_data['macro_f1']:.4f}")
    for d,m in per_dataset.items(): print(f"[Result] {d}: accuracy={m['accuracy']:.4f}, macro-F1={m['macro_f1']:.4f}")
    print(f"[Result] Selected {best['kernel']} SVM C={best['C']}; report={report_path}; checkpoint={model_dir}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ravdess-dir",default=DEFAULT_DATA_DIR)
    parser.add_argument("--crema-d-dir",default=DEFAULT_CREMA_D_DIR)
    parser.add_argument("--output-dir",default="training_outputs/wavlm_multicorpus")
    parser.add_argument("--model-dir",default="checkpoints/voice_emotion_multicorpus_cpu")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--rebuild-features",action="store_true")
    parser.add_argument("--require-crema-d",action="store_true")
    args=parser.parse_args()
    if args.require_crema_d and not Path(args.crema_d_dir).is_dir():
        raise FileNotFoundError(f"CREMA-D required but not found: {args.crema_d_dir}")
    records=assign_speaker_splits(build_manifest(args.ravdess_dir,args.crema_d_dir))
    output=Path(args.output_dir)
    manifest=write_manifest(records,output/"manifest_with_splits.csv")
    summary=summarize(records); (output/"dataset_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"[Dataset] {len(records)} labelled recordings"); print(json.dumps(summary,indent=2)); print(f"[Dataset] Manifest: {manifest}")
    if args.prepare_only: return
    features,labels,speakers,datasets,splits=extract_or_load_features(records,output/"wavlm_features_cache.npz",args.rebuild_features)
    print(f"[Features] Matrix shape: {features.shape}")
    train_classifier(features,labels,speakers,datasets,splits,output,args.model_dir)


if __name__=="__main__":
    main()
