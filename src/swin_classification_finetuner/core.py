from __future__ import annotations
import hashlib, json, os, re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DIGEST_RE=re.compile(r"^sha256:[0-9a-f]{64}$")
TASK="core.task.vision.image-classification"
REP="core.dataset.vision.image-folder"

class TypedRefusal(RuntimeError):
    def __init__(self,code:str,message:str,details:dict[str,Any]|None=None): super().__init__(message); self.code=code; self.details=details or {}

def cbytes(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def djson(v): return "sha256:"+hashlib.sha256(cbytes(v)).hexdigest()
def fdigest(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return "sha256:"+h.hexdigest()
def load_json(p:Path): return json.loads(p.read_text(encoding='utf-8'))

@dataclass(frozen=True)
class ValidatedHandoff:
    root:Path; manifest:dict[str,Any]; data_plan:dict[str,Any]; semantic_schema:dict[str,Any]; logical_manifest:dict[str,Any]
    @property
    def class_names(self): return [self.semantic_schema['labelMap'][k] for k in sorted(self.semantic_schema['labelMap'],key=int)]

def load_validated_handoff(root:Path)->ValidatedHandoff:
    root=Path(root)
    required={n:load_json(root/n) for n in ['validated-dataset-manifest.json','data-plan.json','semantic-dataset-schema.json','logical-dataset-manifest.json']}
    vm,dp,ss,lm=(required['validated-dataset-manifest.json'],required['data-plan.json'],required['semantic-dataset-schema.json'],required['logical-dataset-manifest.json'])
    if ss.get('taskProfile')!=TASK: raise TypedRefusal('HANDOFF_TASK_MISMATCH','Validated handoff is not image classification.',{'observed':ss.get('taskProfile'),'expected':TASK})
    if lm.get('representationProfile')!=REP: raise TypedRefusal('HANDOFF_REPRESENTATION_MISMATCH','Validated handoff is not image-folder.',{'observed':lm.get('representationProfile'),'expected':REP})
    if djson(dp)!=vm.get('dataPlanDigest'): raise TypedRefusal('HANDOFF_DATA_PLAN_DIGEST_MISMATCH','DataPlan digest does not match ValidatedDatasetManifest.')
    if djson(ss)!=vm.get('semanticSchemaDigest'): raise TypedRefusal('HANDOFF_SEMANTIC_SCHEMA_DIGEST_MISMATCH','SemanticDatasetSchema digest does not match ValidatedDatasetManifest.')
    logical=vm.get('logicalDatasetDigest')
    if not logical or dp.get('logicalDatasetDigest')!=logical or lm.get('logicalDatasetDigest')!=logical: raise TypedRefusal('HANDOFF_LOGICAL_IDENTITY_MISMATCH','Logical dataset identity is inconsistent across the handoff.')
    sample_ids=[s['sampleId'] for s in lm.get('samples',[])]
    assigned=[a['sampleId'] for a in dp.get('assignments',[])]
    if len(sample_ids)!=len(set(sample_ids)) or len(assigned)!=len(set(assigned)): raise TypedRefusal('HANDOFF_DUPLICATE_SAMPLE_ID','Handoff contains duplicate sample IDs.')
    if set(sample_ids)!=set(assigned): raise TypedRefusal('HANDOFF_SAMPLE_SET_MUTATION','DataPlan must assign exactly the validated logical sample set; filtering/re-splitting is forbidden.',{'logicalCount':len(sample_ids),'assignmentCount':len(assigned)})
    label_map=ss.get('labelMap')
    if not isinstance(label_map,dict) or not label_map: raise TypedRefusal('HANDOFF_LABEL_MAP_INVALID','SemanticDatasetSchema labelMap must be a non-empty object.',{'observed':type(label_map).__name__})
    keys=set(label_map)
    if keys!={str(i) for i in range(len(label_map))}: raise TypedRefusal('HANDOFF_LABEL_MAP_INVALID','labelMap keys must be exactly the decimal strings 0..N-1.',{'observed':sorted(keys)})
    names=list(label_map.values())
    if len(names)!=len(set(names)) or any(not isinstance(n,str) or not n for n in names): raise TypedRefusal('HANDOFF_LABEL_MAP_INVALID','labelMap class names must be unique non-empty strings.',{'observed':names})
    return ValidatedHandoff(root,vm,dp,ss,lm)

def load_catalog(path:Path)->dict[str,Any]: return load_json(path)

def resolve_base_model(catalog:dict[str,Any],key:str,weights_root:Path,require_qualified:bool=True)->dict[str,Any]:
    entry=catalog.get('entries',{}).get(key)
    if entry is None: raise TypedRefusal('BASE_MODEL_OFF_CATALOG','Requested base model is not allowlisted.',{'key':key})
    q=entry['qualification']
    if require_qualified and q['status']!='QUALIFIED': raise TypedRefusal('BASE_MODEL_NOT_RUNTIME_QUALIFIED','Catalog entry exists but is not runtime-qualified.',{'key':key,'reason':q['reason']})
    for f in entry['source']['files']:
        p=Path(weights_root)/key/f['path']
        if not p.is_file(): raise TypedRefusal('BASE_MODEL_FILE_MISSING','Staged base-model file is missing.',{'path':str(p)})
        actual=fdigest(p)
        if actual!=f['digest']: raise TypedRefusal('BASE_MODEL_DIGEST_MISMATCH','Staged base-model file digest does not match catalog.',{'path':str(p),'expected':f['digest'],'observed':actual})
    weight=entry['source']['files'][0]
    return {'schemaVersion':'1.0','modelDescriptorId':entry['modelDescriptorId'],'modelDescriptorVersion':entry['modelDescriptorVersion'],'contentDigest':weight['digest'],'resolutionMechanism':'preinstalled','revision':entry['source']['revision'],'companions':[{'role':'org.valcorza.base-model-catalog-entry','digest':djson(entry)}]}

def require_accelerator(expected:str|None,observed:str|None)->None:
    if not expected: raise TypedRefusal('EXPECTED_ACCELERATOR_MISSING','DIMER_EXPECTED_ACCELERATOR must be explicit; silent CPU fallback is forbidden.')
    if not observed: raise TypedRefusal('ACCELERATOR_UNAVAILABLE','No accelerator was observed; silent CPU fallback is forbidden.',{'expected':expected})
    if expected.casefold()!=observed.casefold(): raise TypedRefusal('ACCELERATOR_MISMATCH','Observed accelerator does not satisfy expected accelerator binding.',{'expected':expected,'observed':observed})

def expected_accelerator_from_env()->str|None: return os.environ.get('DIMER_EXPECTED_ACCELERATOR')
