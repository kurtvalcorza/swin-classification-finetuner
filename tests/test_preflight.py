from pathlib import Path
import json, pytest
from swin_classification_finetuner.core import TypedRefusal, djson, load_validated_handoff, require_accelerator, resolve_base_model

def catalog(tmp_path):
    return {"entries":{"ok":{"modelDescriptorId":"org.test.model","modelDescriptorVersion":"rev","timmModelName":"x","source":{"repoId":"x/y","revision":"0"*40,"files":[{"path":"model.safetensors","digest":"sha256:"+"0"*64}]},"license":{"spdx":"MIT","datasetTerms":"x"},"input":{"height":256,"width":256},"qualification":{"status":"QUALIFIED","reason":"test","measuredEnvelope":{"baseImageDigest":"sha256:"+"1"*64,"torchVersion":"2.8","cudaVersion":"12.8","device":"test","peakVramMiB":1,"height":256,"width":256,"evidenceDigest":"sha256:"+"2"*64}}}}}

def test_off_catalog_refuses(tmp_path):
    with pytest.raises(TypedRefusal) as e: resolve_base_model(catalog(tmp_path),'nope',tmp_path)
    assert e.value.code=='BASE_MODEL_OFF_CATALOG'

def test_unqualified_catalog_entry_refuses(tmp_path):
    c=catalog(tmp_path); c['entries']['ok']['qualification']={'status':'BLOCKED','reason':'no evidence','measuredEnvelope':None}
    with pytest.raises(TypedRefusal) as e: resolve_base_model(c,'ok',tmp_path)
    assert e.value.code=='BASE_MODEL_NOT_RUNTIME_QUALIFIED'

def test_digest_mismatch_refuses(tmp_path):
    c=catalog(tmp_path); p=tmp_path/'ok'/'model.safetensors'; p.parent.mkdir(); p.write_bytes(b'wrong')
    with pytest.raises(TypedRefusal) as e: resolve_base_model(c,'ok',tmp_path)
    assert e.value.code=='BASE_MODEL_DIGEST_MISMATCH'

def test_accelerator_is_fail_closed():
    with pytest.raises(TypedRefusal) as e: require_accelerator('cuda:0',None)
    assert e.value.code=='ACCELERATOR_UNAVAILABLE'
    with pytest.raises(TypedRefusal) as e: require_accelerator(None,'cuda:0')
    assert e.value.code=='EXPECTED_ACCELERATOR_MISSING'

def write(p,v): p.write_text(json.dumps(v,sort_keys=True,separators=(',',':'))+'\n')
def test_handoff_exact_sample_set_and_digests(tmp_path):
    logical='sha256:'+'a'*64
    dp={'schemaVersion':'1.0','logicalDatasetDigest':logical,'assignments':[{'sampleId':'train/cat/a.png','split':'train','reason':'directory-mapping'}],'seedPolicy':None}
    ss={'schemaVersion':'1.0','taskProfile':'core.task.vision.image-classification','fields':[],'labelMap':{'0':'cat'}}
    lm={'schemaVersion':'1.0','sourceArtifactDigest':'sha256:'+'b'*64,'representationProfile':'core.dataset.vision.image-folder','logicalDatasetDigest':logical,'samples':[{'sampleId':'train/cat/a.png','assetIds':['train/cat/a.png'],'sourceLocator':'train/cat/a.png'}],'assets':[]}
    vm={'schemaVersion':'1.0','logicalDatasetDigest':logical,'datasetProfileDigest':None,'semanticSchemaDigest':djson(ss),'dataPlanDigest':djson(dp),'validationEvidenceDigests':['sha256:'+'c'*64],'validatorWorkerReleaseDigest':'sha256:'+'d'*64,'effectiveValidationConfigDigest':'sha256:'+'e'*64,'policySetDigest':None}
    for n,v in [('data-plan.json',dp),('semantic-dataset-schema.json',ss),('logical-dataset-manifest.json',lm),('validated-dataset-manifest.json',vm)]: write(tmp_path/n,v)
    h=load_validated_handoff(tmp_path); assert h.class_names==['cat']
    dp['assignments']=[]; write(tmp_path/'data-plan.json',dp); vm['dataPlanDigest']=djson(dp); write(tmp_path/'validated-dataset-manifest.json',vm)
    with pytest.raises(TypedRefusal) as e: load_validated_handoff(tmp_path)
    assert e.value.code=='HANDOFF_SAMPLE_SET_MUTATION'

def test_handoff_label_map_is_fail_closed(tmp_path):
    logical='sha256:'+'a'*64
    dp={'schemaVersion':'1.0','logicalDatasetDigest':logical,'assignments':[{'sampleId':'train/cat/a.png','split':'train','reason':'directory-mapping'}],'seedPolicy':None}
    lm={'schemaVersion':'1.0','sourceArtifactDigest':'sha256:'+'b'*64,'representationProfile':'core.dataset.vision.image-folder','logicalDatasetDigest':logical,'samples':[{'sampleId':'train/cat/a.png','assetIds':['train/cat/a.png'],'sourceLocator':'train/cat/a.png'}],'assets':[]}
    def attempt(label_map):
        ss={'schemaVersion':'1.0','taskProfile':'core.task.vision.image-classification','fields':[],'labelMap':label_map}
        vm={'schemaVersion':'1.0','logicalDatasetDigest':logical,'datasetProfileDigest':None,'semanticSchemaDigest':djson(ss),'dataPlanDigest':djson(dp),'validationEvidenceDigests':['sha256:'+'c'*64],'validatorWorkerReleaseDigest':'sha256:'+'d'*64,'effectiveValidationConfigDigest':'sha256:'+'e'*64,'policySetDigest':None}
        for n,v in [('data-plan.json',dp),('semantic-dataset-schema.json',ss),('logical-dataset-manifest.json',lm),('validated-dataset-manifest.json',vm)]: write(tmp_path/n,v)
        return load_validated_handoff(tmp_path)
    assert attempt({'0':'cat','1':'dog'}).class_names==['cat','dog']
    for bad in ({},{'0':'cat','2':'dog'},{'x':'cat'},{'0':'cat','1':'cat'},{'0':''},{'0':None}):
        with pytest.raises(TypedRefusal) as e: attempt(bad)
        assert e.value.code=='HANDOFF_LABEL_MAP_INVALID'
