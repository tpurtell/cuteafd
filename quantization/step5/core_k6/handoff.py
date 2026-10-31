#!/usr/bin/env python3
"""Verified zero-copy Sparknest adoption, six-Spark placement and safe cleanup."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'quantization'))
if __package__ in (None,''):__package__='step5.core_k6'
from kimi_k3.uniform_k2.storage import atomic_json,sha256_file
from .recipe import TARGET_REPO
HOSTS=('ostrich','dodo','emu','kiwi','rhea','moa')

def nest(*args,timeout=3600):
    return json.loads(subprocess.check_output(['nest',*map(str,args),'--json'],text=True,timeout=timeout))

def successful(result):
    job=result.get('job')
    if not job or not job.get('finished') or job.get('error') or job.get('cancelled') or job.get('pending'):
        raise ValueError('Sparknest job did not finish successfully')
    if any(not h.get('finished') or h.get('failed') or h.get('cancelled') or h.get('done_files')!=h.get('total_files') or (h.get('total_bytes',0)>0 and h.get('done_bytes')!=h['total_bytes']) for h in job.get('hosts',{}).values()):
        raise ValueError('Sparknest host job failed')

def safe_remove(path,parent):
    path,parent=Path(path),Path(parent)
    if path.is_symlink() or path.parent.resolve()!=parent.resolve():
        raise ValueError('cleanup must name exactly one owned standalone leaf')
    if path.exists():shutil.rmtree(path)

def verify_files(snapshot,files):
    if snapshot.is_symlink() or not snapshot.is_dir():raise ValueError('managed snapshot unavailable')
    actual={p.relative_to(snapshot).as_posix() for p in snapshot.rglob('*') if p.is_file()}
    if actual!=set(files):raise ValueError('managed file inventory differs')
    for name,meta in files.items():
        path=snapshot/name
        if path.is_symlink() or path.stat().st_size!=meta['bytes'] or sha256_file(path)!=meta['sha256']:
            raise ValueError('managed snapshot checksum differs: '+name)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();marker=root/'handoff-complete.json'
    if marker.exists():
        complete=json.loads(marker.read_text())
        if complete.get('repo_id')!=TARGET_REPO or complete.get('status')!='passed':raise ValueError('handoff marker differs')
        print('Sparknest handoff already completed.');return
    plan=json.loads((root/'plan.json').read_text());published=json.loads((root/'publication-complete.json').read_text())
    if plan['publication_repo']!=TARGET_REPO or published['repo_id']!=TARGET_REPO or published['identity']!=plan['identity']:
        raise ValueError('handoff requires matching verified publication')
    revision=published['revision'];model='models--'+TARGET_REPO.replace('/','--')
    local_hub=Path.home()/'.cache/huggingface/cuteafd-local-hub'
    local=local_hub/model/'snapshots'/revision
    if Path(published['local_snapshot'])!=local:raise ValueError('unexpected standalone cache path')
    managed=Path('/mnt/sparknest/hf-home/hub')/model/'snapshots'/revision
    imported=root/'sparknest-import-complete.json'
    stage_parent=Path('/srv/sparknest/import-sources');stage=stage_parent/model
    if imported.exists():
        saved=json.loads(imported.read_text())
        if saved['revision']!=revision:raise ValueError('prior import revision differs')
        files=saved['files']
    else:
        if local.is_symlink() or not local.is_dir():raise ValueError('verified local HF snapshot missing')
        files={}
        for path in local.rglob('*'):
            if path.is_symlink():raise ValueError('standalone snapshot contains symlink')
            if path.is_file():files[path.relative_to(local).as_posix()]=dict(bytes=path.stat().st_size,sha256=sha256_file(path))
        if len(files)!=published['files']:raise ValueError('standalone HF file count differs')
        stage_parent.mkdir(parents=True,exist_ok=True)
        if stage.is_symlink():raise ValueError('temporary import leaf is symlink')
        snapshot=stage/'snapshots'/revision;snapshot.mkdir(parents=True,exist_ok=True)
        # /home and /srv are separate Btrfs subvolumes: hardlinks cannot cross
        # that boundary, but mandatory reflinks share every physical data block.
        for name,meta in files.items():
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if target.is_symlink() or target.stat().st_size!=meta['bytes'] or sha256_file(target)!=meta['sha256']:
                    raise ValueError('existing import stage differs')
            else:subprocess.run(['cp','--reflink=always','--preserve=timestamps','--',str(local/name),str(target)],check=True,timeout=120)
        refs=stage/'refs';refs.mkdir(exist_ok=True);(refs/'main').write_text(revision)
        result=nest('hf','import',stage,'--offline','--no-copy','--wait',timeout=3600)
        successful(result)
        stat=result.get('import')
        if not stat or not stat.get('finished') or stat.get('errors') or stat.get('copied_bytes')!=0 or stat.get('files')!=len(files):
            raise ValueError('Sparknest zero-copy import not verified')
        verify_files(managed,files)
        atomic_json(imported,dict(status='passed',revision=revision,files=files,copied_bytes=0,result=result))
    # Copy onto the fabric before retiring any off-grid managed copies.
    placement=nest('plan','--place','hf:'+TARGET_REPO,*[v for host in HOSTS for v in ('--host',host)],'--factor','1/6')
    if not placement.get('feasible') or placement.get('blocked'):raise ValueError('six-Spark placement infeasible')
    atomic_json(root/'sparknest-placement-plan.json',placement)
    result=nest('plan','apply',placement['id'],'--wait',timeout=14400);successful(result)
    atomic_json(root/'sparknest-placement-complete.json',result)
    counts=Counter()
    for step in placement.get('steps',[]):
        if step['kind']=='replicate':counts[step['host']]+=sum(c['name'].endswith('.safetensors') for c in step['copies'])
    # Full hashes were checked while local; now verify every managed file's
    # complete fabric placement from authoritative Sparknest metadata.
    seen=set()
    for directory in {str(Path(n).parent) for n in files}:
        entries=nest('ls','/hf-home/hub/'+model+'/snapshots/'+revision+('/'+directory if directory!='.' else ''))['entries']
        for entry in entries:
            relative=(Path(directory)/entry['name']).as_posix()
            if relative not in files:continue
            if (len(set(entry['hosts'])&set(HOSTS))!=1 or entry.get('size')!=files[relative]['bytes']
                    or not entry.get('sealed') or entry.get('writing_on') is not None):
                raise ValueError('file does not have exactly one sealed fabric copy')
            seen.add(relative)
    if seen!=set(files):raise ValueError('fabric file inventory incomplete')
    refs=nest('ls','/hf-home/hub/'+model+'/refs')['entries']
    main=[e for e in refs if e['name']=='main']
    if len(main)!=1 or len(set(main[0]['hosts'])&set(HOSTS))!=1:
        raise ValueError('HF main ref placement differs')
    if (managed.parents[1]/'refs/main').read_text()!=revision:
        raise ValueError('managed HF main ref revision differs')
    safe_remove(stage,stage_parent)
    safe_remove(local_hub/model,local_hub)
    export_parent=Path.home()/'.cache/huggingface/cuteafd-exports'
    export=Path(plan['export'])
    if export.name!='Step-5-Preview-EXL3-K4-Core-K6':raise ValueError('unexpected export cleanup leaf')
    safe_remove(export,export_parent)
    # The remaining live data are Sparknest-managed; remove the off-grid maxq
    # copy only after the single distributed fabric copy has been confirmed.
    offgrid=nest('plan','--offgrid','hf:'+TARGET_REPO)
    if not offgrid.get('feasible') or offgrid.get('blocked'):raise ValueError('off-grid retirement infeasible')
    if offgrid.get('steps'):
        retired=nest('plan','apply',offgrid['id'],'--wait');successful(retired)
    atomic_json(marker,dict(status='passed',repo_id=TARGET_REPO,revision=revision,files=len(files),copied_bytes=0,
        fabric_hosts=HOSTS,one_fabric_copy_per_file=True,standalone_export_removed=True,alternate_snapshot_removed=True,
        temporary_stage_removed=True,offgrid_copies_retired=True,time=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(json.loads(marker.read_text())),flush=True)

if __name__=='__main__':main()
