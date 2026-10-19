#!/usr/bin/env python3
"""Causal append-only FP4 proposal scores against the pinned reference's BF16 scoring expressions."""
import argparse
import ast
import ctypes as C
import hashlib
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-lib', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--partition-only', action='store_true',
                        help='bitwise per-request versus batched launch and graph replay')
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.partition_only:
        qualify_partitions(args)
        return
    if args.reference_dir is None:
        parser.error('--reference-dir is required without --partition-only')
    root = Path(__file__).resolve().parents[3]
    lock = json.loads((root / 'scripts/fixtures/ds41-reference-lock.json').read_text())
    source = args.reference_dir / 'inference/model.py'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash == lock['files']['inference/model.py']
    tree = ast.parse(source.read_text())
    indexer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Indexer')
    forward = next(n for n in indexer.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    scoring = [n for n in forward.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'index_score' for t in n.targets)]
    assert len(scoring) == 2
    function = ast.parse('def score(q, index_k, weights):\n return index_score').body[0]
    function.body = scoring + function.body
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(source), 'exec'), namespace)
    reference = namespace['score']
    torch.cuda.set_device(args.device)
    torch.manual_seed(4132128)
    lib = C.CDLL(str(args.native_lib))
    score = lib.cuteafd_v41_index_scores_overlay
    score.argtypes = [C.c_void_p] * 12 + [C.c_int32] * 4 + [C.c_uint64, C.c_uint64, C.c_void_p]
    score.restype = C.c_int32
    stream = torch.cuda.Stream()
    results = []
    with torch.cuda.stream(stream), torch.no_grad():
        levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device='cuda')
        def decode(values, scales):
            codes = torch.stack((values & 15, values >> 4), -1).flatten(-2).long()
            v = levels[codes & 7] * torch.where(codes & 8 != 0, -1., 1.)
            return (v * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)).bfloat16()
        for queries, count, slots, stride in ((1,3,1,1), (16,255,16,4), (80,511,16,8),
                                              (256,64,16,8), (4096,3,16,1), (1,4096,16,4096), (1,16384,1,64), (16,513,16,1)):
            capacity = slots * stride * 256
            q = torch.randint(256, (queries,32,64), device='cuda', dtype=torch.uint8)
            qs = torch.randint(123,128, (queries,32,4), device='cuda', dtype=torch.uint8)
            w = torch.randn(queries,32, device='cuda').bfloat16()
            keys = torch.randint(256, (capacity,64), device='cuda', dtype=torch.uint8)
            ks = torch.randint(123,128, (capacity,4), device='cuda', dtype=torch.uint8)
            pages = torch.randperm(slots*stride, device='cuda').int().reshape(slots,stride)
            pages[-1,-1] = slots*stride-1  # exercise the last physical row
            lengths = torch.full((slots,), stride*256, device='cuda', dtype=torch.int64)
            meta = torch.zeros(queries,6, device='cuda', dtype=torch.int64)
            meta[:,0] = torch.arange(queries,device='cuda') % slots
            meta[:,1] = torch.randint(0,stride*256+1, (queries,), device='cuda')
            positions = torch.randint(0,stride*256+1,(queries,count),device='cuda',dtype=torch.int64)
            positions[:,1] = -1
            meta[0,0] = slots-1
            meta[0,1] = stride*256
            positions[0,0] = stride*256-1
            if queries > 1:
                pages[0,0] = -1  # malformed physical page
                meta[1,:2] = torch.tensor([0,stride*256],device='cuda')
                positions[1,0] = 0
                meta[-1,0] = slots  # invalid request slot
                lengths[1] = 0
            proposal_capacity=4096
            proposals=torch.randint(256,(proposal_capacity,64),device='cuda',dtype=torch.uint8)
            proposal_scales=torch.randint(123,128,(proposal_capacity,4),device='cuda',dtype=torch.uint8)
            lengths.clamp_(max=1048576-256)
            if count==513:lengths.zero_()  # entirely uncommitted prefill
            slot=meta[:,0].clamp(0,slots-1)
            meta[:,2]=lengths[slot];meta[:,5]=torch.arange(queries,device="cuda")%2+1
            meta[:,3]=256//meta[:,5];meta[:,4]=slot*256
            meta[:,1]=meta[:,2]+meta[:,3]
            positions[:,0]=meta[:,2]       # first proposal row
            positions[:,2]=meta[:,2]+meta[:,3]-1  # last proposal row
            if count>3:
                positions[:,3]=meta[:,2]-1  # last committed row, if any
            if count>4:positions[:,4]=meta[:,2]+meta[:,3]  # unreachable next row
            y = torch.empty(queries,count,device='cuda')
            tensors = [q,qs,w,keys,ks,pages,lengths,meta,positions,y,proposals,proposal_scales]
            def launch():
                assert score(*(t.data_ptr() for t in tensors),queries,count,slots,stride,capacity,proposal_capacity,stream.cuda_stream) == 0
            def compare():
                slot = meta[:,0].clamp(0,slots-1)
                logical = positions.clamp(0,stride*256-1)
                physical = pages[slot[:,None],logical//256].long()*256 + logical%256
                valid = (meta[:,0,None] < slots) & (positions >= 0) & (positions < meta[:,1,None])
                valid &= (positions < lengths[slot,None]) & (positions < stride*256)
                valid &= (physical >= 0) & (physical < capacity)
                physical = physical.clamp(0,capacity-1)
                start,n,offset,step=meta[:,2,None],meta[:,3,None],meta[:,4,None],meta[:,5,None]
                descriptor=(start==lengths[slot,None]) & (start>=0) & (start<=1048576)
                descriptor &= (n>=0) & (n<=1048576-start) & (offset>=0) & (offset<=proposal_capacity) & ((step==1)|(step==2))
                descriptor &= (n==0) | ((offset<proposal_capacity) & ((n-1)*step<proposal_capacity-offset))
                is_proposal=positions>=lengths[slot,None]
                pidx=(offset+(positions-start)*step).clamp(0,proposal_capacity-1)
                proposal_valid=(positions>=start) & (positions<start+n)
                valid=torch.where(is_proposal,proposal_valid,valid) & descriptor
                valid &= (meta[:,0,None]<slots) & (positions>=0) & (positions<meta[:,1,None])
                decoded=torch.where(is_proposal[:,:,None],decode(proposals[pidx],proposal_scales[pidx]),decode(keys[physical],ks[physical]))
                expected = reference(decode(q,qs)[:,None],decoded,w[:,None])[:,0].float()
                expected.masked_fill_(~valid,-float('inf'))
                torch.testing.assert_close(y,expected,rtol=0,atol=0)
                return int(valid.sum().item())
            launch()
            valid = compare()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                launch()
            q.bitwise_xor_(0xff)
            w.neg_()
            keys.bitwise_xor_(0x81)
            proposals.bitwise_xor_(0x43)
            meta[:,3]=128
            meta[:,4]=(meta[:,4]+256)%proposal_capacity
            # Move the last logical page while graph addresses stay fixed.
            pages[-1,-1] = 0
            graph.replay()
            changed = compare()
            saved=meta.clone()
            # Stale append base, huge counts/offsets and mismatched published
            # lengths must suppress committed and proposal reads alike.
            for field,value in [(2,-1),(2,1048577),(3,-1),(3,4097),(4,-1),(4,4097),(5,0),(5,3),(5,-1)]:
                meta.copy_(saved);meta[:,field]=value;graph.replay();compare()
                assert torch.isneginf(y).all().item()
            meta.copy_(saved)
            lengths.add_(1);graph.replay();compare();assert torch.isneginf(y).all().item();lengths.sub_(1)
            meta[:,1].zero_()
            graph.replay()
            compare()
            assert torch.isneginf(y).all().item()
            ptrs = [t.data_ptr() for t in tensors]
            for i in range(12):
                bad = ptrs.copy()
                bad[i] = 0
                assert score(*bad,queries,count,slots,stride,capacity,proposal_capacity,stream.cuda_stream) != 0
                bad[i] = (1<<64)-1
                assert score(*bad,queries,count,slots,stride,capacity,proposal_capacity,stream.cuda_stream) != 0
                if i != 9:
                    bad = ptrs.copy()
                    bad[9] = ptrs[i]
                    assert score(*bad,queries,count,slots,stride,capacity,proposal_capacity,stream.cuda_stream) != 0
            for shape in ((0,count,slots,stride,capacity),(4097,count,slots,stride,capacity),
                          (queries,0,slots,stride,capacity),(queries,16385,slots,stride,capacity),
                          (queries,count,17,stride,capacity),(queries,count,slots,4097,capacity),
                          (queries,count,slots,stride,16777217)):
                assert score(*ptrs,*shape,proposal_capacity,stream.cuda_stream) != 0
            for bad_capacity in [0,4097,2**64-1]:
                assert score(*ptrs,queries,count,slots,stride,capacity,bad_capacity,stream.cuda_stream)!=0
            results.append({'queries':queries,'candidates':count,'slots':slots,'stride':stride,
                            'capacity':capacity,'proposal_capacity':proposal_capacity,'malformed_descriptors_masked':True,'valid_scores':valid,'changed_valid_scores':changed,
                            'reference_bf16_exact':True,'all_masked_exact':True,'guards':True})
            print(f'PASS queries={queries} candidates={count} capacity={capacity}',flush=True)
            del graph,tensors,q,qs,w,keys,ks,pages,lengths,meta,positions,y
        stream.synchronize()
    args.output.write_text(json.dumps({'scope':'Committed history plus causal append-only proposals; no cache mutation',
        'gpu':torch.cuda.get_device_name(args.device),'device':args.device,
        'reference_model_sha256':source_hash,'native_library_sha256':hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
        'cases':results},indent=2)+'\n')


def qualify_partitions(args):
    """Same row arithmetic, changing only slot addressing and launch partition."""
    torch.cuda.set_device(args.device)
    torch.manual_seed(500)
    library = C.CDLL(str(args.native_lib))
    score = library.cuteafd_v41_index_scores_overlay
    score.argtypes = [C.c_void_p] * 12 + [C.c_int32] * 4 + [C.c_uint64, C.c_uint64, C.c_void_p]
    score.restype = C.c_int32
    stream = torch.cuda.Stream()
    cases = []
    with torch.cuda.stream(stream), torch.no_grad():
        rows, slots, stride, capacity = 16, 16, 4, 16 * 4 * 256
        q = torch.randint(256, (rows, 32, 64), device='cuda', dtype=torch.uint8)
        qs = torch.randint(123, 128, (rows, 32, 4), device='cuda', dtype=torch.uint8)
        weights = torch.randn(rows, 32, device='cuda').bfloat16()
        keys = torch.randint(256, (capacity, 64), device='cuda', dtype=torch.uint8)
        scales = torch.randint(123, 128, (capacity, 4), device='cuda', dtype=torch.uint8)
        pages = torch.randperm(slots * stride, device='cuda').int().reshape(slots, stride)
        lengths = torch.arange(slots, device='cuda', dtype=torch.int64) * 11 + 256
        proposals = torch.randint(256, (64, 64), device='cuda', dtype=torch.uint8)
        proposal_scales = torch.randint(123, 128, (64, 4), device='cuda', dtype=torch.uint8)
        for width in [512, 1024, 16384]:
            meta = torch.empty(rows, 6, device='cuda', dtype=torch.int64)
            local_meta = torch.empty_like(meta)
            positions = torch.empty(rows, width, device='cuda', dtype=torch.int64)
            batched = torch.empty(rows, width, device='cuda')
            separate = torch.empty_like(batched)

            def launch(begin, count, table, published, descriptors, output, slot_count):
                tensors = [q[begin:], qs[begin:], weights[begin:], keys, scales,
                           table, published, descriptors[begin:], positions[begin:],
                           output[begin:], proposals, proposal_scales]
                assert score(*(t.data_ptr() for t in tensors), count, width, slot_count,
                             stride, capacity, len(proposals), stream.cuda_stream) == 0

            def batch():
                launch(0, rows, pages, lengths, meta, batched, slots)

            batch_graph = None
            for partition in [[1] * 16, [8, 8], [1, 15], [15, 1], [3, 5, 2, 6], [16]]:
                offset = 0
                for request, count in enumerate(partition):
                    slot = (request * 7 + 3) % slots
                    end = int(lengths[slot].item())
                    for row in range(offset, offset + count):
                        step = 1 + row % 2
                        proposal_count = min(3 + row % 5, (64 - row - 1) // step)
                        meta[row] = torch.tensor([slot, end + proposal_count, end,
                            proposal_count, row, step], device='cuda')
                        positions[row] = torch.arange(width, device='cuda') + row % 7
                        positions[row, :5] = torch.tensor([-1, end - 1, end,
                            end + proposal_count - 1, end + proposal_count], device='cuda')
                    offset += count
                local_meta.copy_(meta)
                local_meta[:, 0] = 0
                separate.fill_(12345)
                offset = 0
                for request, count in enumerate(partition):
                    slot = (request * 7 + 3) % slots
                    launch(offset, count, pages[slot], lengths[slot:], local_meta, separate, 1)
                    offset += count
                if batch_graph is None:
                    batch()
                    batch_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(batch_graph, stream=stream):
                        batch()
                batched.fill_(54321)
                batch_graph.replay()
                stream.synchronize()
                assert torch.equal(separate.view(torch.int32), batched.view(torch.int32))
                assert torch.isfinite(batched).any().item()
                assert torch.isneginf(batched).any().item()
                cases.append({'partition': partition, 'width': width,
                              'score_bits_equal': True, 'replay_with_changed_metadata': True})
            del batch_graph
        qualify_long_prefill(score, stream, cases)
        stream.synchronize()
    args.output.write_text(json.dumps({'scope': 'bitwise partition invariance; committed and proposal rows',
        'native_library_sha256': hashlib.sha256(args.native_lib.read_bytes()).hexdigest(),
        'cases': cases}, indent=2) + '\n')
    print(f'PASS {len(cases)} bitwise partition/replay cases', flush=True)


def qualify_long_prefill(score, stream, cases):
    """Exercise the full prefix wave, page crossings and mutable pooled tables."""
    rows, slots, stride = 1436, 16, 8
    capacity = slots * stride * 256
    q = torch.randint(256, (rows, 32, 64), device='cuda', dtype=torch.uint8)
    qs = torch.randint(123, 128, (rows, 32, 4), device='cuda', dtype=torch.uint8)
    weights = torch.randn(rows, 32, device='cuda').bfloat16()
    keys = torch.randint(256, (capacity, 64), device='cuda', dtype=torch.uint8)
    scales = torch.randint(123, 128, (capacity, 4), device='cuda', dtype=torch.uint8)
    proposals = torch.randint(256, (rows, 64), device='cuda', dtype=torch.uint8)
    proposal_scales = torch.randint(123, 128, (rows, 4), device='cuda', dtype=torch.uint8)
    pages = torch.empty(slots, stride, device='cuda', dtype=torch.int32)
    lengths = torch.zeros(slots, device='cuda', dtype=torch.int64)
    for step in [2, 1]:
        width = ((rows // step + 511) // 512) * 512
        positions = torch.arange(width, device='cuda', dtype=torch.int64).repeat(rows, 1)
        meta = torch.empty(rows, 6, device='cuda', dtype=torch.int64)
        local_meta = torch.empty_like(meta)
        pooled = torch.empty(rows, width, device='cuda')
        separate = torch.empty_like(pooled)

        def launch(table, published, descriptors, output, slot_count):
            tensors = [q, qs, weights, keys, scales, table, published, descriptors,
                       positions, output, proposals, proposal_scales]
            assert score(*(t.data_ptr() for t in tensors), rows, width, slot_count,
                         stride, capacity, rows, stream.cuda_stream) == 0

        graph = None
        for committed in [False, True]:
            for slot in [3, 11]:
                # Install a replacement binding and publish its length before replay.
                pages.copy_(torch.randperm(slots * stride, device='cuda').int().reshape(slots, stride))
                lengths.zero_()
                end = rows // step if committed else 0
                lengths[slot] = end
                meta[:, 0] = slot
                meta[:, 1] = (torch.arange(rows, device='cuda') + 1) // step
                meta[:, 2] = end
                meta[:, 3] = 0 if committed else rows // step
                meta[:, 4] = 0 if committed or step == 1 else 1
                meta[:, 5] = step
                local_meta.copy_(meta)
                local_meta[:, 0] = 0
                separate.fill_(12345)
                launch(pages[slot], lengths[slot:], local_meta, separate, 1)
                if graph is None:
                    launch(pages, lengths, meta, pooled, slots)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        launch(pages, lengths, meta, pooled, slots)
                pooled.fill_(54321)
                graph.replay()
                stream.synchronize()
                assert torch.equal(separate.view(torch.int32), pooled.view(torch.int32))
                assert torch.equal(torch.isfinite(pooled), positions < meta[:, 1, None])
                cases.append({'kind': 'long_prefill', 'rows': rows, 'width': width,
                              'step': step, 'committed': committed, 'slot': slot,
                              'score_bits_equal': True, 'replaced_pool_metadata': True})
                print(f'PASS long prefill rows={rows} step={step} committed={committed} slot={slot}', flush=True)
        del graph


if __name__ == '__main__':
    main()
