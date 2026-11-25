//! Teacher-forced width oracle on the admitted serving executor and real Spark experts.
use super::{cache::{InklingPrefix, Placement}, engine::Execution, golden::GoldenArgs, owners::{UnitBuffers,UnitOwner}};
use crate::shared::{prefill_pipeline::{Unit,UnitSegment}, speculation::target::{BlockVerifyTarget,VerifyRequest,VerifyRow}};
use anyhow::{ensure,Context,Result};
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use sha2::{Digest,Sha256};
use std::{cell::RefCell,collections::BTreeMap,sync::Arc};

#[derive(Default)]
pub(crate) struct Trace{pub samples:Vec<Sample>}
pub(crate) struct Sample{layer:usize,stage:String,rows:usize,buffers:BTreeMap<String,Vec<u8>>}
impl Trace{
    pub fn capture(&mut self,layer:usize,stage:&str,rows:usize,owner:&UnitOwner<'_>)->Result<()>{
        let fields:&[&str]=match stage{
            "entry"|"exit"=>&["partial"],
            "attention"=>&["q_bf16","k","v","conv_k","conv_v","k_clipped","v_norm",
                "relative","sinks","attended","gated","attention","projection_input"],
            "experts"=>&["projection_input","expert_wire","route_logits","route_ids","route_weights","expert_output"],
            _=>anyhow::bail!("unknown Inkling width trace stage {stage}")};
        owner.stream.drain()?;
        let buffers=fields.iter().map(|&name|{
            let buffer=owner.buffer(name)?;
            ensure!(buffer.bytes.is_multiple_of(owner.layout.rows),"width trace {name} has an invalid rank-local row stride");
            let bytes=rows.checked_mul(buffer.bytes/owner.layout.rows).context("width trace extent overflow")?;
            ensure!(bytes<=buffer.bytes,"width trace {name} outside owner");
            let mut data=vec![0;bytes];owner.stream.device.library.copy_d2h(&mut data,Buffer{bytes,..buffer})?;
            Ok((name.to_owned(),data))
        }).collect::<Result<_>>()?;
        self.samples.push(Sample{layer,stage:stage.into(),rows,buffers});Ok(())
    }
}
fn unit(start:usize,rows:usize,layers:usize)->Unit{
    Unit{index:0,sequence:0,rows:0..rows,layers:0..layers,slot:0,barrier:false,segments:vec![UnitSegment{
        sequence:0,rows:0..rows,tokens:start..start+rows,prompt_end:false,snapshot:false,head:true}]}
}
fn decode(execution:&mut Execution<'_>,placement:&mut Placement,tokens:&[u32])->Result<Vec<u8>>{
    let unit=unit(placement.len,tokens.len(),execution.cfg.layers);
    let logits=execution.decode(std::slice::from_mut(placement),&[tokens],&unit,1)?;
    let bytes=tokens.len()*logits.vocab*4;let mut data=vec![0;bytes];
    execution.requests.decode.stream.device.library.copy_d2h(&mut data,Buffer{ptr:logits.ptr.cast_mut(),bytes,
        ..execution.head.logits.buffer})?;Ok(data)
}
fn verify(execution:&mut Execution<'_>,placement:&mut Placement,tokens:&[u32])->Result<Vec<u8>>{
    verify_kept(execution,placement,tokens,None)
}
fn verify_kept(execution:&mut Execution<'_>,placement:&mut Placement,tokens:&[u32],keep:Option<usize>)->Result<Vec<u8>>{
    let taps=execution.draft.as_ref().context("width oracle needs admitted FIR verifier")?.layout.config.taps.clone();
    let rows=tokens.iter().enumerate().map(|(offset,&token)|VerifyRow{slot:placement.slot,position:placement.len+offset,token:Some(token)}).collect::<Vec<_>>();
    let library=execution.requests.decode.stream.device.library;
    let mut target=super::verify_target::Target::new(execution,std::slice::from_mut(placement));
    let mark=target.checkpoint(&[0])?;
    let result:Result<Vec<u8>>=(||{
        let logits=target.verify(&mark,VerifyRequest{rows:&rows,tap_sites:&taps})?.logits;
        let values=logits.to_host(library)?;
        crate::shared::spark_waves::healthy()?;
        Ok(values.into_iter().flat_map(f32::to_le_bytes).collect())
    })();
    if result.is_err(){target.restore(&mark)?;}
    let result=result?;
    if let Some(rows)=keep{target.commit(&mark,&[crate::shared::speculation::target::AcceptedPrefix{slot:0,rows}])?;}
    else{target.restore(&mark)?;}
    Ok(result)
}
fn state(execution:&Execution<'_>,placement:&Placement)->Result<BTreeMap<String,Vec<u8>>>{
    let mut output=BTreeMap::new();
    output.insert("bookkeeping".into(),serde_json::to_vec(&(placement.len,placement.state_len,&placement.units))?);
    for (rank,requests) in execution.request_ranks().enumerate(){
        let bank=&requests.cache;
        let stream=&requests.decode.stream;
        stream.drain()?;
        stream.device.run(||{
            let read=|buffer:Buffer|->Result<Vec<u8>>{let mut bytes=vec![0;buffer.bytes];stream.device.library.copy_d2h(&mut bytes,buffer)?;Ok(bytes)};
            output.insert(format!("rank.{rank}.conv"),read(bank.conv_for_sequence(placement.slot)?)?);
            for &layer in &bank.layout.global_layers{
                let stride=bank.layout.page_bytes/64;let mut bytes=Vec::new();
                for position in (0..placement.len).step_by(64){
                    let page=bank.page(layer,placement.units[position/64] as usize)?;
                    bytes.extend(read(Buffer{bytes:(placement.len-position).min(64)*stride,..page})?);
                }
                output.insert(format!("rank.{rank}.global.layer.{layer}"),bytes);
            }
            for (layer,geometry) in bank.layout.layers.iter().enumerate(){if geometry.ring_rows==0{continue;}
                let ring=bank.field(placement.slot,layer,"ring")?;let stride=geometry.kv_width*4;
                let mut bytes=Vec::new();
                for position in placement.len.saturating_sub(execution.cfg.window)..placement.len{
                    let offset=position%geometry.ring_rows*stride;
                    bytes.extend(read(Buffer{ptr:ring.ptr.wrapping_byte_add(offset),bytes:stride,..ring})?);
                }
                output.insert(format!("rank.{rank}.ring.layer.{layer}"),bytes);
            }
            Ok(())
        })?;
    }
    Ok(output)
}
fn restore(execution:&Execution<'_>)->Result<()> {
    for requests in execution.request_ranks(){requests.cache.restore(0,0)?;requests.cache.drain()?;}
    Ok(())
}

fn state_difference(actual:&BTreeMap<String,Vec<u8>>,expected:&BTreeMap<String,Vec<u8>>)->Option<serde_json::Value>{
    actual.iter().find_map(|(owner,bytes)|{let reference=expected.get(owner)?;
        if bytes==reference{return None;}
        let first=bytes.iter().zip(reference).position(|(a,b)|a!=b).unwrap_or(bytes.len().min(reference.len()));
        Some(serde_json::json!({"owner":owner,"byte":first,"actual_bytes":bytes.len(),"reference_bytes":reference.len(),
            "actual":bytes.get(first),"reference":reference.get(first)}))
    })
}
fn digest(bytes:&[u8])->String { format!("{:x}",Sha256::digest(bytes)) }
fn commit_sweep(execution:&mut Execution<'_>,saved:&Placement,tokens:&[u32],widths:&[usize])->Result<serde_json::Value>{
    let mut passes=Vec::new();let mut clean=true;
    execution.width_trace=None;
    for &width in widths{for keep in 0..=width{
        restore(execution)?;let mut reference=saved.clone();
        for token in &tokens[..keep]{decode(execution,&mut reference,std::slice::from_ref(token))?;}
        let expected=state(execution,&reference)?;
        let next_reference=decode(execution,&mut reference,&tokens[keep..keep+1])?;
        restore(execution)?;let mut candidate=saved.clone();
        verify_kept(execution,&mut candidate,&tokens[..width],Some(keep))?;
        let actual=state(execution,&candidate)?;let difference=state_difference(&actual,&expected);
        let next=decode(execution,&mut candidate,&tokens[keep..keep+1])?;
        let next_equal=next==next_reference;clean&=difference.is_none()&&next_equal;
        let result=serde_json::json!({"width":width,"accepted":keep,"first_state_difference":difference,"next_logits_equal":next_equal,"next_logits_sha256":digest(&next)});
        println!("Inkling commit: {}",result);passes.push(result);
    }}Ok(serde_json::json!({"pass":clean,"passes":passes}))
}
fn compare(candidate:&Trace,reference:&[Trace],logits:&[u8],base_logits:&[Vec<u8>],vocab:usize)->Result<Vec<serde_json::Value>>{
    let mut differences=Vec::new();
    for (row,base) in reference.iter().enumerate(){
        let mut first=None;
        ensure!(candidate.samples.len()==base.samples.len(),"width trace stage inventory differs");
        for (sample,other) in candidate.samples.iter().zip(&base.samples){
            ensure!(sample.layer==other.layer && sample.stage==other.stage && other.rows==1,"width trace order differs");
            let order:Vec<&str>=if sample.stage=="experts"{
                vec!["projection_input","expert_wire","route_logits","route_ids","route_weights","expert_output"]
            }else if sample.stage=="attention" && sample.buffers.contains_key("q_bf16"){
                vec!["q_bf16","k","v","conv_k","conv_v","k_clipped","v_norm","relative","sinks","attended","gated","attention","projection_input"]
            }else{sample.buffers.keys().map(String::as_str).collect()};
            let mut equality=BTreeMap::new();
            let mut operand=None;
            for name in order{
                let data=sample.buffers.get(name).context("candidate stage operand missing")?;
                let baseline=other.buffers.get(name).context("reference stage operand missing")?;
                ensure!(data.len()==sample.rows*baseline.len(),"width trace operand pitch differs");
                let actual=&data[row*baseline.len()..(row+1)*baseline.len()];
                equality.insert(name,actual==baseline);
                if actual!=baseline && operand.is_none(){
                    operand=Some((name,actual.iter().zip(baseline).filter(|(a,b)|a!=b).count()));
                }
            }
            if let Some((name,bytes))=operand{
                first=Some(serde_json::json!({"row":row,"layer":sample.layer,"stage":sample.stage,"operand":name,
                    "different_bytes":bytes,"operands_equal":equality}));
            }
            if first.is_some(){break;}
        }
        let baseline=&base_logits[row];let actual=&logits[row*vocab*4..(row+1)*vocab*4];
        let logit_diff=actual!=baseline;
        if first.is_some()||logit_diff{
            differences.push(serde_json::json!({"row":row,"first":first,"logits_differ":logit_diff}));
        }
    }Ok(differences)
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn width_comparison_reports_first_layer_operand_and_row_pitch()->Result<()>{
        let sample=|rows,bytes|Sample{layer:0,stage:"attention".into(),rows,
            buffers:BTreeMap::from([("attention".into(),bytes)])};
        let candidate=Trace{samples:vec![sample(2,vec![1,2,3,5])]};
        let reference=[Trace{samples:vec![sample(1,vec![1,2])]},Trace{samples:vec![sample(1,vec![3,4])]}];
        let result=compare(&candidate,&reference,&[0;8],&[vec![0;4],vec![0;4]],1)?;
        assert_eq!(result.len(),1);assert_eq!(result[0]["row"],1);
        assert_eq!(result[0]["first"]["operand"],"attention");
        assert_eq!(result[0]["first"]["different_bytes"],1);
        assert_eq!(result[0]["logits_differ"],false);Ok(())
    }
    #[test]
    fn golden_cli_exposes_actual_served_width_configuration()->Result<()>{
        use clap::Parser;
        let cli=crate::cli::Cli::try_parse_from(["cuteafd","inkling2-golden","--snapshot","/model",
            "--native-lib","/lib/native.so","--tokens","/tokens.json","--save-logits","/out",
            "--width-invariance","1,8","--peers","10.55.0.1:19885,10.55.0.2:19885,10.55.0.3:19885,10.55.0.4:19885",
            "--prefill","74","--rtx-expert-layers","auto"])?;
        let crate::cli::Commands::Inkling2Golden(args)=cli.command else{anyhow::bail!("wrong golden command")};
        assert_eq!(args.width_invariance,[1,8]);assert_eq!(args.prefill,74);assert_eq!(args.peers.len(),4);Ok(())
    }
}

/// Prove this exact row width really replayed, not just that lazy first-touch
/// agreed with eager execution. Unlike the width oracle, do not compare
/// different widths: long contexts can reorder router ties (#1128).
fn graph_parity(execution: &mut Execution<'_>, saved: &Placement, tokens: &[u32], args: &GoldenArgs) -> Result<()> {
    use super::graphs::Mode;
    let mut digests = Vec::new();
    let mut evidence = Vec::new();
    for (mode, width) in std::iter::once((Mode::Decode, 1)).chain(args.width_invariance.iter().map(|&width| (Mode::Verify, width))) {
        let mut expected = None;
        let mut before_replay = 0;
        for pass in 0..3 {
            restore(execution)?;
            let mut placement = saved.clone();
            let window = &tokens[..width];
            let logits = match mode {
                Mode::Decode => decode(execution, &mut placement, window)?,
                Mode::Verify => verify(execution, &mut placement, window)?,
            };
            let sha = digest(&logits);
            ensure!(expected.as_ref().is_none_or(|expected| *expected == sha),
                "Inkling {mode:?} row {width} changed logits on pass {pass}");
            expected = Some(sha);
            if pass == 1 { before_replay = cuteafd_ffi::graph_captures(); }
        }
        ensure!(cuteafd_ffi::graph_captures() == before_replay, "Inkling {mode:?} row {width} recaptured on replay");
        let banks = std::iter::once(execution.graphs.as_ref())
            .chain(execution.peer.iter().map(|p| p.graphs.as_ref()));
        let mut retained = Vec::new();
        for graph in banks {
            // Every rank captures attention once per target layer; route-only
            // capture is not a valid replay witness.
            let n = graph.map_or(0, |g| g.retained(width, mode));
            ensure!(graph.is_none() || n >= execution.cfg.layers,
                "Inkling {mode:?} row {width} retained only {n} graphs on a rank");
            retained.push(n);
        }
        digests.push((format!("{mode:?}"), width, expected.unwrap()));
        evidence.push((format!("{mode:?}"), width, retained));
    }
    std::fs::create_dir_all(&args.save_logits)?;
    std::fs::write(args.save_logits.join("logits-sha256.json"), serde_json::to_vec_pretty(&digests)?)?;
    std::fs::write(args.save_logits.join("graph-parity.json"), serde_json::to_vec_pretty(&evidence)?)?;
    Ok(())
}

fn kl_and_top1(candidate: &[u8], reference: &[u8], vocab: usize) -> Result<(usize, f64, f32, usize)> {
    ensure!(candidate.len() == reference.len() && candidate.len().is_multiple_of(vocab * 4),
        "wide-batch logit buffer length differs");
    let rows = candidate.len() / (vocab * 4);
    let mut top1_matches = 0usize;
    let mut bit_identical = 0usize;
    let mut kl_sum = 0f64;
    let mut max_abs = 0f32;
    for r in 0..rows {
        let c_bytes = &candidate[r * vocab * 4..(r + 1) * vocab * 4];
        let e_bytes = &reference[r * vocab * 4..(r + 1) * vocab * 4];
        if c_bytes == e_bytes { bit_identical += 1; }
        let c: Vec<f32> = c_bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let e: Vec<f32> = e_bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        ensure!(c.iter().all(|v| v.is_finite()) && e.iter().all(|v| v.is_finite()), "wide-batch nonfinite logits at row {r}");
        let (mut c_max, mut c_arg) = (f32::NEG_INFINITY, 0usize);
        let (mut e_max, mut e_arg) = (f32::NEG_INFINITY, 0usize);
        for i in 0..vocab {
            if c[i] > c_max { c_max = c[i]; c_arg = i; }
            if e[i] > e_max { e_max = e[i]; e_arg = i; }
            let diff = (c[i] - e[i]).abs();
            if diff > max_abs { max_abs = diff; }
        }
        if c_arg == e_arg { top1_matches += 1; }
        let c_lse = c_max as f64 + c.iter().map(|&v| ((v as f64) - c_max as f64).exp()).sum::<f64>().ln();
        let e_lse = e_max as f64 + e.iter().map(|&v| ((v as f64) - e_max as f64).exp()).sum::<f64>().ln();
        let mut row_kl = 0f64;
        for i in 0..vocab {
            let log_p = e[i] as f64 - e_lse;
            let log_q = c[i] as f64 - c_lse;
            row_kl += log_p.exp() * (log_p - log_q);
        }
        kl_sum += row_kl.max(0.0);
    }
    Ok((top1_matches, kl_sum / rows as f64, max_abs, bit_identical))
}

fn wide_batch_oracle(
    execution: &mut Execution<'_>,
    inputs: &cuteafd_loader::placement::families::inkling2::InklingInputs,
    admitted: &cuteafd_loader::placement::families::inkling2::AdmittedInkling,
    tokens: &[u32],
    args: &GoldenArgs,
) -> Result<()> {
    use super::graphs::Mode;
    use crate::shared::speculation::target::AcceptedPrefix;
    let seqs = 16usize;
    let width = 8usize;
    let total_rows = seqs * width;
    ensure!(args.decode_rows as usize == total_rows && args.max_sequences >= seqs && inputs.prefix_mark_slots >= seqs,
        "Inkling wide-batch oracle requires --decode-rows 128, --max-sequences >= 16 and >= 16 prefix mark slots");
    let needed = args.prefill + width + 1;
    ensure!(tokens.len() >= needed, "wide-batch oracle needs >= {} tokens, got {}", needed, tokens.len());
    let stride_max = tokens.len() - needed;
    let bank = execution.requests.cache.clone();
    let pages_per_seq = needed.div_ceil(64);
    ensure!(seqs * pages_per_seq <= bank.layout.pool_pages, "wide-batch oracle exceeds admitted target pages");
    let mut seq_slices: Vec<&[u32]> = Vec::with_capacity(seqs);
    let mut saved: Vec<Placement> = Vec::with_capacity(seqs);
    for s in 0..seqs {
        let off = if stride_max == 0 { 0 } else { (s * 17) % (stride_max + 1) };
        let slice = &tokens[off..off + needed];
        seq_slices.push(slice);
        let base_page = (s * pages_per_seq) as u32;
        let mut p = Placement {
            media: None,
            units: (0..pages_per_seq as u32).map(|k| base_page + k).collect(),
            slot: s,
            len: 0,
            state_len: 0,
        };
        InklingPrefix::for_execution(execution).bind_slot(&mut p, Some(args.prefill + width))?;
        let mut units = crate::shared::prefill_pipeline::sequence_units(args.prefill, args.prefill_rows, args.prefill_waves as usize, execution.cfg.layers)?;
        for unit in &mut units {
            unit.segments = vec![UnitSegment { sequence: 0, rows: 0..unit.rows.len(), tokens: unit.rows.clone(), prompt_end: false, snapshot: false, head: false }];
        }
        execution.prefill(std::slice::from_mut(&mut p), &[&slice[..args.prefill]], &units, 1, super::prefill::HeadRows::All, &mut |_, _, _| Ok(()))?;
        for requests in execution.request_ranks() { requests.cache.capture(s, s)?; requests.cache.drain()?; }
        saved.push(p);
    }
    let restore_slot = |exec: &Execution<'_>, s: usize| -> Result<()> {
        for requests in exec.request_ranks() { requests.cache.restore(s, s)?; requests.cache.drain()?; }
        Ok(())
    };
    let restore_all = |exec: &Execution<'_>| -> Result<()> {
        for requests in exec.request_ranks() {
            for s in 0..seqs { requests.cache.restore(s, s)?; }
            requests.cache.drain()?;
        }
        Ok(())
    };
    // 1) Solo-sequence serial width-1 reference (16 x 8 = 128 rows).
    let mut solo_logits = Vec::with_capacity(total_rows * execution.cfg.unpadded_vocab_size * 4);
    for s in 0..seqs {
        restore_slot(execution, s)?;
        let mut p = saved[s].clone();
        for step in 0..width {
            let row = decode(execution, &mut p, &seq_slices[s][args.prefill + step..args.prefill + step + 1])?;
            solo_logits.extend_from_slice(&row);
        }
    }
    // 2) 128-row graphed verify: 3 passes (capture, replay, replay).
    let taps = execution.draft.as_ref().context("wide-batch oracle needs admitted FIR verifier")?.layout.config.taps.clone();
    let slots: Vec<usize> = (0..seqs).collect();
    let mut rows = Vec::with_capacity(total_rows);
    for s in 0..seqs {
        for offset in 0..width {
            rows.push(VerifyRow { slot: s, position: args.prefill + offset, token: Some(seq_slices[s][args.prefill + offset]) });
        }
    }
    let library = execution.requests.decode.stream.device.library;
    let mut pass_digests = Vec::new();
    let mut wide_logits = Vec::new();
    let mut before_replay = 0u64;
    for pass in 0..3 {
        restore_all(execution)?;
        let mut placements = saved.clone();
        let mut target = super::verify_target::Target::new(execution, &mut placements);
        let mark = target.checkpoint(&slots)?;
        let res: Result<Vec<u8>> = (|| {
            let logits = target.verify(&mark, VerifyRequest { rows: &rows, tap_sites: &taps })?.logits;
            let values = logits.to_host(library)?;
            crate::shared::spark_waves::healthy()?;
            Ok(values.into_iter().flat_map(f32::to_le_bytes).collect())
        })();
        target.restore(&mark)?;
        let bytes = res?;
        let sha = digest(&bytes);
        if let Some(first) = pass_digests.first() {
            ensure!(*first == sha, "Inkling 128-row verify changed logits on pass {pass}");
        }
        pass_digests.push(sha);
        if pass == 0 { wide_logits = bytes; }
        if pass == 1 { before_replay = cuteafd_ffi::graph_captures(); }
    }
    let replay_recaptures = cuteafd_ffi::graph_captures() - before_replay;
    ensure!(replay_recaptures == 0, "Inkling 128-row verify recaptured {replay_recaptures} graphs on replay");
    let mut retained = Vec::new();
    for graph in std::iter::once(execution.graphs.as_ref()).chain(execution.peer.iter().map(|p| p.graphs.as_ref())) {
        let n = graph.map_or(0, |g| g.retained(total_rows, Mode::Verify));
        ensure!(graph.is_none() || n >= execution.cfg.layers, "Inkling 128-row verify retained only {n} graphs on a rank");
        retained.push(n);
    }
    let (top1_matches, mean_kl, max_abs_diff, bit_identical_rows) = kl_and_top1(&wide_logits, &solo_logits, execution.cfg.unpadded_vocab_size)?;
    ensure!(top1_matches == total_rows && mean_kl <= 1e-3,
        "Inkling 128-row fidelity failed vs solo eager: top1={top1_matches}/{total_rows}, mean_kl={mean_kl}");
    // 3) 128-row fir_history_commit_m128 accepted-prefix commit across all 16 slots.
    restore_all(execution)?;
    let mut placements = saved.clone();
    let accepted: Vec<AcceptedPrefix> = (0..seqs).map(|s| AcceptedPrefix { slot: s, rows: s % 9 }).collect();
    {
        let mut target = super::verify_target::Target::new(execution, &mut placements);
        let mark = target.checkpoint(&slots)?;
        let _ = target.verify(&mark, VerifyRequest { rows: &rows, tap_sites: &taps })?;
        target.commit(&mark, &accepted)?;
    }
    let mut actual_states = Vec::with_capacity(seqs);
    let mut actual_next = Vec::with_capacity(seqs);
    for s in 0..seqs {
        actual_states.push(state(execution, &placements[s])?);
        let keep = accepted[s].rows;
        actual_next.push(decode(execution, &mut placements[s], &seq_slices[s][args.prefill + keep..args.prefill + keep + 1])?);
    }
    let mut commit_checks = Vec::with_capacity(seqs);
    let mut commit_clean = true;
    for s in 0..seqs {
        let keep = accepted[s].rows;
        restore_slot(execution, s)?;
        let mut reference = saved[s].clone();
        for step in 0..keep {
            decode(execution, &mut reference, &seq_slices[s][args.prefill + step..args.prefill + step + 1])?;
        }
        let expected_state = state(execution, &reference)?;
        let diff = state_difference(&actual_states[s], &expected_state);
        let expected_next = decode(execution, &mut reference, &seq_slices[s][args.prefill + keep..args.prefill + keep + 1])?;
        let next_equal = actual_next[s] == expected_next;
        commit_clean &= diff.is_none() && next_equal;
        commit_checks.push(serde_json::json!({"slot": s, "accepted": keep, "state_diff": diff, "next_logits_equal": next_equal}));
    }
    let report = serde_json::json!({
        "pass": commit_clean,
        "decode_rows": total_rows,
        "sequences": seqs,
        "width_per_sequence": width,
        "pool_tokens": admitted.solved.pool_tokens,
        "onboard": admitted.onboard,
        "pass_sha256": pass_digests,
        "replay_recaptures": replay_recaptures,
        "retained_graphs_per_rank": retained,
        "solo_vs_wide": {
            "rows": total_rows,
            "top1_matches": top1_matches,
            "top1": top1_matches as f64 / total_rows as f64,
            "mean_kl": mean_kl,
            "max_abs_diff": max_abs_diff,
            "bit_identical_rows": bit_identical_rows
        },
        "commit_128": { "pass": commit_clean, "slots": commit_checks }
    });
    std::fs::create_dir_all(&args.save_logits)?;
    std::fs::write(args.save_logits.join("wide-batch-128.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("Inkling wide-batch-128: pass={commit_clean} top1={top1_matches}/{total_rows} mean_kl={mean_kl:.6} max_abs={max_abs_diff:.6} bit_identical={bit_identical_rows}/{total_rows} retained={retained:?} recaptures={replay_recaptures}");
    ensure!(commit_clean, "Inkling 128-row FIR commit oracle failed; see wide-batch-128.json");
    Ok(())
}

pub(crate) fn run(args:&mut GoldenArgs)->Result<()>{
    use cuteafd_loader::{plan::Checkpoint,placement::planner::PrefillShape};
    ensure!(args.width_invariance.iter().all(|width|(1..=8).contains(width)),"Inkling width invariance supports1..=8 total verify rows, anchor included");
    ensure!(!args.save_logits.exists(),"width oracle output must be new");
    let maximum=*args.width_invariance.iter().max().context("width list empty")?;
    let tokens:Vec<u32>=serde_json::from_slice(&std::fs::read(&args.tokens)?)?;
    ensure!(args.prefill>0 && tokens.len()>=args.prefill+maximum+usize::from(args.commit_invariance),"width oracle needs prefix plus maximum-width teacher-forced tokens");
    let view=args.attention.selected_view.clone().map(Ok).unwrap_or_else(||args.attention.select(&args.snapshot))?;
    args.attention.validate_view(&view)?;
    let checkpoint=Checkpoint::open_view(&view)?;
    let cfg=cuteafd_loader::families::inkling2::Inkling2Config::from_hf(&checkpoint.config)?;
    ensure!(tokens.iter().all(|&token|(token as usize)<cfg.unpadded_vocab_size),"width oracle invalid token id");
    args.prefix.resolve_entries(args.max_sequences);
    ensure!(args.prefix.prefix_cache_entries>0,"width oracle requires exact prefix staging marks");
    cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::INKLING2)
        .map_err(|previous|anyhow::anyhow!("Inkling width oracle requires isolated process geometry, found {previous:?}"))?;
    // SAFETY: all runtime/native owners are destroyed before this library and its programs.
    let library=Arc::new(unsafe{cuteafd_ffi::NativeLibrary::load(&args.native_lib)?});
    library.cuda_set_device(args.device)?;
    let programs=library.programs()?.with_manifest(&args.manifest)?;
    let vision=crate::shared::vision::still::Options { vision:cuteafd_loader::plan::MediaMode::Off,..Default::default() };
    let (mut execution,inputs,admitted,_,_media)=super::startup::whole(&library,&programs,&view,&checkpoint,&cfg,super::startup::Options{
        device:args.device,split_device:args.split_device,peers:&args.peers,manifest:&args.manifest,native_lib:&args.native_lib,
        prefill:PrefillShape{rows:args.prefill_rows as u64,waves:args.prefill_waves},decode_rows:args.decode_rows,sequences:args.max_sequences as u64,
        context_tokens:0,pool_tokens:args.pool_tokens,kv_format:args.kv_format,drafter:Some(args.draft_kv_cache),
        onboard:args.rtx_expert_layers.resolve(cfg.layers),prefix:&args.prefix,
        resident_weights:crate::shared::memory::budget::resident_weight_limit(),expert_usage:false,vision:&vision})?;
    ensure!(inputs.prefix_mark_slots>0,"width oracle has no admitted target mark");
    if args.wide_batch {
        return wide_batch_oracle(&mut execution, &inputs, &admitted, &tokens, args);
    }
    let bank=execution.requests.cache.clone();
    let prefix=super::cache::InklingPrefix::for_execution(&execution);
    let mut placement=Placement{media:None,units:(0..(args.prefill+maximum+usize::from(args.commit_invariance)).div_ceil(64)).map(|page|page as u32).collect(),slot:0,len:0,state_len:0};
    ensure!(placement.units.len()<=bank.layout.pool_pages,"width oracle prefix exceeds admitted target pages");
    prefix.bind_slot(&mut placement,Some(args.prefill+maximum))?;
    let mut units=crate::shared::prefill_pipeline::sequence_units(args.prefill,args.prefill_rows,args.prefill_waves as usize,cfg.layers)?;
    for unit in &mut units{unit.segments=vec![UnitSegment{sequence:0,rows:0..unit.rows.len(),tokens:unit.rows.clone(),prompt_end:false,snapshot:false,head:false}];}
    execution.prefill(std::slice::from_mut(&mut placement),&[&tokens[..args.prefill]],&units,1,super::prefill::HeadRows::All,&mut|_,_,_|Ok(()))?;
    for requests in execution.request_ranks(){requests.cache.capture(0,0)?;requests.cache.drain()?;}
    let saved=placement.clone();
    if args.graph_parity {
        return graph_parity(&mut execution, &saved, &tokens[args.prefill..], args);
    }
    let mut reference=Vec::new();let mut base_logits=Vec::new();
    for row in 0..maximum{
        execution.width_trace=Some(RefCell::new(Trace::default()));
        base_logits.push(decode(&mut execution,&mut placement,&tokens[args.prefill+row..args.prefill+row+1])?);
        reference.push(execution.width_trace.take().unwrap().into_inner());
    }
    std::fs::create_dir_all(&args.save_logits)?;
    let mut passes=Vec::new();let mut clean=true;
    let widths=std::iter::once(1).chain(args.width_invariance.iter().copied().filter(|&width|width>1)).chain(std::iter::once(1));
    for width in widths{
        restore(&execution)?;placement=saved.clone();
        execution.width_trace=Some(RefCell::new(Trace::default()));
        let logits=verify(&mut execution,&mut placement,&tokens[args.prefill..args.prefill+width])?;
        let trace=execution.width_trace.take().unwrap().into_inner();
        let differences=compare(&trace,&reference[..width],&logits,&base_logits[..width],cfg.unpadded_vocab_size)?;
        println!("Inkling width {width}: {}",serde_json::to_string(&differences)?);
        clean&=differences.is_empty();passes.push(serde_json::json!({"width":width,"differences":differences,"logits_sha256":digest(&logits)}));
    }
    let commits=if args.commit_invariance{Some(commit_sweep(&mut execution,&saved,&tokens[args.prefill..],&args.width_invariance)?)}else{None};
    clean&=commits.as_ref().is_none_or(|result|result["pass"]==true);
    let report=serde_json::json!({"commits":commits,"prefill":args.prefill,"pool_tokens":admitted.solved.pool_tokens,"onboard":admitted.onboard,
        "target":"served startup/push/Spark path","reference":"serial width1 decode from exact prompt state","passes":passes,"pass":clean});
    let digests=serde_json::json!({
        "widths": report["passes"].as_array().unwrap().iter().map(|p| serde_json::json!([p["width"],p["logits_sha256"]])).collect::<Vec<_>>(),
        "accepted_prefix": report["commits"]["passes"].as_array().into_iter().flatten()
            .map(|p| serde_json::json!([p["width"],p["accepted"],p["next_logits_sha256"]])).collect::<Vec<_>>()
    });
    std::fs::write(args.save_logits.join("logits-sha256.json"),serde_json::to_vec_pretty(&digests)?)?;
    std::fs::write(args.save_logits.join("width-invariance.json"),serde_json::to_vec_pretty(&report)?)?;
    ensure!(clean,"Inkling width invariance failed; first layer/stage operands in width-invariance.json");Ok(())
}
