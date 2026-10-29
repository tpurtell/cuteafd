//! Structural coverage of the two graph-warming executors' peer generations.

fn body<'a>(source: &'a str, first: &str, next: &str) -> &'a str {
    source.split_once(first).unwrap().1.split_once(next).unwrap().0
}

#[test]
fn masked_warming_has_matching_pushes_and_waits_on_every_ffn_path() {
    let engine = include_str!("../engine.rs");
    let layers = include_str!("layers.rs");
    let core = include_str!("core.rs");
    let peer = body(engine, "fn peer_attention(", "fn peer_post(");
    assert!(peer.contains("(true, true) => self.ffn_on(1"));
    let masks_shared = peer.contains("if !layer.dense && self.startup_graphs && self.capturing.get() { return Ok(()); }");
    assert!(peer.contains("if self.tp2_layer(index) || self.unit_prefill.get() { return Ok(()) }"));
    let graph = body(engine, "fn decode_graphed(", "/// The vocabulary projection");
    let masked = body(graph, "if self.warming_graphs.get() {", "} else if self.startup_graphs {");
    assert!(!masked.contains("self.ffn(") && !masked.contains("self.tp2_pair("));
    assert!(masked.contains("cuda_zero_bytes_async(w.delta.buffer"));
    assert!(masked.contains("if self.tp2_layer(index) { w1.delta.buffer } else { w1.shared.buffer }"));
    assert!(graph.contains("if layer.dense {\n                    self.ffn("));
    assert!(graph.contains("self.tp2_publish(index, 0, t, w1)?"));
    let wrapper = body(engine, "fn peer_segment(", "fn decode_primitive(");
    assert!(wrapper.contains("if self.startup_graphs && !layer.dense && !self.tp2_layer(index)"));
    let warm_wrapper = body(wrapper, "if self.warming_graphs.get() {", "} else {");
    assert!(!warm_wrapper.contains("self.ffn_on("));
    assert!(warm_wrapper.contains("cuda_zero_bytes_async(w1.shared.buffer"));
    let begin = body(layers, "fn begin_ffn(", "fn finish_ffn(");
    assert!(begin.contains("let masked = self.graphed && e.warming_graphs.get() && !layer.dense;"));
    assert!(begin.contains("if !masked && !e.tp2_layer(index)"));
    assert!(begin.contains("if masked {") && begin.contains("} else if layer.dense {"));
    let finish = body(layers, "fn finish_ffn(", "#[cfg(test)]");
    let native = body(core, "fn core_ffn(", "fn core_kda(");
    let core_push = native.matches("self.core_publish(").count();
    let core_wait = native.matches("self.exchange_wait(").count();
    assert_eq!((core_push, core_wait), (1, 1));
    let meet = body(engine, "fn meet_ffn(", "fn peer_ffn_out(");
    assert!(meet.contains("if index + 1 < layers {"));
    let post = body(engine, "fn peer_post(", "fn peer_segment(");
    assert!(post.contains("let Some(next) = peer.layers.get(index + 1) else { return Ok(()) };"));
    for placement_layers in [false, true] {
        for tp2 in [false, true] {
            for dense in [false, true] {
                for native in [false, true] {
                    for last in [false, true] {
                        for peer_present in [false, true] {
                            let mut pushes = [0; 2];
                            let mut waits = [0; 2];
                            if peer_present {
                                // The old captured peer path adds a native generation only on rank 1.
                                for (rank, active) in [dense, dense || (!placement_layers && !tp2 && !masks_shared)].into_iter().enumerate() {
                                    if native && active { pushes[rank] += core_push; waits[rank] += core_wait; }
                                }
                                pushes[1] += if placement_layers {
                                    finish.matches("e.exchange_push(").count()
                                } else if tp2 {
                                    body(engine, "fn tp2_publish(", "fn router_on(").matches("self.exchange_push(").count()
                                } else if !dense && masks_shared {
                                    wrapper.matches("self.exchange_push(").count()
                                } else { peer.matches("self.exchange_push(").count() - 1 }; // remove attention
                                waits[0] += meet.matches("self.exchange_wait(").count();
                                if !last {
                                    pushes[0] += meet.matches("self.exchange_push(").count();
                                    waits[1] += post.matches("self.exchange_wait(").count();
                                }
                            }
                            assert_eq!(pushes[0], waits[1], "layers={placement_layers} tp2={tp2} dense={dense} native={native} last={last} peer={peer_present}");
                            assert_eq!(pushes[1], waits[0]);
                            let cores = usize::from(peer_present && dense && native);
                            assert_eq!(pushes, [cores + usize::from(peer_present && !last), cores + usize::from(peer_present)]);
                            assert_eq!(waits, [cores + usize::from(peer_present), cores + usize::from(peer_present && !last)]);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn captured_replay_plus_wrapper_matches_eager_before_consumers() {
    let source = include_str!("../engine.rs");
    let peer = body(source, "fn peer_attention(", "fn peer_post(");
    let guard = peer.find("if !layer.dense && self.startup_graphs && self.capturing.get() { return Ok(()); }").unwrap();
    assert!(guard < peer.find("self.ffn_on(1").unwrap());
    let wrapper = body(source, "fn peer_segment(", "fn decode_primitive(");
    assert!(wrapper.find("self.peer_attention(").unwrap() < wrapper.find("})?;").unwrap());
    assert!(wrapper.find("})?;").unwrap() < wrapper.find("let layer = &self.peer()?.layers[index]").unwrap());
    assert!(wrapper.find("self.ffn_on(1").unwrap() < wrapper.find("self.exchange_push(").unwrap());
    assert!(wrapper.contains("Scalar::I32(real as i32)") && wrapper.contains("if real < t"));
    // The same segment's captured attention does not read shared/delta; the following
    // segment's peer_post reads them, after this synchronous host wrapper queued its write.
    let attention = peer.split("let out = self.peer_ffn_out").next().unwrap();
    assert!(!attention.contains("w1.shared"));
    let consumer = body(source, "fn peer_post(", "fn peer_segment(");
    assert!(consumer.find("self.exchange_wait(").unwrap() < consumer.find("self.peer_ffn_out(").unwrap());
    let graph = body(source, "fn decode_graphed(", "/// The vocabulary projection");
    assert!(graph.find("self.peer_segment(0,").unwrap() < graph.find("self.peer_segment(index + 1,").unwrap());
    assert!(graph.find("self.peer_segment(index + 1,").unwrap() < graph.find("real_row_moe(").unwrap());
    for native in [false, true] {
        for tp2 in [false, true] {
            for dense in [false, true] {
                for last in [false, true] {
                    // C = retained native core exchange; B = ordinary BF16 publication.
                    let core = usize::from(native);
                    let eager_push = [core + usize::from(!last), core + 1];
                    let eager_wait = [core + 1, core + usize::from(!last)];
                    let deferred = !dense && !tp2;
                    let captured_core = if deferred { 0 } else { core };
                    let wrapper_core = if deferred { core } else { 0 };
                    let captured_boundary = usize::from(!deferred);
                    let wrapper_boundary = usize::from(deferred);
                    let replay_push = [core + usize::from(!last), captured_core + wrapper_core + captured_boundary + wrapper_boundary];
                    let replay_wait = [core + 1, captured_core + wrapper_core + usize::from(!last)];
                    assert_eq!(replay_push, eager_push);
                    assert_eq!(replay_wait, eager_wait);
                    assert_eq!(replay_push[0], replay_wait[1]);
                    assert_eq!(replay_push[1], replay_wait[0]);
                }
            }
        }
    }
}

#[test]
fn native_mla_trace_reads_live_stages_with_rank_local_extents() {
    let source = include_str!("../engine.rs");
    let trace = body(source, "fn mla_over(", "fn moe(");
    let native = body(trace, "if native {", "} else {");
    assert!(native.contains("slot.stage(\"q\")?"));
    assert!(native.contains("slot.stage(\"k\")?"));
    assert!(!native.contains("w.scratch"));
    assert!(trace.contains("?.stage(\"values\")?"));
    assert!(trace.contains("t * heads * self.cfg.v_head_dim * 2"));
    assert!(!trace.contains("t * self.cfg.heads"));
}
