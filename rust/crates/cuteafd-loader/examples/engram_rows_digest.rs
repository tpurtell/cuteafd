//! Digest of engram rows gathered through the serving pipeline (prefetch,
//! gather worker, staging) for a fixed pseudo-random token stream: compare
//! two builds for byte-identical gathers. Usage: SNAPSHOT [WAVES].
use anyhow::{Context, Result};
use cuteafd_loader::{read_official_v41_catalog, EngramGatherPoll, EngramPipeline, EngramRequestTokens, EngramTokenMap};
use sha2::{Digest, Sha256};
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let snapshot = Path::new(args.get(1).context("missing snapshot")?);
    let waves: usize = args.get(2).map_or(Ok(24), |w| w.parse())?;
    let catalog = read_official_v41_catalog(snapshot)?;
    let map = EngramTokenMap::from_file(&snapshot.join("tokenizer.json"))?;
    // SAFETY: the checkpoint snapshot is immutable while this process runs.
    let pipeline = unsafe { EngramPipeline::new(&catalog, map, 256, 4, 256 * 64 * 1024)? };
    let mut histories = [pipeline.new_history()?, pipeline.new_history()?];
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % 129_000) as u32
    };
    let mut digest = Sha256::new();
    let mut rows = 0usize;
    let started = std::time::Instant::now();
    for wave_index in 0..waves {
        // Decode-, verify- and prefill-shaped waves of two requests.
        let lengths = [[1, 1], [6, 3], [96, 31]][wave_index % 3];
        let tokens: Vec<Vec<u32>> = lengths.iter().map(|&n| (0..n).map(|_| next()).collect()).collect();
        let requests: Vec<_> = histories.iter().zip(&tokens)
            .map(|(history, token_ids)| EngramRequestTokens { history, token_ids, image_mask: None }).collect();
        let mut wave = pipeline.prepare(&requests)?;
        for layer in 0..2 {
            let view_histories: Vec<_> = histories.iter().collect();
            loop {
                match pipeline.poll(&mut wave, &view_histories, layer)? {
                    EngramGatherPoll::Pending => std::thread::yield_now(),
                    EngramGatherPoll::Cancelled => anyhow::bail!("gather cancelled"),
                    EngramGatherPoll::Ready(lease) => {
                        let view = lease.view()?;
                        digest.update(view.weights);
                        digest.update(view.scales);
                        digest.update(view.text_mask);
                        rows += view.rows * 24;
                        break;
                    }
                }
            }
        }
        let [a, b] = &mut histories;
        wave.commit(&mut [a, b], &lengths)?;
    }
    println!("engram rows {rows} sha256 {:x} in {:.3}s", digest.finalize(), started.elapsed().as_secs_f64());
    Ok(())
}
