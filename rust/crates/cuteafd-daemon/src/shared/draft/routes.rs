//! One round's routed expert ids, per layer and verifier row.
//!
//! Families fill a [`RoundRoutes`] where the ids already are on the host: Spark
//! layers copy them from the per-layer request staging before it is reused
//! ([`push`](RoundRoutes::push)), local layers decode a device ring after the
//! round's sync, and V4.1 converts its lane capture once per round
//! ([`fill`](RoundRoutes::fill)). Rows are the step's verifier rows in request
//! order; the policy keeps each request's accepted-input prefix as committed
//! history (`offset..offset + accepted`, then `offset += rows`).
use thiserror::Error;

/// Why a round's routes cannot be observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub(crate) enum RouteError {
    #[error("route layer {0} outside the geometry")]
    Layer(usize),
    #[error("route layer {layer} has {found} ids, expected {expected}")]
    Extent { layer: usize, found: usize, expected: usize },
    #[error("route layer {0} missing from the round")]
    Missing(usize),
    #[error("route id {0} exceeds u16")]
    Id(u32),
}

/// Expert ids `[layer][row][slot]` of one round, as `u16` after the family's
/// flag mask. Storage is reused across rounds.
#[derive(Clone, Debug)]
pub(crate) struct RoundRoutes {
    layers: usize,
    topk: usize,
    rows: usize,
    ids: Vec<u16>,
    filled: Vec<bool>,
    /// Layers with no routed experts this round (their ids stay zero).
    dense: Vec<bool>,
}

impl RoundRoutes {
    pub fn new(layers: usize, topk: usize) -> Self {
        Self { layers, topk, rows: 0, ids: Vec::new(), filled: vec![false; layers],
            dense: vec![false; layers] }
    }
    pub fn layers(&self) -> usize { self.layers }
    pub fn topk(&self) -> usize { self.topk }
    pub fn rows(&self) -> usize { self.rows }
    /// Whether `layer` is dense (unrouted) this round.
    pub fn is_dense(&self, layer: usize) -> bool { self.dense.get(layer).copied().unwrap_or(false) }
    /// Whether `layer` was filled this round.
    pub fn is_filled(&self, layer: usize) -> bool { self.filled.get(layer).copied().unwrap_or(false) }
    /// Start a round of `rows` verifier rows; every layer is unfilled.
    pub fn begin(&mut self, rows: usize) {
        self.rows = rows;
        self.ids.clear();
        self.ids.resize(self.layers * rows * self.topk, 0);
        self.filled.fill(false);
        self.dense.fill(false);
    }
    /// One layer's ids, `rows * topk` in row order, masked by `mask` (the
    /// family's expert-index bits; router flag bits above them are dropped).
    /// Spark layers copy them here from the request staging before reuse.
    #[allow(dead_code, reason = "D2: GLM Flash pushes staged Spark routes per layer")]
    pub fn push(&mut self, layer: usize, ids: &[u32], mask: u32) -> Result<(), RouteError> {
        let slot = self.slot(layer, ids.len())?;
        for (out, &id) in slot.iter_mut().zip(ids) { *out = narrow(id & mask)?; }
        self.filled[layer] = true;
        Ok(())
    }
    /// Copy an already validated owner probe's u16 layer without widening it.
    pub fn push_narrow(&mut self, layer: usize, ids: &[u16]) -> Result<(), RouteError> {
        self.slot(layer, ids.len())?.copy_from_slice(ids);
        self.filled[layer] = true;
        Ok(())
    }
    /// As [`push`](Self::push), from little-endian u32 pinned staging bytes.
    pub fn push_bytes(&mut self, layer: usize, bytes: &[u8], mask: u32) -> Result<(), RouteError> {
        if bytes.len() % 4 != 0 {
            return Err(RouteError::Extent { layer, found: bytes.len(), expected: self.rows * self.topk * 4 });
        }
        let slot = self.slot(layer, bytes.len() / 4)?;
        for (out, word) in slot.iter_mut().zip(bytes.chunks_exact(4)) {
            let id = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
            *out = narrow(id & mask)?;
        }
        self.filled[layer] = true;
        Ok(())
    }
    /// As [`push`](Self::push), from `[row][slot]` arrays of `K == topk` ids.
    pub fn push_rows<const K: usize>(&mut self, layer: usize, rows: &[[u32; K]], mask: u32)
        -> Result<(), RouteError> {
        let slot = self.slot(layer, rows.len() * K)?;
        for (out, &id) in slot.iter_mut().zip(rows.iter().flatten()) { *out = narrow(id & mask)?; }
        self.filled[layer] = true;
        Ok(())
    }
    /// A dense (unrouted) layer this round: its ids stay zero and the policy
    /// never prices it (its geometry class is `None`).
    #[allow(dead_code, reason = "D2: GLM Flash's leading dense layers")]
    pub fn dense(&mut self, layer: usize) -> Result<(), RouteError> {
        if layer >= self.layers { return Err(RouteError::Layer(layer)); }
        self.filled[layer] = true;
        self.dense[layer] = true;
        Ok(())
    }
    /// Replace the round with a whole `[layer][row][slot]` capture of `rows`
    /// rows; every layer must be present with exactly `rows` rows.
    pub fn fill<const K: usize>(&mut self, capture: &[Vec<[u32; K]>], rows: usize, mask: u32)
        -> Result<(), RouteError> {
        if capture.len() != self.layers { return Err(RouteError::Missing(capture.len().min(self.layers))); }
        self.begin(rows);
        for (layer, routes) in capture.iter().enumerate() { self.push_rows(layer, routes, mask)?; }
        Ok(())
    }
    /// As [`fill`](Self::fill), but a layer whose capture is empty was not
    /// captured this round and stays unfilled (V4.1 prefill and partial
    /// captures). Filled layers must still have exactly `rows` rows.
    pub fn fill_partial<const K: usize>(&mut self, capture: &[Vec<[u32; K]>], rows: usize, mask: u32)
        -> Result<(), RouteError> {
        if capture.len() != self.layers { return Err(RouteError::Missing(capture.len().min(self.layers))); }
        self.begin(rows);
        for (layer, routes) in capture.iter().enumerate() {
            if routes.is_empty() { continue; }
            self.push_rows(layer, routes, mask)?;
        }
        Ok(())
    }
    fn slot(&mut self, layer: usize, len: usize) -> Result<&mut [u16], RouteError> {
        if layer >= self.layers { return Err(RouteError::Layer(layer)); }
        let expected = self.rows * self.topk;
        if len != expected || self.topk == 0 && len != 0 {
            return Err(RouteError::Extent { layer, found: len, expected });
        }
        Ok(&mut self.ids[layer * expected..][..expected])
    }
    /// Every layer was filled this round.
    pub fn complete(&self) -> Result<(), RouteError> {
        match self.filled.iter().position(|filled| !filled) {
            Some(layer) => Err(RouteError::Missing(layer)),
            None => Ok(()),
        }
    }
    /// Layer-major ids for `DraftRoundObservation::routes`.
    pub fn flat(&self) -> &[u16] { &self.ids }
}

fn narrow(id: u32) -> Result<u16, RouteError> { u16::try_from(id).map_err(|_| RouteError::Id(id)) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_and_narrow_routes_match_wide_routes() {
        let mut routes = RoundRoutes::new(2, 2);
        routes.begin(2);
        let bytes: Vec<_> = [1u32, 2, 512 | 3, 4].into_iter().flat_map(u32::to_le_bytes).collect();
        routes.push_bytes(0, &bytes, 511).unwrap();
        routes.push_narrow(1, &[5, 6, 7, 8]).unwrap();
        routes.complete().unwrap();
        assert_eq!(routes.flat(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(matches!(routes.push_bytes(0, &[0; 3], u32::MAX), Err(RouteError::Extent { .. })));
        assert!(matches!(routes.push_narrow(1, &[0; 3]), Err(RouteError::Extent { .. })));
        let bytes: Vec<_> = [70_000u32, 1, 2, 3].into_iter().flat_map(u32::to_le_bytes).collect();
        assert_eq!(routes.push_bytes(0, &bytes, u32::MAX), Err(RouteError::Id(70_000)));
    }

    #[test]
    fn layer_major_ids_with_flags_masked() {
        let mut routes = RoundRoutes::new(2, 3);
        routes.begin(2);
        assert_eq!(routes.complete(), Err(RouteError::Missing(0)));
        routes.push(1, &[1, 2, 3, 512 | 4, 5, 6], 511).unwrap();
        routes.push_rows(0, &[[7, 8, 9], [10, 11, 1024 | 12]], 511).unwrap();
        routes.complete().unwrap();
        assert_eq!(routes.flat(), [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]);
        // The whole-capture form matches the per-layer pushes.
        let mut whole = RoundRoutes::new(2, 3);
        whole.fill(&[vec![[7, 8, 9], [10, 11, 12]], vec![[1, 2, 3], [4, 5, 6]]], 2, 511).unwrap();
        assert_eq!(whole.flat(), routes.flat());
        // A dense layer completes the round with zero ids.
        let mut dense = RoundRoutes::new(2, 3);
        dense.begin(2);
        dense.dense(0).unwrap();
        dense.push(1, &[1, 2, 3, 4, 5, 6], 511).unwrap();
        dense.complete().unwrap();
        assert_eq!(dense.flat()[..6], [0; 6]);
        assert_eq!(dense.dense(2), Err(RouteError::Layer(2)));
        // A new round clears every layer.
        routes.begin(1);
        assert_eq!((routes.complete(), routes.flat().len()), (Err(RouteError::Missing(0)), 6));
    }

    #[test]
    fn partial_capture_leaves_empty_layers_unfilled() {
        let mut routes = RoundRoutes::new(3, 6);
        routes.fill_partial(&[vec![], vec![[1u32; 6], [2; 6]], vec![]], 2, u32::MAX).unwrap();
        assert_eq!(routes.rows(), 2);
        assert!(!routes.is_filled(0) && routes.is_filled(1) && !routes.is_filled(2));
        assert_eq!(&routes.flat()[12..24], &[1; 6].iter().chain(&[2; 6]).copied().collect::<Vec<_>>()[..]);
        // A filled layer with the wrong row count is still rejected.
        assert!(routes.fill_partial(&[vec![], vec![[1u32; 6]], vec![]], 2, u32::MAX).is_err());
    }

    #[test]
    fn geometry_and_dense_layers_are_readable() {
        let mut routes = RoundRoutes::new(3, 8);
        assert_eq!((routes.layers(), routes.topk(), routes.rows()), (3, 8, 0));
        routes.begin(2);
        routes.dense(0).unwrap();
        routes.push(1, &[0; 16], 511).unwrap();
        assert!(routes.is_dense(0) && routes.is_filled(0));
        assert!(!routes.is_dense(1) && routes.is_filled(1));
        assert!(!routes.is_dense(2) && !routes.is_filled(2));
        // A new round clears the dense marks too.
        routes.begin(1);
        assert!(!routes.is_dense(0) && !routes.is_filled(0));
    }

    #[test]
    fn rejects_shapes_that_do_not_match_the_round() {
        let mut routes = RoundRoutes::new(2, 3);
        routes.begin(2);
        assert_eq!(routes.push(2, &[0; 6], u32::MAX), Err(RouteError::Layer(2)));
        assert_eq!(routes.push(0, &[0; 5], u32::MAX), Err(RouteError::Extent { layer: 0, found: 5, expected: 6 }));
        assert_eq!(routes.push(0, &[70_000; 6], u32::MAX), Err(RouteError::Id(70_000)));
        assert!(routes.fill(&[vec![[0u32; 3]; 2]], 2, 511).is_err());
        assert!(routes.fill(&[vec![[0u32; 3]; 2], vec![[0u32; 3]; 1]], 2, 511).is_err());
    }
}
