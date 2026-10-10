//! Replicated routers and the route-identity check.
//!
//! Under the head split both GPUs hold the same post-attention hidden state,
//! so each runs its own router replica. [`RouteIdentity::check`] compares the
//! two ranks' ids and weights byte for byte once at startup (after graph
//! warm-up, on a fixed prefill plus one decode step). On a mismatch it logs
//! the first layer and row and the engine falls back to
//! [`RouteSource::Broadcast`] (rank 0 pushes ids + weights each MoE layer).
//! `CUTEAFD_ROUTE_CHECK=N` re-checks every N steps (off by default).

/// Where rank 1's routes come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouteSource {
    /// Rank 1 runs its own router replica.
    Replicated,
    /// Rank 0 pushes ids + weights (`rows * topk * 8` bytes) per MoE layer.
    Broadcast,
}

/// One MoE layer's routes from both ranks, as host bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RouteCheck {
    pub layer: usize,
    pub rows: usize,
    pub topk: usize,
    /// `[rows, topk]` U32 ids then `[rows, topk]` FP32 weights, per rank.
    pub ranks: [Vec<u8>; 2],
}

/// The result of comparing replicated routes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RouteIdentity {
    pub layers_checked: usize,
    pub rows_checked: usize,
    /// First `(layer, row)` whose ids or weights differ.
    pub first_mismatch: Option<(usize, usize)>,
}

impl RouteIdentity {
    /// Compares every layer's two ranks; stops at the first differing row.
    pub(crate) fn check(layers: &[RouteCheck]) -> Self {
        let mut result = Self::default();
        for check in layers {
            result.layers_checked += 1;
            result.rows_checked += check.rows;
            let row_bytes = check.topk * 4;
            let ids = check.rows * row_bytes;
            for row in 0..check.rows {
                let span = |bytes: &[u8]| -> (Vec<u8>, Vec<u8>) {
                    (bytes[row * row_bytes..(row + 1) * row_bytes].to_vec(),
                     bytes[ids + row * row_bytes..ids + (row + 1) * row_bytes].to_vec())
                };
                if span(&check.ranks[0]) != span(&check.ranks[1]) {
                    result.first_mismatch = Some((check.layer, row));
                    return result;
                }
            }
        }
        result
    }

    /// The route source the engine should use after this check.
    pub(crate) fn source(&self) -> RouteSource {
        if self.first_mismatch.is_some() { RouteSource::Broadcast } else { RouteSource::Replicated }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(layer: usize, rows: usize, a: Vec<u8>, b: Vec<u8>) -> RouteCheck {
        RouteCheck { layer, rows, topk: 2, ranks: [a, b] }
    }

    #[test]
    fn identical_routes_stay_replicated_and_a_flip_names_layer_and_row() {
        let rows = 3;
        let bytes: Vec<u8> = (0..rows * 2 * 8).map(|i| i as u8).collect();
        let same = RouteIdentity::check(&[check(4, rows, bytes.clone(), bytes.clone())]);
        assert_eq!((same.layers_checked, same.rows_checked, same.first_mismatch), (1, 3, None));
        assert_eq!(same.source(), RouteSource::Replicated);
        let mut weight = bytes.clone();
        // Row 2's first weight (after the ids section).
        weight[rows * 8 + 2 * 8] ^= 1;
        let differ = RouteIdentity::check(&[check(4, rows, bytes.clone(), bytes.clone()), check(5, rows, bytes, weight)]);
        assert_eq!(differ.first_mismatch, Some((5, 2)));
        assert_eq!(differ.source(), RouteSource::Broadcast);
    }
}
