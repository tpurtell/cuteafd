//! Named EXL3 capacity owners and direct bindings from selected export manifests.
use super::{CacheGeometryError, product, sum};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exl3WorkspaceOwner {
    pub name: String,
    pub bytes: u64,
    /// Shared scratch must agree across capacities; physical requests are bytes.
    pub dtype: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exl3WorkspaceBinding {
    pub name: String,
    pub backing: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exl3WorkspaceLayout {
    pub owners: Vec<Exl3WorkspaceOwner>,
    pub bindings: Vec<Exl3WorkspaceBinding>,
}

struct Buffer<'a> {
    bytes: u64,
    dtype: &'a str,
    allocation: &'a str,
    zero: bool,
}

fn invalid(name: &str, what: &str) -> CacheGeometryError {
    CacheGeometryError::ResidentTensor {
        name: format!("EXL3 workspace/{name}"),
        what: what.into(),
    }
}

impl Exl3WorkspaceLayout {
    pub fn from_manifests(manifests: &[Value], fp8_k32: bool, epilogue: Exl3Epilogue) -> Result<Self, CacheGeometryError> {
        if manifests.is_empty() {
            return Err(invalid("manifests", "no capacity specializations"));
        }
        let mut shared = BTreeMap::<String, (u64, String)>::new();
        let mut capacities = BTreeSet::new();
        let mut owners = Vec::new();
        let mut states = Vec::new();
        for manifest in manifests {
            let rows = manifest["capacity"]
                .as_u64()
                .filter(|&n| n > 0)
                .ok_or_else(|| invalid("capacity", "missing or empty row extent"))?;
            if !capacities.insert(rows) {
                return Err(invalid("capacity", "duplicate retained specialization"));
            }
            let prefix = format!("capacity.{rows}");
            let buffers = manifest["buffers"]
                .as_object()
                .ok_or_else(|| invalid("buffers", "missing allocation map"))?;
            let mut state = BTreeMap::new();
            for (name, spec) in buffers {
                if name.is_empty() {
                    return Err(invalid("buffers", "empty buffer name"));
                }
                let buffer = Buffer {
                    bytes: spec["bytes"]
                        .as_u64()
                        .ok_or_else(|| invalid(name, "missing byte extent"))?,
                    dtype: spec["dtype"]
                        .as_str()
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| invalid(name, "missing dtype"))?,
                    allocation: spec["allocation"]
                        .as_str()
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| invalid(name, "missing allocation owner"))?,
                    zero: spec["zero_on_create"]
                        .as_bool()
                        .ok_or_else(|| invalid(name, "missing initialization policy"))?,
                };
                if !epilogue.allocates(buffer.allocation) { continue; }
                if name == buffer.allocation {
                    let bytes = buffer.bytes.max(16);
                    if buffer.zero {
                        owners.push(Exl3WorkspaceOwner {
                            name: format!("{prefix}.storage.{name}"),
                            bytes,
                            dtype: buffer.dtype.into(),
                        });
                    } else {
                        let entry = shared
                            .entry(name.clone())
                            .or_insert((0, buffer.dtype.into()));
                        if entry.1 != buffer.dtype {
                            return Err(invalid(
                                name,
                                "shared capacity arena has conflicting dtypes",
                            ));
                        }
                        entry.0 = entry.0.max(bytes);
                    }
                }
                state.insert(name.as_str(), buffer);
            }
            owners.push(Exl3WorkspaceOwner {
                name: format!("{prefix}.trellis_lut"),
                bytes: manifest["trellis_lut"]["bytes"]
                    .as_u64()
                    .ok_or_else(|| invalid("trellis_lut", "missing asset extent"))?
                    .max(16),
                dtype: "bytes".into(),
            });
            match manifest["input_format"].as_str() {
                Some("e4m3_k32") => (),
                None | Some("bf16") if fp8_k32 => {
                    let hidden = manifest["hidden"]
                        .as_u64()
                        .filter(|&n| n > 0)
                        .ok_or_else(|| invalid("hidden", "missing or empty hidden extent"))?;
                    owners.push(Exl3WorkspaceOwner {
                        name: format!("{prefix}.wire"),
                        bytes: product("EXL3 wire decode", &[rows, hidden, 2])?,
                        dtype: "bytes".into(),
                    });
                }
                None | Some("bf16") => (),
                _ => return Err(invalid("input_format", "unknown wire representation")),
            }
            states.push((prefix, state));
        }
        owners.extend(
            shared
                .iter()
                .map(|(name, (bytes, dtype))| Exl3WorkspaceOwner {
                    name: format!("shared.{name}"),
                    bytes: *bytes,
                    dtype: dtype.clone(),
                }),
        );
        let mut names = BTreeSet::new();
        if owners
            .iter()
            .any(|owner| !names.insert(owner.name.as_str()))
        {
            return Err(invalid("buffers", "ambiguous physical owner identity"));
        }
        let mut bindings = Vec::new();
        for (prefix, state) in states {
            for (&name, buffer) in &state {
                let backing = state
                    .get(buffer.allocation)
                    .ok_or_else(|| invalid(name, "missing allocation owner"))?;
                if backing.allocation != buffer.allocation {
                    return Err(invalid(name, "binding must name a direct allocation owner"));
                }
                let (owner, bytes) = if backing.zero {
                    (
                        format!("{prefix}.storage.{}", buffer.allocation),
                        backing.bytes.max(16),
                    )
                } else {
                    (
                        format!("shared.{}", buffer.allocation),
                        shared[buffer.allocation].0,
                    )
                };
                if buffer.bytes > bytes {
                    return Err(invalid(name, "binding exceeds physical owner"));
                }
                // Empty kernel bindings retain their real floor-sized owner but
                // do not describe a nonempty memory region to reconcile.
                if buffer.bytes != 0 {
                    bindings.push(Exl3WorkspaceBinding {
                        name: format!("{prefix}.view.{name}"),
                        backing: owner,
                        bytes: buffer.bytes,
                    });
                }
            }
        }
        Ok(Self { owners, bindings })
    }

    pub fn bytes(&self) -> Result<u64, CacheGeometryError> {
        sum(
            "EXL3 capacity arenas",
            &self
                .owners
                .iter()
                .map(|owner| owner.bytes)
                .collect::<Vec<_>>(),
        )
    }

    pub fn allocations(
        &self,
        prefix: &str,
        location: crate::placement::planner::Location,
        lifetime: crate::placement::planner::Lifetime,
    ) -> Vec<crate::placement::planner::Allocation> {
        use crate::placement::planner::{Allocation, View};
        use cuteafd_core::memory_layout::Category;
        let allocation = |name: &str, bytes| {
            Allocation::buffer(
                format!("{prefix}.{name}"),
                location,
                bytes,
                lifetime,
                Category::Workspace,
                "EXL3 expert workspace",
            )
        };
        self.owners
            .iter()
            .map(|owner| allocation(&owner.name, owner.bytes))
            .chain(self.bindings.iter().map(|binding| {
                let mut view = allocation(&binding.name, binding.bytes);
                view.view = Some(View {
                    backing: format!("{prefix}.{}", binding.backing),
                    offset: 0,
                });
                view
            }))
            .collect()
    }
}

/// The routed-expert epilogue a family's runtime executes after the EXL3 core.
///
/// The weighted top-k sum is its own native object; it reads the executor's
/// `output` owner. Families whose experts carry a per-expert output norm reply
/// unweighted per-expert partials instead, launch the core only and never read
/// that owner. Planner censuses and worker allocations must declare the same
/// epilogue, or admission and allocation disagree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Exl3Epilogue {
    /// Core then the weighted top-k sum into the token output.
    #[default]
    Sum,
    /// Core only; the coordinator combines per-expert partial replies.
    PartialReplies,
}

/// The weighted token sum's allocation owner.
const OUTPUT_OWNER: &str = "output";

impl Exl3Epilogue {
    /// The epilogue the checkpoint's routed experts declare.
    pub fn for_catalog(catalog: &crate::OfficialV41Catalog) -> Self {
        if catalog.exl3().is_some() && catalog.per_expert_output_norm() {
            Self::PartialReplies
        } else {
            Self::Sum
        }
    }
    /// Whether `owner` is allocated for this epilogue.
    pub fn allocates(self, owner: &str) -> bool {
        self == Self::Sum || owner != OUTPUT_OWNER
    }
}

pub fn exl3_workspace_bytes(manifests: &[Value], fp8_k32: bool, epilogue: Exl3Epilogue)
    -> Result<u64, CacheGeometryError> {
    Exl3WorkspaceLayout::from_manifests(manifests, fp8_k32, epilogue)?.bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn manifest(rows: u64, bytes: u64) -> Value {
        json!({"capacity": rows, "hidden": 8, "trellis_lut": {"bytes": 7}, "buffers": {
            "scratch": {"bytes": bytes, "allocation": "scratch", "dtype": "f16", "zero_on_create": false},
            "alias": {"bytes": 32, "allocation": "scratch", "dtype": "f16", "zero_on_create": false},
            "state": {"bytes": 0, "allocation": "state", "dtype": "i32", "zero_on_create": true},
            "state_alias": {"bytes": 8, "allocation": "state", "dtype": "i32", "zero_on_create": false}}})
    }

    #[test]
    fn aliases_share_scratch_but_capacity_state_and_wire_decode_stay_private() {
        let manifests = [manifest(1, 64), manifest(16, 1024)];
        let layout = Exl3WorkspaceLayout::from_manifests(&manifests, true, Exl3Epilogue::Sum).unwrap();
        assert_eq!(layout.bytes().unwrap(), 1024 + 2 * (16 + 16) + 17 * 8 * 2);
        assert_eq!(
            exl3_workspace_bytes(&manifests, false, Exl3Epilogue::Sum).unwrap(),
            1024 + 2 * (16 + 16)
        );
        assert_eq!(layout.owners.len(), 7);
        assert_eq!(layout.bindings.len(), 6);
        for binding in &layout.bindings {
            if binding.name.ends_with(".state_alias") {
                assert!(binding.backing.ends_with(".storage.state"));
                assert_eq!(binding.bytes, 8);
            } else {
                assert_eq!(binding.backing, "shared.scratch");
            }
        }
        assert!(
            layout
                .owners
                .iter()
                .any(|owner| owner.name == "capacity.1.storage.state" && owner.bytes == 16)
        );
        assert!(
            layout
                .owners
                .iter()
                .any(|owner| owner.name == "capacity.16.wire" && owner.bytes == 256)
        );
        let mut native_wire = manifests.clone();
        native_wire[1]["input_format"] = json!("e4m3_k32");
        assert_eq!(
            exl3_workspace_bytes(&native_wire, true, Exl3Epilogue::Sum).unwrap(),
            layout.bytes().unwrap() - 256
        );
        // A binding can use the maximum physical backing shared by capacities,
        // exactly as the execution constructor's bounds check permits.
        let mut larger_alias = manifests.clone();
        larger_alias[0]["buffers"]["alias"]["bytes"] = json!(512);
        Exl3WorkspaceLayout::from_manifests(&larger_alias, true, Exl3Epilogue::Sum).unwrap();
    }

    #[test]
    fn partial_replies_omit_output_owner_and_bindings() {
        let manifests = [manifest(1, 64), manifest(16, 1024)];
        let mut output = manifests.clone();
        for (manifest, bytes) in output.iter_mut().zip([64, 1024]) {
            manifest["buffers"]["output"] = json!({"bytes": bytes, "allocation": "output",
                "dtype": "bf16", "zero_on_create": false});
            manifest["buffers"]["output_alias"] = json!({"bytes": 32, "allocation": "output",
                "dtype": "bf16", "zero_on_create": false});
        }
        let baseline = Exl3WorkspaceLayout::from_manifests(&manifests, true, Exl3Epilogue::Sum).unwrap();
        let partial = Exl3WorkspaceLayout::from_manifests(&output, true, Exl3Epilogue::PartialReplies).unwrap();
        assert_eq!(partial, baseline);
        let summed = Exl3WorkspaceLayout::from_manifests(&output, true, Exl3Epilogue::Sum).unwrap();
        assert_eq!(summed.bytes().unwrap(), baseline.bytes().unwrap() + 1024);
        assert_eq!(summed.owners.len(), baseline.owners.len() + 1);
        assert_eq!(summed.bindings.len(), baseline.bindings.len() + 4);
    }

    #[test]
    fn malformed_capacity_owners_and_unbounded_bindings_refuse() {
        let originals = [manifest(1, 64), manifest(16, 1024)];
        let refuses = |manifests: &[Value]| {
            assert!(Exl3WorkspaceLayout::from_manifests(manifests, true, Exl3Epilogue::Sum).is_err())
        };
        refuses(&[]);
        refuses(&[originals[0].clone(), originals[0].clone()]);
        for (field, value) in [("allocation", json!("missing")), ("bytes", json!(1025))] {
            let mut bad = originals.clone();
            bad[0]["buffers"]["alias"][field] = value;
            refuses(&bad);
        }
        let mut bad = originals.clone();
        bad[1]["buffers"]["scratch"]["dtype"] = json!("f32");
        refuses(&bad);
        let mut bad = originals.clone();
        bad[0]["buffers"]["alias"]["allocation"] = json!("alias");
        bad[0]["buffers"]["scratch"]["allocation"] = json!("alias");
        bad[0]["buffers"]["state"]["allocation"] = json!("scratch");
        refuses(&bad);
        let mut bad = originals.clone();
        bad[0]["hidden"] = json!(u64::MAX);
        refuses(&bad);
        let mut bad = originals.clone();
        bad[0]["buffers"]["state_alias"]["bytes"] = json!(17);
        refuses(&bad);
        let mut bad = originals.clone();
        bad[0]["input_format"] = json!("unknown");
        refuses(&bad);
    }
}
