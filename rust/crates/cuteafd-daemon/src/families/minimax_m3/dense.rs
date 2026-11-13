//! One caller-admitted serial projection workspace per lane or prefill owner.
use crate::shared::memory::{device::{Allocation, Device}, receipts::Inventory};
use anyhow::{ensure, Result};
use cuteafd_ffi::{native_dense::Mxfp8A8, programs::Programs, CuteafdDeviceBuffer};
use super::weights::TargetWeights;
use std::ffi::c_void;
use std::{collections::BTreeMap, rc::Rc};
use cuteafd_core::memory_layout::Tag;
use cuteafd_loader::families::minimax_m3::weight_layout::{CoordinatorWeights, DenseInput, DenseWorkspace};

/// Loaded handles borrow the installed program table; no weights or writable
/// workspaces are allocated here. Each physical device binds its own handles.
pub(crate) struct NativePrograms<'a> {
    pub device: Device<'a>,
    pub linears: BTreeMap<String, Mxfp8A8<'a>>,
}

impl<'a> NativePrograms<'a> {
    pub fn bind(device: Device<'a>, programs: &'a Programs<'a>, weights: &CoordinatorWeights,
        manifest: &serde_json::Value, rows: u32) -> Result<Self> {
        ensure!(std::ptr::eq(device.library, programs.library()), "MiniMax dense program library differs from device");
        weights.dense_workspace(rows)?;
        let specs = weights.resolve_dense_programs(manifest, rows)?;
        // Validate every selected entry before loading any module. A separately
        // validated manifest is not proof that this library has the same ABI.
        for expected in &specs {
            let actual = programs.spec(&expected.name)?;
            ensure!(actual.pointers == expected.pointers && actual.scalar_kinds == expected.scalar_kinds
                && actual.scratch == expected.scratch && actual.capacity_rows == expected.capacity_rows
                && actual.geometry == expected.geometry,
                "MiniMax installed program {} differs from admitted manifest", expected.name);
        }
        device.run(|| {
            let mut linears = BTreeMap::new();
            for projection in &weights.projections {
                let name = projection.program_name();
                if !linears.contains_key(&name) {
                    let quantizer = format!("exlr8_dense_quant_k{}_bf16_a8", projection.input);
                    let linear = Mxfp8A8::bind(programs, &name, &quantizer)?;
                    ensure!(linear.geometry() == projection.geometry(), "MiniMax bound dense geometry differs");
                    linears.insert(name, linear);
                }
            }
            Ok(Self { device, linears })
        })
    }
}

pub(crate) struct Workspace<'a> {
    pub layout: DenseWorkspace,
    pub inputs: Allocation<'a>,
    pub records: Allocation<'a>,
    pub output: Allocation<'a>,
}

impl<'a> Workspace<'a> {
    /// Admission must include these three physical requests and their tails.
    /// Inputs are a one-entry BF16 pointer table, not an activation conversion.
    pub fn new(device: Device<'a>, weights: &CoordinatorWeights, rows: u32) -> Result<Self> {
        let layout = weights.dense_workspace(rows)?;
        let records_bytes = usize::try_from(layout.records)?;
        let output_bytes = usize::try_from(layout.output)?;
        let inputs = Allocation::new(device, 8, Tag::workspace("minimax/dense/inputs"))?;
        let records = Allocation::new(device, records_bytes, Tag::workspace("minimax/dense/records"))?;
        let output = Allocation::new(device, output_bytes, Tag::workspace("minimax/dense/output"))?;
        Ok(Self { layout, inputs, records, output })
    }

    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, allocation) in [("inputs", &self.inputs), ("records", &self.records), ("output", &self.output)] {
            allocation.inventory(inventory, &format!("{prefix}.{name}"))?;
        }
        Ok(())
    }
}

/// One independently live lane/unit: admit both input and workspace requests
/// first. All producers and consumers use this same device and ordered stream.
pub(crate) struct ExecutionOwner<'a> {
    pub workspace: Workspace<'a>,
    pub input_layout: DenseInput,
    pub activation: Rc<Allocation<'a>>,
}

impl<'a> ExecutionOwner<'a> {
    pub fn new(device: Device<'a>, weights: &CoordinatorWeights, rows: u32) -> Result<Self> {
        let input_layout = weights.dense_input(rows)?;
        weights.dense_workspace(rows)?;
        let activation = Rc::new(Allocation::new(device, usize::try_from(input_layout.bytes)?,
            Tag::workspace("minimax/dense/activation"))?);
        let workspace = Workspace::new(device, weights, rows)?;
        // Fresh storage only, before capture or enqueue: no per-call host table
        // uploads, nor copies from a second producer destination, are required.
        device.run(|| device.library.copy_h2d(workspace.inputs.buffer,
            &(activation.buffer.ptr as u64).to_le_bytes()))?;
        Ok(Self { workspace, input_layout, activation })
    }

    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        self.workspace.inventory(inventory, prefix)?;
        self.activation.inventory(inventory, &format!("{prefix}.activation"))
    }

    /// The producer writes packed BF16 rows here, at the selected projection's
    /// width. It need not produce into another allocation and copy afterward.
    pub fn producer_destination(&self) -> CuteafdDeviceBuffer { self.activation.buffer }

    pub fn prepare<'o, 'p>(&'o mut self, programs: &'p NativePrograms<'a>,
        projection: &cuteafd_loader::families::minimax_m3::weight_layout::Mxfp8Projection,
        rows: u32, stream: *mut c_void) -> Result<Prepared<'o, 'p, 'a>> {
        ensure!((1..=self.input_layout.rows).contains(&rows), "MiniMax rows exceed execution owner");
        ensure!(projection.input as u64 <= self.input_layout.width, "MiniMax input exceeds execution owner");
        let device = self.activation.device;
        ensure!(device.id == programs.device.id && std::ptr::eq(device.library, programs.device.library),
            "MiniMax dense programs differ from execution owner device/library");
        let linear = programs.linears.get(&projection.program_name())
            .ok_or_else(|| anyhow::anyhow!("MiniMax unbound dense projection"))?;
        ensure!(linear.geometry() == projection.geometry(), "MiniMax prepared projection geometry differs");
        device.run(|| Ok(linear.quantize(rows, self.workspace.inputs.buffer,
            self.activation.buffer, self.workspace.records.buffer, stream)?))?;
        Ok(Prepared { owner: self, programs, width: projection.input, rows, stream })
    }
}

/// Borrowing the execution owner prevents another quantization from replacing
/// records between sibling projections. All enqueues stay on its ordered stream.
pub(crate) struct Prepared<'o, 'p, 'a> {
    owner: &'o mut ExecutionOwner<'a>,
    programs: &'p NativePrograms<'a>,
    width: usize,
    rows: u32,
    stream: *mut c_void,
}

impl<'o, 'p, 'a> Prepared<'o, 'p, 'a> {
    /// The result borrows this prepared owner exclusively. A sibling cannot
    /// overwrite its F32 temporary until the result is consumed or discarded.
    pub fn project<'r>(&'r mut self, weights: &TargetWeights<'_>,
        projection: &cuteafd_loader::families::minimax_m3::weight_layout::Mxfp8Projection)
        -> Result<ProjectionOutput<'r, 'o, 'p, 'a>> {
        ensure!(projection.input == self.width, "MiniMax prepared input width differs");
        let linear = self.programs.linears.get(&projection.program_name())
            .ok_or_else(|| anyhow::anyhow!("MiniMax unbound dense projection"))?;
        ensure!(linear.geometry() == projection.geometry(), "MiniMax projection geometry differs");
        let payload = weights.operand(&format!("{}.weight", projection.stem))?;
        let scales = weights.operand(&format!("{}.weight_scale_inv", projection.stem))?;
        let device = self.programs.device;
        for operand in [payload, scales] {
            ensure!(std::ptr::eq(device.library, operand.device.library), "MiniMax weight library differs");
        }
        device.run(|| Ok(linear.project(self.rows, self.owner.workspace.inputs.buffer,
            payload.buffer, scales.buffer, self.owner.workspace.records.buffer,
            self.owner.workspace.output.buffer, self.stream)?))?;
        Ok(ProjectionOutput { prepared: self, linear })
    }
}

/// No raw-buffer accessor or Copy implementation: retaining an output retains
/// the mutable Prepared borrow, so another projection cannot overwrite it.
#[must_use = "consume the projection into an admitted BF16 destination or explicitly discard it"]
pub(crate) struct ProjectionOutput<'r, 'o, 'p, 'a> {
    prepared: &'r mut Prepared<'o, 'p, 'a>,
    linear: &'p Mxfp8A8<'a>,
}

impl ProjectionOutput<'_, '_, '_, '_> {
    /// MiniMax reference project() narrows gate/up separately before SwiGLU.
    /// This enqueues the existing BF16-RNE conversion, not a new epilogue.
    pub fn narrow(self, destination: &Allocation<'_>) -> Result<()> {
        let prepared = self.prepared;
        let device = prepared.programs.device;
        ensure!(std::ptr::eq(device.library, destination.device.library),
            "MiniMax retained destination library differs");
        device.run(|| Ok(self.linear.narrow(prepared.rows, prepared.owner.workspace.output.buffer,
            destination.buffer, prepared.stream)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::{families::minimax_m3::weight_layout::Mxfp8Projection,
        placement::planner::{ComponentKind, CoordinatorDevices, Lifetime}, serving_capacity::WeightArenaLayout};
    use cuteafd_core::memory_layout::Category;

    fn weights() -> Result<CoordinatorWeights> {
        Ok(CoordinatorWeights { rank: 0, layers: vec![0],
            arena: WeightArenaLayout::new("fixture", [("unused".into(), 256)])?,
            projections: vec![Mxfp8Projection { stem: "fixture".into(), input: 128, output: 32 }] })
    }

    #[test]
    fn serial_workspace_receipts_match_admitted_requests_and_restore_device() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.strict_device_free(&library, true)?;
        fixture.configure_memory(&library, 1, 1 << 20)?;
        let device = Device { library: &library, id: 1 };
        let weights = weights()?;
        let workspace = Workspace::new(device, &weights, 17)?;
        assert_eq!(workspace.layout, DenseWorkspace { rows: 17, records: 2 * 4 * 528, output: 17 * 32 * 4 });
        assert_eq!(library.cuda_get_device()?, 0);
        let devices = [1];
        let mut inventory = Inventory::new(CoordinatorDevices::new(&devices)?, "minimax", Lifetime::TARGET,
            Category::Workspace, "native serial dense projection");
        workspace.inventory(&mut inventory, "dense")?;
        assert_eq!(inventory.receipts.len(), 3);
        let expected = workspace.layout.allocations("minimax.dense", 0, Lifetime::TARGET);
        for (actual, expected) in inventory.receipts.iter().zip(expected) {
            assert_eq!(actual.allocation, expected);
            assert!(actual.address != 0);
        }
        assert_eq!(workspace.inputs.buffer.bytes, 8);
        assert_eq!(fixture.events()?.matches('D').count(), 3);
        let bytes = 8 + workspace.layout.records as usize + workspace.layout.output as usize;
        assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, (1 << 20) - bytes);
        // Distinct live owners must not share their writable quantization records.
        let other = Workspace::new(device, &weights, 17)?;
        assert_ne!(workspace.records.buffer.ptr, other.records.buffer.ptr);
        drop(other);
        drop(workspace);
        assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, 1 << 20);
        assert_eq!(fixture.events()?.matches('d').count(), 6);
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        let wide = weights.dense_workspace(4096)?;
        let components = wide.components("dense", 0, Lifetime::TARGET, 32)?;
        assert_eq!(components.iter().filter(|c| matches!(c.kind, ComponentKind::Allocation { .. })).count(), 3);
        Ok(())
    }

    #[test]
    fn stable_input_table_names_one_direct_producer_destination_and_one_charge() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.strict_device_free(&library, true)?;
        fixture.configure_memory(&library, 1, 1 << 20)?;
        let device = Device { library: &library, id: 1 };
        let weights = weights()?;
        let owner = ExecutionOwner::new(device, &weights, 17)?;
        assert_eq!(owner.input_layout, DenseInput { rows: 17, width: 128, bytes: 17 * 128 * 2 });
        let mut table = [0; 8];
        device.run(|| library.copy_d2h(&mut table, owner.workspace.inputs.buffer))?;
        assert_eq!(u64::from_le_bytes(table), owner.activation.buffer.ptr as u64);
        let devices = [1];
        let mut inventory = Inventory::new(CoordinatorDevices::new(&devices)?, "minimax", Lifetime::TARGET,
            Category::Workspace, "native serial dense projection");
        owner.inventory(&mut inventory, "dense")?;
        assert_eq!(inventory.receipts.len(), 4);
        let activation = inventory.receipts.last().unwrap();
        let expected = owner.input_layout.allocation("minimax.dense", 0, Lifetime::TARGET);
        assert_eq!(activation.allocation, expected);
        assert_eq!(fixture.events()?.matches('D').count(), 4);
        let other = ExecutionOwner::new(device, &weights, 17)?;
        assert_ne!(owner.activation.buffer.ptr, other.activation.buffer.ptr);
        assert_ne!(owner.workspace.inputs.buffer.ptr, other.workspace.inputs.buffer.ptr);
        assert_eq!(library.cuda_get_device()?, 0);
        drop(other);
        drop(owner);
        assert_eq!(device.run(|| library.cuda_physical_memory_info())?.0, 1 << 20);
        assert_eq!(fixture.events()?.matches('d').count(), 8);
        assert_eq!(fixture.device_free_mismatch(&library)?, (0, -1, -1));
        Ok(())
    }

    fn manifest() -> serde_json::Value {
        use serde_json::json;
        let scalars = json!([{"name":"rows","type":"int32"},
            {"name":"rotation_id","type":"int32"}, {"name":"cta_limit","type":"int32"}]);
        json!({"programs":[
            {"name":"exlr8_dense_mxfp8_k128_n32_s1_a8","op":"dense_mxfp8",
                "capacity_rows":4096,"scratch_bytes_at_capacity":{},"scalars":scalars,
                "geometry":{"input_features":128,"output_features":32,"sharded_side":1,"encoding":"mxfp8",
                    "input_dtype":"bf16","rotation_count":1,"max_rows":4096,"abi_version":2},
                "pointers":[{"name":"rotated_inputs","dtype":"int64","shape":"[1]","role":"in"},
                    {"name":"payload","dtype":"uint8","shape":"[4096]","role":"in"},
                    {"name":"scales","dtype":"uint8","shape":"[128]","role":"in"},
                    {"name":"records","dtype":"uint8","shape":"[record_bytes]","role":"in"},
                    {"name":"output","dtype":"float32","shape":"[rows,32]","role":"out"}]},
            {"name":"exlr8_dense_quant_k128_bf16_a8","op":"dense_mxfp8_quant",
                "capacity_rows":4096,"scratch_bytes_at_capacity":{},"scalars":scalars,
                "geometry":{"input_features":128,"input_dtype":"bf16","rotation_count":1,"max_rows":4096,"abi_version":2},
                "pointers":[{"name":"rotated_inputs","dtype":"int64","shape":"[1]","role":"in"},
                    {"name":"records","dtype":"uint8","shape":"[record_bytes]","role":"out"}]}
        ]})
    }

    #[test]
    fn binds_selected_native_programs_on_owner_device_without_workspace_allocation() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 1 };
        let mut weights = weights()?;
        // Multiple checkpoint projections share the same shape's loaded handles.
        weights.projections.push(weights.projections[0].clone());
        let manifest = manifest();
        fixture.register_programs(&library, manifest["programs"].as_array().unwrap())?;
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("PROGRAMS.json");
        std::fs::write(&path, serde_json::to_vec(&manifest)?)?;
        let programs = library.programs()?.with_manifest(&path)?;
        let bound = NativePrograms::bind(device, &programs, &weights, &manifest, 17)?;
        assert_eq!(bound.device.id, 1);
        assert_eq!(bound.linears.len(), 1);
        assert_eq!(bound.linears.values().next().unwrap().geometry(), weights.projections[0].geometry());
        assert_eq!(library.cuda_get_device()?, 0);
        assert_eq!(fixture.events()?.matches('D').count(), 0);

        let mut wrong = manifest.clone();
        wrong["programs"][0]["geometry"]["admission_marker"] = serde_json::json!(1);
        let error = NativePrograms::bind(device, &programs, &weights, &wrong, 17).err().unwrap();
        assert!(error.to_string().contains("differs from admitted manifest"), "{error:#}");
        let mut missing = manifest.clone();
        missing["programs"].as_array_mut().unwrap().pop();
        assert!(NativePrograms::bind(device, &programs, &weights, &missing, 17).is_err());
        assert!(NativePrograms::bind(device, &programs, &weights, &manifest, 0).is_err());
        weights.projections.clear();
        assert!(NativePrograms::bind(device, &programs, &weights, &manifest, 17).is_err());
        assert_eq!(library.cuda_get_device()?, 0);
        assert_eq!(fixture.events()?.matches('D').count(), 0);
        Ok(())
    }

    #[test]
    fn prepared_dense_refuses_incompatible_owners_before_dispatch() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let manifest = manifest();
        fixture.register_programs(&library, manifest["programs"].as_array().unwrap())?;
        let path = fixture.directory().join("PROGRAMS.json");
        std::fs::write(&path, serde_json::to_vec(&manifest)?)?;
        let programs = library.programs()?.with_manifest(&path)?;
        let weights = weights()?;
        let device = Device { library: &library, id: 1 };
        let bound = NativePrograms::bind(device, &programs, &weights, &manifest, 17)?;
        let mut owner = ExecutionOwner::new(device, &weights, 17)?;
        assert_eq!(owner.producer_destination().ptr, owner.activation.buffer.ptr);
        let projection = &weights.projections[0];
        let stream = std::ptr::null_mut();
        for rows in [0, 18, 4097] { assert!(owner.prepare(&bound, projection, rows, stream).is_err()); }
        let mut wide = projection.clone();
        wide.input = 256;
        assert!(owner.prepare(&bound, &wide, 17, stream).is_err());
        let other = NativePrograms::bind(Device { library: &library, id: 0 }, &programs, &weights, &manifest, 17)?;
        assert!(owner.prepare(&other, projection, 17, stream).is_err());
        // Real CPU fixture allocations are host fallback: never mask their flags
        // in daemon code just to pass the CUDA-only checked launch contract.
        let error = owner.prepare(&bound, projection, 17, stream).err().unwrap();
        assert!(error.to_string().contains("buffer extent/alignment/device"), "{error:#}");
        assert_eq!(library.cuda_get_device()?, 0);
        assert_eq!(fixture.events()?.matches('D').count(), 4);
        Ok(())
    }

    #[test]
    fn invalid_dense_owner_refuses_before_allocation() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 1 };
        let mut weights = weights()?;
        for rows in [0, 4097] { assert!(Workspace::new(device, &weights, rows).is_err()); }
        weights.projections.clear();
        assert!(Workspace::new(device, &weights, 1).is_err());
        assert_eq!(fixture.events()?.matches('D').count(), 0);
        assert_eq!(library.cuda_get_device()?, 0);
        Ok(())
    }
}
