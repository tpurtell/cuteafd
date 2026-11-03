//! The server's identity for a report: build, hardware, resolved configuration
//! and the fingerprint that keys the baseline.
use crate::context::{self, ServerContext};
use crate::report::{BuildInfo, Configuration, FabricPort, Gpu, Hardware, QuantGroup, ServerInfo, Setting, Spark};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::process::Command;

/// Build identity: runtime release environment first, then what the build baked in.
pub fn build_info() -> BuildInfo {
    let baked = |value: &'static str| (!value.is_empty()).then(|| value.to_string());
    let env = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
    let mut build = BuildInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        release: env("CUTEAFD_RELEASE_VERSION"),
        image: env("CUTEAFD_IMAGE"),
        remote: env("CUTEAFD_GIT_REMOTE").or_else(|| baked(env!("CUTEAFD_BUILD_REMOTE"))),
        commit: baked(env!("CUTEAFD_BUILD_COMMIT")),
        dirty: match env!("CUTEAFD_BUILD_DIRTY") { "true" => Some(true), "false" => Some(false), _ => None },
    };
    // Release images: `REV` or `REV-dirty-MANIFEST12`.
    if let Some(engine) = env("CUTEAFD_ENGINE_COMMIT") {
        match engine.split_once("-dirty") {
            Some((rev, _)) => (build.commit, build.dirty) = (Some(rev.to_string()), Some(true)),
            None => (build.commit, build.dirty) = (Some(engine), Some(false)),
        }
    }
    if build.remote.as_deref().is_some_and(|r| r.starts_with("git@github.com:")) {
        // A shareable URL rather than an SSH remote.
        build.remote = build.remote.map(|r| format!("https://github.com/{}", r.trim_start_matches("git@github.com:")
            .trim_end_matches(".git")));
    }
    build
}

fn run(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output.status.success().then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// SM counts of the GPUs this project targets (nvidia-smi does not report them).
fn sm_count(name: &str) -> Option<u32> {
    let name = name.to_ascii_uppercase();
    [("RTX PRO 6000", 188), ("RTX 5090", 170), ("GB10", 48), ("RTX 5080", 84), ("H100", 132), ("B200", 148)]
        .iter().find(|(key, _)| name.contains(key)).map(|&(_, sms)| sms)
}

fn setting<'a>(settings: &'a [Setting], names: &[&str]) -> Option<&'a str> {
    settings.iter().find(|s| names.contains(&s.name.as_str())).and_then(|s| s.value.as_deref())
}

/// GPUs from nvidia-smi, marking the ones this server drives.
fn gpus(settings: &[Setting]) -> (Vec<Gpu>, Option<String>, Option<String>) {
    let fields = "index,name,uuid,memory.total,power.limit,power.max_limit,compute_cap,pcie.link.gen.max,\
        pcie.link.width.max,driver_version";
    let Some(text) = run("nvidia-smi", &[&format!("--query-gpu={fields}"), "--format=csv,noheader,nounits"]) else {
        return (Vec::new(), None, None);
    };
    let mut driver = None;
    let mut gpus: Vec<Gpu> = text.lines().filter_map(|line| {
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        if cols.len() < 10 {
            return None;
        }
        let number = |s: &str| s.parse::<f64>().ok();
        driver = Some(cols[9].to_string());
        Some(Gpu {
            index: cols[0].parse().ok()?,
            name: cols[1].to_string(),
            uuid: Some(cols[2].to_string()),
            memory_mib: cols[3].parse().ok(),
            power_limit_w: number(cols[4]),
            power_max_w: number(cols[5]),
            sm_count: sm_count(cols[1]),
            compute_cap: Some(cols[6].to_string()),
            pcie: match (cols[7].parse::<u32>(), cols[8].parse::<u32>()) {
                (Ok(gen), Ok(width)) => Some(format!("Gen{gen} x{width}")),
                _ => None,
            },
            used: false,
        })
    }).collect();
    let cuda = run("nvidia-smi", &[]).and_then(|text| {
        let at = text.find("CUDA Version:")?;
        Some(text[at + 13..].split_whitespace().next()?.to_string())
    });
    // CUDA_VISIBLE_DEVICES (indices or UUIDs) orders the devices the process sees.
    let visible: Vec<usize> = match std::env::var("CUDA_VISIBLE_DEVICES") {
        Ok(list) if !list.trim().is_empty() => list.split(',').filter_map(|entry| {
            let entry = entry.trim();
            gpus.iter().position(|g| g.uuid.as_deref() == Some(entry) || g.index.to_string() == entry
                || g.uuid.as_deref().is_some_and(|u| u.starts_with(entry)))
        }).collect(),
        _ => (0..gpus.len()).collect(),
    };
    let ordinal = |name: &[&str]| setting(settings, name).and_then(|v| v.parse::<usize>().ok());
    let mut used: Vec<usize> = Vec::new();
    if let Some(count) = ordinal(&["rtx-gpus"]) {
        used.extend(visible.iter().take(count));
    } else if let Some(device) = ordinal(&["device"]) {
        used.extend(visible.get(device));
        if let Some(split) = ordinal(&["split-device"]) {
            used.extend(visible.get(split));
        }
    } else if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
        used.extend(visible.first());
    }
    if used.is_empty() {
        used.extend(visible.first());
    }
    for index in used {
        if let Some(gpu) = gpus.get_mut(index) {
            gpu.used = true;
        }
    }
    (gpus, driver, cuda)
}

/// Spark ranks from `--peers` (TP order), named from /etc/hosts when listed.
fn sparks(settings: &[Setting]) -> Vec<Spark> {
    let Some(peers) = setting(settings, &["peers"]) else { return Vec::new() };
    let hosts = std::fs::read_to_string("/etc/hosts").unwrap_or_default();
    peers.split(',').filter(|p| !p.trim().is_empty()).enumerate().map(|(rank, peer)| {
        let address = peer.trim().rsplit_once(':').map_or(peer.trim(), |(host, _)| host).to_string();
        let name = hosts.lines().filter(|l| !l.trim_start().starts_with('#')).find_map(|line| {
            let mut words = line.split_whitespace();
            (words.next()? == address).then(|| words.next().map(str::to_string)).flatten()
        });
        Spark { address, name, rank: rank as u32 }
    }).collect()
}

pub fn hardware(settings: &[Setting]) -> Hardware {
    let (gpus, driver, cuda) = gpus(settings);
    let (fabric, rails) = match cuteafd_transport::fabric::discover() {
        Ok(report) => (report.ports.iter().map(|port| FabricPort {
            device: port.device.clone(),
            port: port.port,
            active: port.active,
            link_gbps: port.link_gbps,
            pcie: port.pci.as_ref().map(|pci| format!("{} GT/s x{}", pci.gts, pci.width)),
            netdev: port.netdev.clone(),
            subnets: port.subnets.iter().map(|(net, bits)| format!("{net}/{bits}")).collect(),
        }).collect(), Some(report.summary())),
        Err(_) => (Vec::new(), None),
    };
    Hardware {
        host: std::fs::read_to_string("/proc/sys/kernel/hostname").ok().map(|h| h.trim().to_string())
            .or_else(|| std::env::var("HOSTNAME").ok()),
        gpus, driver, cuda, sparks: sparks(settings), fabric, rails, power: None,
    }
}

/// Storage formats per tensor group, from the loader's plan of the snapshot.
fn quant(snapshot: &Path, spark_ranks: usize) -> Vec<QuantGroup> {
    use cuteafd_loader::plan::{plan, ExpertPlacement, PlanOptions};
    let candidate = PlanOptions { placement: ExpertPlacement::from_spark_ranks(spark_ranks), ..PlanOptions::default() };
    let ranks = if candidate.validate().is_ok() { spark_ranks } else { 4 };
    let options = PlanOptions { placement: ExpertPlacement::from_spark_ranks(ranks), ..PlanOptions::default() };
    let Ok(report) = plan(snapshot, &options) else { return Vec::new() };
    report.components.iter().filter(|c| !c.formats.is_empty()).map(|c| QuantGroup {
        group: serde_json::to_value(c.component).ok().and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_else(|| format!("{:?}", c.component)).replace('_', " "),
        formats: c.formats.keys().cloned().collect(),
    }).collect()
}

/// The speculator the options select.
fn speculator(_family: Option<&str>, settings: &[Setting]) -> Option<String> {
    let truthy = |names: &[&str]| setting(settings, names).is_some_and(|v| v == "true");
    let number = |names: &[&str]| setting(settings, names).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    if truthy(&["dspark"]) {
        return Some(match setting(settings, &["dspark-draft-limit"]) {
            Some(limit) => format!("dSpark (≤{limit} drafts)"),
            None => "dSpark".into(),
        });
    }
    if let Some(draft) = setting(settings, &["draft"]).filter(|v| *v != "false") {
        let name = Path::new(draft).file_name().and_then(|n| n.to_str()).unwrap_or(draft);
        let kind = if draft.to_ascii_lowercase().contains("dspark") { "dSpark" } else { "DFlash2" };
        return Some(if draft == "true" { "draft model".into() } else { format!("{kind} ({name})") });
    }
    let mtp = number(&["mtp"]);
    if mtp > 0 {
        return Some(format!("MTP ×{mtp}"));
    }
    None
}

fn layout(hardware: &Hardware, settings: &[Setting]) -> String {
    let mut parts = vec![format!("{} RTX", hardware.used_gpus())];
    if setting(settings, &["split-device"]).is_some() {
        parts[0].push_str(" (head split)");
    }
    match hardware.sparks.len() {
        0 => parts.push("experts on the coordinator".into()),
        n => {
            let tp = setting(settings, &["spark-tp"]).and_then(|v| v.parse::<usize>().ok());
            let ep = setting(settings, &["spark-ep"]).and_then(|v| v.parse::<usize>().ok());
            parts.push(match (tp, ep) {
                (Some(tp), Some(ep)) => format!("experts TP{tp}×EP{ep} over {n} Sparks"),
                _ => format!("experts TP{n} over {n} Sparks"),
            });
        }
    }
    if let Some(placement) = context::mapped_placement() { parts.push(placement.label()); }
    parts.join(" · ")
}

pub fn configuration(context: &ServerContext, hardware: &Hardware) -> Configuration {
    let settings = context.settings.clone();
    let quant = context.snapshot.as_deref().map(|s| quant(s, hardware.sparks.len())).unwrap_or_default();
    Configuration {
        supports_thinking_off: context.supports_thinking_off,
        snapshot: context.snapshot.as_ref().map(|p| p.display().to_string()),
        quant,
        speculator: speculator(context.family.as_deref(), &settings),
        layout: Some(layout(hardware, &settings)),
        mapped_placement: context::mapped_placement(),
        settings,
    }
}

/// Everything a report says about the server (`model` from `/v1/models`).
pub fn server_info(model: &str) -> ServerInfo {
    let context = context::get();
    server_info_from_context(model, &context, hardware(&context.settings))
}

fn server_info_from_context(model: &str, context: &ServerContext, hardware: Hardware) -> ServerInfo {
    let configuration = configuration(context, &hardware);
    ServerInfo {
        model: model.to_string(),
        executor: (!context.command.is_empty()).then(|| context.command.clone()),
        family: context.family.clone(),
        revision: context.snapshot.as_deref().and_then(revision),
        build: build_info(),
        hardware,
        configuration,
        readiness_s: context::readiness_s(),
        started: Some(crate::report::rfc3339(context::started())),
    }
}

/// `…/snapshots/<rev>` -> `<rev>`.
pub fn revision(snapshot: &Path) -> Option<String> {
    let parent = snapshot.parent()?;
    (parent.file_name()? == "snapshots").then(|| snapshot.file_name()?.to_str().map(str::to_string)).flatten()
}

/// The config fingerprint: model, checkpoint revision, build, non-default
/// options and hardware. Equal fingerprints share a baseline.
pub fn fingerprint(info: &ServerInfo) -> String {
    let mut hasher = Sha256::new();
    let mut line = |key: &str, value: &str| {
        hasher.update(key.as_bytes());
        hasher.update(b"=");
        hasher.update(value.as_bytes());
        hasher.update(b"\n");
    };
    if !info.configuration.supports_thinking_off() { line("thinking", info.configuration.thinking_label()); }
    line("prompt_content", crate::text::PROMPT_CONTENT);
    line("model", &info.model);
    line("revision", info.revision.as_deref().unwrap_or(""));
    line("build", &info.build.label());
    let mut options: Vec<String> = info.configuration.non_default()
        .filter(|s| !context::DEPLOYMENT.contains(&s.name.as_str()))
        .map(Setting::chip).collect();
    options.sort();
    line("options", &options.join(" "));
    for gpu in info.hardware.gpus.iter().filter(|g| g.used) {
        line("gpu", &format!("{} {:?}", gpu.name, gpu.power_limit_w));
    }
    line("sparks", &info.hardware.sparks.len().to_string());
    let mut links: Vec<String> = info.hardware.fabric.iter().filter(|p| p.active)
        .map(|p| format!("{}:{}", p.device, p.link_gbps)).collect();
    links.sort();
    line("fabric", &links.join(","));
    let digest = hasher.finalize();
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loaded_config_identity_is_independent_of_executor_in_reports_and_probes() {
        let glm = cuteafd_loader::families::glm5::GlmDsaConfig::from_hf(
            &cuteafd_loader::plan::testing::glm5_config()).unwrap();
        let mimo = cuteafd_loader::families::mimo_v2::MimoV2Config::from_hf(
            &cuteafd_loader::plan::testing::mimo_flash_config()).unwrap();
        assert_eq!(glm.family_id(), "glm5");
        assert_eq!(mimo.family_id(), "mimo_v2");
        for (executor, loaded_family) in [("serve-glm", glm.family_id()), ("serve-mimo", mimo.family_id())] {
            let mut context = ServerContext { command: executor.into(), ..Default::default() };
            assert!(context.family.is_none());
            context.set_family(loaded_family);
            let report = server_info_from_context("served-alias", &context, Hardware::default());
            assert_eq!(report.family.as_deref(), Some(loaded_family));
            assert_eq!(report.executor.as_deref(), Some(executor));
            let probe = crate::runner::probe_server("served-alias", &context);
            assert_eq!(probe["family"], loaded_family);
            assert_eq!(probe["executor"], executor);
            assert_eq!(probe["model"], "served-alias");
            // A shared executor can serve a different canonical checkpoint family.
            context.set_family("other_checkpoint_family");
            assert_eq!(server_info_from_context("served-alias", &context, Hardware::default())
                .family.as_deref(), Some("other_checkpoint_family"));
            assert_eq!(crate::runner::probe_server("served-alias", &context)["executor"], executor);
        }
    }

    #[test]
    fn ml4_loaded_identity_uses_glm_executor_without_relabeling_the_checkpoint() {
        let config = cuteafd_loader::families::glm5::GlmDsaConfig::from_hf(
            &cuteafd_loader::plan::testing::mistral_large4_config()).unwrap();
        assert_eq!(config.family_id(), "mistral_large4");
        let mut context = ServerContext { command: "serve-glm".into(), ..Default::default() };
        context.set_family(config.family_id());
        let report = server_info_from_context("mistralai/Mistral-Large-4-1T-A52B", &context, Hardware::default());
        assert_eq!(report.family.as_deref(), Some("mistral_large4"));
        assert_eq!(report.executor.as_deref(), Some("serve-glm"));
        let probe = crate::runner::probe_server(&report.model, &context);
        assert_eq!(probe["family"], "mistral_large4");
        assert_eq!(probe["executor"], "serve-glm");
    }

    #[test]
    fn legacy_reports_without_executor_still_deserialize() {
        let report = ServerInfo { model: "legacy".into(), ..Default::default() };
        let mut value = serde_json::to_value(report).unwrap();
        value.as_object_mut().unwrap().remove("executor");
        let loaded: ServerInfo = serde_json::from_value(value).unwrap();
        assert_eq!(loaded.executor, None);
    }

    fn s(name: &str, value: &str, default: Option<&str>, source: &str) -> Setting {
        Setting { name: name.into(), value: Some(value.into()), default: default.map(str::to_string), source: source.into() }
    }

    #[test]
    fn speculator_follows_the_options() {
        assert_eq!(speculator(Some("deepseek_v41"), &[s("dspark", "true", Some("false"), "cli"),
            s("dspark-draft-limit", "5", Some("5"), "default")]).as_deref(), Some("dSpark (≤5 drafts)"));
        assert_eq!(speculator(Some("qwen4"), &[s("mtp", "3", Some("0"), "cli")]).as_deref(), Some("MTP ×3"));
        assert_eq!(speculator(Some("glm5"), &[s("draft", "/hf/GLM-5.3-DFlash2", None, "cli")]).as_deref(),
            Some("DFlash2 (GLM-5.3-DFlash2)"));
        assert_eq!(speculator(Some("glm5_flash"), &[s("draft",
            "/hf/models--RedHatAI--GLM-5.3-Flash-speculator.dspark-preview/snapshots/1972f1f0", None, "cli")]).as_deref(),
            Some("dSpark (1972f1f0)"));
        assert_eq!(speculator(Some("qwen4"), &[]), None);
        assert_eq!(speculator(Some("mimo_v2"), &[]), None);
    }

    #[test]
    fn fingerprint_ignores_default_and_deployment_options() {
        let mut info = ServerInfo { model: "m".into(), ..ServerInfo::default() };
        info.configuration.settings = vec![s("max-context", "8192", Some("8192"), "cli"),
            s("snapshot", "/a", None, "cli")];
        let base = fingerprint(&info);
        info.configuration.settings[1].value = Some("/b".into());
        assert_eq!(fingerprint(&info), base);
        info.configuration.settings[0].value = Some("4096".into());
        assert_ne!(fingerprint(&info), base);
        assert_eq!(base.len(), 16);
    }

    #[test]
    fn renderer_capability_reaches_configuration_and_separates_baselines() {
        let context = ServerContext { supports_thinking_off: Some(false), ..Default::default() };
        let config = configuration(&context, &Hardware::default());
        assert_eq!(config.supports_thinking_off, Some(false));
        let mut info = ServerInfo::default();
        let legacy = fingerprint(&info);
        info.configuration.supports_thinking_off = Some(true);
        assert_eq!(fingerprint(&info), legacy);
        info.configuration = config;
        assert_ne!(fingerprint(&info), legacy);
    }

    #[test]
    fn revision_of_a_snapshot_path() {
        assert_eq!(revision(Path::new("/hub/models--a--b/snapshots/abc123")).as_deref(), Some("abc123"));
        assert_eq!(revision(Path::new("/tmp/model")), None);
    }
}
