//! Prints the driver's statistics for compute shaders (VK_KHR_pipeline_executable_properties): registers, scratch,
//! LDS and whatever else the driver reports, per pipeline executable. A development tool for kernel work; it builds
//! its own device so the extension and the statistics flag are on.
//!
//! ```text
//! cargo run --release -p vk --example pipeline_stats -- [--ir DIR] FILE.spv[@SG][=V] ...
//! ```
//!
//! `@SG` (for example `experts_down.spv@32`) builds that pipeline with a required subgroup size of SG and full
//! subgroups; without it the driver picks the size. `=V` sets specialization constant 0 to the u32 V (for example
//! `experts_down_tokens.spv@32=5`). `--ir DIR` also writes each executable's internal representations
//! (the driver's ISA, when it gives one) to DIR.

use ash::vk;
use std::ffi::CStr;

fn text(s: &[std::ffi::c_char]) -> String {
    // SAFETY: Vulkan fills these fixed arrays with NUL-terminated strings.
    unsafe { CStr::from_ptr(s.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn value(s: &vk::PipelineExecutableStatisticKHR<'_>) -> String {
    // SAFETY: `format` names the union member the driver wrote.
    unsafe {
        match s.format {
            vk::PipelineExecutableStatisticFormatKHR::BOOL32 => (s.value.b32 != 0).to_string(),
            vk::PipelineExecutableStatisticFormatKHR::INT64 => s.value.i64.to_string(),
            vk::PipelineExecutableStatisticFormatKHR::UINT64 => s.value.u64.to_string(),
            vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => s.value.f64.to_string(),
            _ => "?".into(),
        }
    }
}

fn main() {
    let mut ir: Option<std::path::PathBuf> = None;
    let mut files: Vec<(String, Option<u32>, Option<u32>)> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--ir" {
            ir = Some(args.next().expect("--ir DIR").into());
            continue;
        }
        let (a, spec) = match a.rsplit_once('=') {
            Some((f, v)) => (f.to_string(), Some(v.parse().expect("=V is a number"))),
            None => (a, None),
        };
        match a.rsplit_once('@') {
            Some((f, sg)) => files.push((
                f.to_string(),
                Some(sg.parse().expect("@SG is a number")),
                spec,
            )),
            None => files.push((a, None, spec)),
        }
    }
    assert!(
        !files.is_empty(),
        "usage: pipeline_stats [--ir DIR] FILE.spv[@SG][=V] ..."
    );
    // SAFETY: a development tool; every create info outlives its call and every handle is used on this thread.
    unsafe {
        let entry = ash::Entry::load().expect("the Vulkan loader");
        let app = vk::ApplicationInfo::default()
            .application_name(c"pipeline_stats")
            .api_version(vk::API_VERSION_1_3);
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )
            .expect("an instance");
        let pd = instance
            .enumerate_physical_devices()
            .expect("physical devices")
            .into_iter()
            .find(|&p| {
                instance.get_physical_device_properties(p).device_type
                    == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .expect("a discrete GPU");
        let props = instance.get_physical_device_properties(pd);
        println!("device: {}", text(&props.device_name));
        let family = instance
            .get_physical_device_queue_family_properties(pd)
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .expect("a compute queue") as u32;
        let prio = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&prio)];
        let exts = [ash::khr::pipeline_executable_properties::NAME.as_ptr()];
        let mut pep = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default()
            .pipeline_executable_info(true);
        let mut e11 =
            vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut e12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .shader_int8(true)
            .shader_float16(true)
            .storage_buffer8_bit_access(true);
        let mut e13 = vk::PhysicalDeviceVulkan13Features::default()
            .subgroup_size_control(true)
            .compute_full_subgroups(true);
        let core = vk::PhysicalDeviceFeatures::default()
            .shader_int64(true)
            .shader_int16(true);
        let mut feats = vk::PhysicalDeviceFeatures2::default()
            .features(core)
            .push_next(&mut pep)
            .push_next(&mut e11)
            .push_next(&mut e12)
            .push_next(&mut e13);
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&exts)
            .push_next(&mut feats);
        let device = instance.create_device(pd, &info, None).expect("a device");
        let pe = ash::khr::pipeline_executable_properties::Device::new(&instance, &device);
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(128)];
        let layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges),
                None,
            )
            .expect("a pipeline layout");
        for (file, sg, spec) in &files {
            let bytes = std::fs::read(file).unwrap_or_else(|e| panic!("{file}: {e}"));
            let code = ash::util::read_spv(&mut std::io::Cursor::new(&bytes)).expect("SPIR-V");
            let module = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
                .expect("a shader module");
            let mut req = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                .required_subgroup_size(sg.unwrap_or(0));
            let mut stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            if sg.is_some() {
                stage = stage
                    .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
                    .push_next(&mut req);
            }
            let entry = [vk::SpecializationMapEntry::default()
                .constant_id(0)
                .offset(0)
                .size(4)];
            let data = spec.unwrap_or(0).to_ne_bytes();
            let spec_info = vk::SpecializationInfo::default()
                .map_entries(&entry)
                .data(&data);
            if spec.is_some() {
                stage = stage.specialization_info(&spec_info);
            }
            let mut flags = vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR;
            if ir.is_some() {
                flags |= vk::PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS_KHR;
            }
            let ci = [vk::ComputePipelineCreateInfo::default()
                .flags(flags)
                .stage(stage)
                .layout(layout)];
            let pipeline = device
                .create_compute_pipelines(vk::PipelineCache::null(), &ci, None)
                .map_err(|(_, e)| e)
                .expect("a pipeline")[0];
            let pinfo = vk::PipelineInfoKHR::default().pipeline(pipeline);
            let execs = pe
                .get_pipeline_executable_properties(&pinfo)
                .expect("executable properties");
            for (i, x) in execs.iter().enumerate() {
                println!(
                    "{file}{}: executable {i} {} (subgroup size {})",
                    spec.map(|v| format!("={v}")).unwrap_or_default(),
                    text(&x.name),
                    x.subgroup_size
                );
                let ei = vk::PipelineExecutableInfoKHR::default()
                    .pipeline(pipeline)
                    .executable_index(i as u32);
                for s in pe
                    .get_pipeline_executable_statistics(&ei)
                    .expect("statistics")
                {
                    println!("  {:<32} {}", text(&s.name), value(&s));
                }
                if let Some(dir) = &ir {
                    // ash's wrapper can't pass data buffers, so the two-call pattern goes through the raw function.
                    let f = pe.fp().get_pipeline_executable_internal_representations_khr;
                    let mut n = 0u32;
                    f(pe.device(), &ei, &mut n, std::ptr::null_mut())
                        .result()
                        .expect("representation count");
                    let mut reps = vec![
                        vk::PipelineExecutableInternalRepresentationKHR::default();
                        n as usize
                    ];
                    f(pe.device(), &ei, &mut n, reps.as_mut_ptr())
                        .result()
                        .expect("representation sizes");
                    let mut bufs: Vec<Vec<u8>> =
                        reps.iter().map(|r| vec![0u8; r.data_size]).collect();
                    for (r, b) in reps.iter_mut().zip(bufs.iter_mut()) {
                        r.p_data = b.as_mut_ptr().cast();
                    }
                    f(pe.device(), &ei, &mut n, reps.as_mut_ptr())
                        .result()
                        .expect("representations");
                    std::fs::create_dir_all(dir).expect("the --ir folder");
                    let stem = std::path::Path::new(file)
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    for (k, (r, b)) in reps.iter().zip(&bufs).enumerate() {
                        let out = dir.join(format!("{stem}.{i}.{k}.txt"));
                        let body = if r.is_text != 0 {
                            String::from_utf8_lossy(b)
                                .trim_end_matches('\0')
                                .to_string()
                        } else {
                            format!("({} bytes of binary)", b.len())
                        };
                        std::fs::write(
                            &out,
                            format!("{}\n{}\n\n{body}", text(&r.name), text(&r.description)),
                        )
                        .expect("writing a representation");
                        println!("  wrote {}", out.display());
                    }
                }
            }
            device.destroy_pipeline(pipeline, None);
            device.destroy_shader_module(module, None);
        }
        device.destroy_pipeline_layout(layout, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
}
