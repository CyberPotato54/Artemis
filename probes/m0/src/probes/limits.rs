//! Everything the driver reports that the design depends on. Needs only an instance, so it runs even if
//! creating the device would fail.
use crate::{ctx, record::Record};
use anyhow::Result;
use ash::vk;
use serde_json::{json, Map, Value};

const EXTENSIONS: &[&str] = &[
    "VK_KHR_cooperative_matrix",
    "VK_EXT_memory_budget",
    "VK_AMD_device_coherent_memory",
    "VK_EXT_external_memory_host",
    "VK_KHR_shader_integer_dot_product",
    "VK_KHR_calibrated_timestamps",
];

pub fn run(want: Option<&str>) -> Result<Record> {
    let (entry, instance, pd) = ctx::open_instance(want)?;
    // SAFETY: property and feature queries on a valid physical device.
    unsafe {
        let mut p11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut p13 = vk::PhysicalDeviceVulkan13Properties::default();
        let mut p2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut p11)
            .push_next(&mut p13);
        instance.get_physical_device_properties2(pd, &mut p2);
        let props = p2.properties;

        let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut f2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f11)
            .push_next(&mut f12)
            .push_next(&mut f13);
        instance.get_physical_device_features2(pd, &mut f2);
        let core = f2.features;

        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut m2 = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        instance.get_physical_device_memory_properties2(pd, &mut m2);
        let mp = m2.memory_properties;

        let exts: Vec<String> = instance
            .enumerate_device_extension_properties(pd)?
            .iter()
            .map(|e| {
                e.extension_name_as_c_str()
                    .unwrap_or(c"")
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        let coop: Vec<Value> = if exts.iter().any(|e| e == "VK_KHR_cooperative_matrix") {
            ash::khr::cooperative_matrix::Instance::new(&entry, &instance)
                .get_physical_device_cooperative_matrix_properties(pd)?
                .iter()
                .map(|c| {
                    json!({"m": c.m_size, "n": c.n_size, "k": c.k_size, "a": format!("{:?}", c.a_type),
                           "b": format!("{:?}", c.b_type), "c": format!("{:?}", c.c_type),
                           "result": format!("{:?}", c.result_type), "scope": format!("{:?}", c.scope)})
                })
                .collect()
        } else {
            vec![]
        };
        let heaps: Vec<Value> = (0..mp.memory_heap_count as usize)
            .map(|i| {
                json!({"size_mib": mp.memory_heaps[i].size >> 20, "flags": format!("{:?}", mp.memory_heaps[i].flags),
                       "budget_mib": budget.heap_budget[i] >> 20, "usage_mib": budget.heap_usage[i] >> 20})
            })
            .collect();
        let types: Vec<Value> = (0..mp.memory_type_count as usize)
            .map(|i| json!({"index": i, "heap": mp.memory_types[i].heap_index, "flags": format!("{:?}", mp.memory_types[i].property_flags)}))
            .collect();
        let on = |b: vk::Bool32| b == vk::TRUE;
        let l = props.limits;
        let extensions: Map<String, Value> = EXTENSIONS
            .iter()
            .map(|e| (e.to_string(), Value::Bool(exts.iter().any(|x| x == e))))
            .collect();
        let results = json!({
            "api_version": format!("{}.{}.{}", vk::api_version_major(props.api_version),
                                   vk::api_version_minor(props.api_version), vk::api_version_patch(props.api_version)),
            "heaps": heaps,
            "memory_types": types,
            "max_memory_allocation_mib": p11.max_memory_allocation_size >> 20,
            "max_storage_buffer_range_mib": l.max_storage_buffer_range >> 20,
            "max_push_constants_size": l.max_push_constants_size,
            "max_compute_work_group_count": l.max_compute_work_group_count,
            "max_compute_work_group_invocations": l.max_compute_work_group_invocations,
            "max_compute_shared_memory_size": l.max_compute_shared_memory_size,
            "timestamp_period_ns": l.timestamp_period,
            "subgroup_size": p11.subgroup_size,
            "min_subgroup_size": p13.min_subgroup_size,
            "max_subgroup_size": p13.max_subgroup_size,
            "int8_dot_accelerated": on(p13.integer_dot_product4x8_bit_packed_signed_accelerated),
            "features": {
                "shader_int64": on(core.shader_int64), "shader_int16": on(core.shader_int16),
                "storage_buffer16_bit_access": on(f11.storage_buffer16_bit_access),
                "buffer_device_address": on(f12.buffer_device_address), "timeline_semaphore": on(f12.timeline_semaphore),
                "host_query_reset": on(f12.host_query_reset), "shader_int8": on(f12.shader_int8),
                "shader_float16": on(f12.shader_float16), "storage_buffer8_bit_access": on(f12.storage_buffer8_bit_access),
                "vulkan_memory_model": on(f12.vulkan_memory_model),
                "vulkan_memory_model_device_scope": on(f12.vulkan_memory_model_device_scope),
                "synchronization2": on(f13.synchronization2), "subgroup_size_control": on(f13.subgroup_size_control),
            },
            "extensions": extensions,
            "cooperative_matrix": coop,
        });
        let (name, driver) = ctx::names(&props);
        Ok(Record::new("limits", &name, &driver, json!({}), results))
    }
}
