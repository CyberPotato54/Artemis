// Slot-table entries, 16 bytes each: { u64 addr; u32 where; u32 reserved }. HOST and VRAM entries run
// the same code at a different address, so results never depend on where an expert lives. Everything else is refused
// without reading its address: CPU entries (except where a kernel's serve mode computes them for the CPU miss path),
// and entries with an unknown `where` or no address at all. The kernel writes NaN and records the entry in the step's
// status buffer.
const uint WHERE_VRAM = 0u;
const uint WHERE_HOST = 1u;
const uint WHERE_CPU = 2u;

layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer Entries { uvec4 v[]; };

struct Slot {
    uint64_t addr;
    uint loc;
};

Slot slot_at(uint64_t table, uint i) {
    uvec4 e = Entries(table).v[i];
    Slot s;
    s.addr = packUint2x32(e.xy);
    s.loc = e.z;
    return s;
}

bool usable(Slot s) {
    return s.loc <= WHERE_HOST && s.addr != 0ul;
}

// status = [cpu, bad, -, -], each `layer << 16 | expert` of the lowest such entry a step reached, or all ones for none.
// The workgroups run in parallel, so the lowest, not the first, is what's deterministic.
void refuse(uint64_t status, Slot s, uint layer, uint expert) {
    atomicMin(U32s(status).v[s.loc == WHERE_CPU ? 0 : 1], (layer << 16) | expert);
}
