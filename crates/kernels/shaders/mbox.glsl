// The CPU miss path's mailbox in AMD device-uncached host memory (the only memory whose
// atomics a running shader and the CPU both see promptly), byte offsets, mirrored in kernels::dispatch::mbox.
// Flags: layer L's publish flag at word pub_word(L), its done flag at word done_word(L), each on its own 128-byte
// line. X area: layer L's at X0 + L * XSTRIDE: a 256-byte header { seq, token, layer, mask, -, -, -, -, ids[8] },
// then x (n_embd <= 2048 f32). Y area: layer L's at Y0 + L * YSTRIDE: the CPU's outputs, [8][n_embd] f32, one
// expert per routed index j, never sorted by expert id. The wait's status words at STATUS_WORD: [timeout count,
// skip count], byte 12288, on layer 48's publish-flag line, hence n_layer + 1 <= 48. XSTRIDE and YSTRIDE are sized
// for the maximum (n_embd 2048, 8 experts), so smaller models waste bytes and read nothing past their own buffers.
const uint X0 = 16384u;
const uint XSTRIDE = 8448u;
const uint Y0 = 557056u;
const uint YSTRIDE = 65536u;
const uint N_EMBD = 2048u;
const uint MAX_E = 8u;
const uint STATUS_WORD = 3072u;

// The mailbox itself, reached through its device address: coherent and volatile, so atomics and plain reads
// against the CPU's writes behave.
layout(buffer_reference, std430, buffer_reference_align = 4) coherent volatile buffer MU { uint v[]; };

uint pub_word(uint layer) {
    return 64u * layer;
}
uint done_word(uint layer) {
    return 64u * layer + 32u;
}
