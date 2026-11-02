// artemis-oracle: llama.cpp b11509 as Artemis's oracle. It links libllama and writes what the tests compare against:
// every named intermediate tensor (taps), greedy generations with each step's top two logits (greedy), runs
// teacher-forced along given tokens with each step's top two logits (forced), token ids (tokenize), the vocabulary's
// end-of-generation set and token attributes (vocab), and the MTP block's intermediates over the main model's own
// hidden rows (mtp-taps). Our code, MIT OR Apache-2.0. llama.cpp (MIT) is used as a library; none of its code is copied
// here.
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "llama.h"
#include "llama-ext.h"  // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (src/, see CMakeLists)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Args {
    std::string mode, model, backend = "cpu", tokens, force, out, file, filter = ".*";
    int n = 256;
    bool experts_on_cpu = false, no_parse_special = false;
};

[[noreturn]] static void die(const std::string & msg) {
    fprintf(stderr, "artemis-oracle: %s\n", msg.c_str());
    exit(1);
}

// --n's value: a whole number of steps from 1 to 65,536. std::stoi alone took "-5" and threw, uncaught, on "abc".
static int parse_n(const std::string & s) {
    size_t used = 0;
    long v = -1;
    try {
        v = std::stol(s, &used);
    } catch (const std::exception &) {
        used = 0;
    }
    bool digits = !s.empty() && s.find_first_not_of("0123456789") == std::string::npos;
    if (!digits || used == 0 || used != s.size() || v < 1 || v > 65536) {
        die("--n takes a whole number of steps from 1 to 65536, not \"" + s + "\"");
    }
    return (int) v;
}

static Args parse(int argc, char ** argv) {
    if (argc < 2) {
        die("usage: artemis-oracle taps|mtp-taps|greedy|forced|tokenize|vocab --model PATH [options]");
    }
    Args a;
    a.mode = argv[1];
    for (int i = 2; i < argc; i++) {
        const std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) {
                die(k + " needs a value");
            }
            return argv[++i];
        };
        if (k == "--model") a.model = val();
        else if (k == "--backend") a.backend = val();
        else if (k == "--tokens") a.tokens = val();
        else if (k == "--force") a.force = val();
        else if (k == "--out") a.out = val();
        else if (k == "--file") a.file = val();
        else if (k == "--filter") a.filter = val();
        else if (k == "--n") a.n = parse_n(val());
        else if (k == "--experts-on-cpu") a.experts_on_cpu = true;
        else if (k == "--no-parse-special") a.no_parse_special = true;
        else die("unknown option " + k);
    }
    if (a.model.empty()) die("--model is required");
    if (a.backend != "cpu" && a.backend != "vulkan") die("--backend is cpu or vulkan");
    return a;
}

static std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot read " + path);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), {});
}

static void write_file(const fs::path & path, const void * data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) die("cannot write " + path.string());
    f.write((const char *) data, (std::streamsize) n);
}

static std::vector<llama_token> read_tokens(const std::string & path) {
    auto b = read_file(path);
    if (b.size() % 4) die(path + " is not a whole number of u32 tokens");
    std::vector<llama_token> t(b.size() / 4);
    memcpy(t.data(), b.data(), b.size());
    return t;
}

static ggml_backend_dev_t no_devices[1] = { nullptr };
static llama_model_tensor_buft_override expert_override[2] = {
    { "\\.ffn_(gate|up|down)_exps\\.", nullptr },
    { nullptr, nullptr },
};

static llama_model * load(const Args & a, bool vocab_only) {
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = vocab_only;
    mp.load_mtp = a.mode == "mtp-taps";  // b11509 loads the nextn tensors only when asked (common.cpp:1687)
    if (a.backend == "cpu") {
        mp.n_gpu_layers = 0;
        mp.devices = no_devices;  // an empty, null-terminated device list: nothing is offloaded
    } else {
        mp.n_gpu_layers = 999;
        if (a.experts_on_cpu) {
            // A 16 GB card can't hold the experts. They stay in host memory, as with llama.cpp's -ot exps=CPU.
            expert_override[0].buft = ggml_backend_cpu_buffer_type();
            mp.tensor_buft_overrides = expert_override;
        }
    }
    llama_model * m = llama_model_load_from_file(a.model.c_str(), mp);
    if (!m) die("cannot load " + a.model);
    return m;
}

static llama_context * context(llama_model * m, const Args & a, uint32_t n_ctx, uint32_t n_batch, uint32_t n_ubatch,
                               ggml_backend_sched_eval_callback cb, void * user_data,
                               llama_context_type ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = std::max<uint32_t>(n_ctx, 256);
    cp.n_batch = n_batch;
    cp.n_ubatch = n_ubatch;
    cp.cb_eval = cb;
    cp.cb_eval_user_data = user_data;
    cp.ctx_type = ctx_type;
    if (a.backend == "cpu") {
        cp.offload_kqv = false;
        cp.op_offload = false;
    }
    llama_context * c = llama_init_from_model(m, cp);
    if (!c) die("cannot create a context");
    return c;
}

// ---- taps ----

struct Taps {
    std::regex filter;
    fs::path dir;
    std::map<std::string, int> seen;
    std::string index;
    int files = 0;
};

// llama.cpp leaves the MTP block's `pregate * sigmoid(gate)` node unnamed (qwen35moe.cpp:667), but `wo`'s Q8_0
// activation is its output, so test 2 needs it as a tap. It is the GGML_OP_MUL node whose first source is `mtp_attn_pregate-N`; the oracle
// names it `mtp_attn_gated-N`. llama.cpp is unchanged.
static std::string tap_name(const ggml_tensor * t) {
    static const char pregate[] = "mtp_attn_pregate-";
    if (t->op == GGML_OP_MUL && t->src[0] && strncmp(t->src[0]->name, pregate, sizeof pregate - 1) == 0) {
        return std::string("mtp_attn_gated-") + (t->src[0]->name + sizeof pregate - 1);
    }
    return t->name;
}

static bool on_tensor(ggml_tensor * t, bool ask, void * user_data) {
    auto * s = (Taps *) user_data;
    const std::string name = tap_name(t);
    const bool want = std::regex_match(name, s->filter);
    if (ask) return want;
    if (!want) return true;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_I32) {
        fprintf(stderr, "skipping %s: type %s\n", name.c_str(), ggml_type_name(t->type));
        return true;
    }
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    // Gather by strides, so views come out in logical order with ne[0] fastest.
    std::vector<uint8_t> out((size_t) ggml_nelements(t) * 4);
    size_t k = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
        for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
            for (int64_t i1 = 0; i1 < t->ne[1]; i1++)
                for (int64_t i0 = 0; i0 < t->ne[0]; i0++, k += 4) {
                    const uint8_t * p = raw.data() + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
                    if (t->type == GGML_TYPE_F16) {
                        const float v = ggml_fp16_to_fp32(*(const ggml_fp16_t *) p);
                        memcpy(&out[k], &v, 4);
                    } else {
                        memcpy(&out[k], p, 4);
                    }
                }
    char file[32];
    snprintf(file, sizeof file, "%05d.bin", s->files++);
    write_file(s->dir / file, out.data(), out.size());
    const int occurrence = s->seen[name]++;
    char entry[512];
    snprintf(entry, sizeof entry,
             "%s{\"name\":\"%s\",\"occurrence\":%d,\"ne\":[%lld,%lld,%lld,%lld],\"dtype\":\"%s\",\"file\":\"%s\"}",
             s->index.empty() ? "" : ",\n", name.c_str(), occurrence, (long long) t->ne[0], (long long) t->ne[1],
             (long long) t->ne[2], (long long) t->ne[3], t->type == GGML_TYPE_I32 ? "i32" : "f32", file);
    s->index += entry;
    return true;
}

static int run_taps(const Args & a) {
    if (a.tokens.empty() || a.out.empty()) die("taps needs --tokens and --out");
    const auto toks = read_tokens(a.tokens);
    fs::create_directories(a.out);
    Taps s{ std::regex(a.filter), a.out, {}, {}, 0 };
    llama_model * m = load(a, false);
    const auto n = (uint32_t) toks.size();
    llama_context * c = context(m, a, n, n, n, on_tensor, &s);
    llama_batch b = llama_batch_init((int32_t) n, 0, 1);
    for (uint32_t i = 0; i < n; i++) {
        b.token[i] = toks[i];
        b.pos[i] = (llama_pos) i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;  // logits everywhere, so the last layer's rows aren't cropped
    }
    b.n_tokens = (int32_t) n;
    if (llama_decode(c, b) != 0) die("llama_decode failed");
    const std::string json = "{\"source\":\"llama.cpp b11509\",\"backend\":\"" + a.backend + "\",\"n_tokens\":" +
                             std::to_string(n) + ",\"tensors\":[\n" + s.index + "\n]}\n";
    write_file(fs::path(a.out) / "index.json", json.data(), json.size());
    fprintf(stderr, "%d tensors written to %s\n", s.files, a.out.c_str());
    llama_batch_free(b);
    llama_free(c);
    llama_model_free(m);
    return 0;
}

// ---- mtp-taps ----

// The MTP block (`blk.40`) over the main model's own hidden rows (plan 5b M13). The target model runs over the tokens
// with the next-token embeddings on, which gives each position's final hidden row (after output_norm); those go to
// DIR/hidden.bin (raw f32 [n][n_embd]). Then an MTP context (ctx_type MTP) runs rows q = 1 .. n - 1, each pairing token
// q with the hidden row at q - 1 at position q, as the speculative driver does (common/speculative.cpp's draft-mtp
// catch-up batch), except that the driver also writes a row at position 0 (its first token paired with an all-zero
// pending hidden row), which neither this mode nor the reference's dump has; the MTP KV here starts at position 1. And the callback writes every matching tensor as `taps`
// does. The index's n_tokens counts those rows: row i is position i + 1. Both contexts are the CPU's, so the rounding is the one test 2's CPU taps use.
static int run_mtp_taps(const Args & a) {
    if (a.backend != "cpu") die("mtp-taps is CPU only: --backend cpu");
    if (a.tokens.empty() || a.out.empty()) die("mtp-taps needs --tokens and --out");
    const auto toks = read_tokens(a.tokens);
    if (toks.size() < 2) die("mtp-taps needs at least 2 tokens: row q pairs token q with the hidden row at q - 1");
    fs::create_directories(a.out);
    Taps s{ std::regex(a.filter), a.out, {}, {}, 0 };
    llama_model * m = load(a, false);
    const auto n = (uint32_t) toks.size();
    const size_t n_embd = (size_t) llama_model_n_embd_out(m);

    // The target's pass: no callback, hidden rows for every position (masked == false).
    std::vector<float> hidden(n * n_embd);
    {
        llama_context * t = context(m, a, n, n, n, nullptr, nullptr);
        llama_set_embeddings_nextn(t, true, /*masked*/ false);
        llama_batch b = llama_batch_init((int32_t) n, 0, 1);
        for (uint32_t i = 0; i < n; i++) {
            b.token[i] = toks[i];
            b.pos[i] = (llama_pos) i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = 1;
        }
        b.n_tokens = (int32_t) n;
        if (llama_decode(t, b) != 0) die("llama_decode failed on the target");
        for (uint32_t i = 0; i < n; i++) {
            const float * h = llama_get_embeddings_nextn_ith(t, (int32_t) i);
            if (!h) die("the target gave no hidden row " + std::to_string(i));
            memcpy(&hidden[i * n_embd], h, n_embd * sizeof(float));
        }
        llama_batch_free(b);
        llama_free(t);
    }
    write_file(fs::path(a.out) / "hidden.bin", hidden.data(), hidden.size() * sizeof(float));

    // The MTP pass over rows 1 .. n - 1.
    llama_context * c = context(m, a, n, n, n, on_tensor, &s, LLAMA_CONTEXT_TYPE_MTP);
    llama_batch_ext * b = llama_batch_ext_init(c);
    for (uint32_t q = 1; q < n; q++) {
        const int32_t idx = llama_batch_ext_add_token(b, 0, toks[q]);
        if (idx < 0) die("llama_batch_ext_add_token failed (" + std::to_string(idx) + ")");
        const llama_pos pos[4] = { (llama_pos) q, 0, 0, 0 };  // a text token: only pos[0] is read
        if (!llama_batch_ext_set_pos(b, idx, pos)) die("llama_batch_ext_set_pos failed");
        const llama_embd h = { &hidden[(size_t) (q - 1) * n_embd], 1, n_embd };
        if (!llama_batch_ext_set_embd_token(b, idx, h)) die("llama_batch_ext_set_embd_token failed");
        if (!llama_batch_ext_set_output_logits(b, idx, true)) die("llama_batch_ext_set_output_logits failed");
    }
    if (llama_process(c, LLAMA_PROCESS_TYPE_DECODE, b) != 0) die("llama_process failed on the MTP context");
    const std::string json = "{\"source\":\"llama.cpp b11509\",\"backend\":\"" + a.backend + "\",\"n_tokens\":" +
                             std::to_string(n - 1) + ",\"tensors\":[\n" + s.index + "\n]}\n";
    write_file(fs::path(a.out) / "index.json", json.data(), json.size());
    fprintf(stderr, "%d tensors for %u MTP rows written to %s\n", s.files, n - 1, a.out.c_str());
    llama_batch_ext_free(b);
    llama_free(c);
    llama_model_free(m);
    return 0;
}

// ---- greedy and forced ----

// The prompt in batches of 2048 tokens (ubatches of 512), with logits for its last token only.
static void decode_prompt(llama_context * c, const std::vector<llama_token> & prompt) {
    for (size_t i = 0; i < prompt.size(); i += 2048) {
        const size_t n = std::min<size_t>(2048, prompt.size() - i);
        llama_batch b = llama_batch_init((int32_t) n, 0, 1);
        for (size_t j = 0; j < n; j++) {
            b.token[j] = prompt[i + j];
            b.pos[j] = (llama_pos) (i + j);
            b.n_seq_id[j] = 1;
            b.seq_id[j][0] = 0;
            b.logits[j] = (i + j + 1 == prompt.size());
        }
        b.n_tokens = (int32_t) n;
        if (llama_decode(c, b) != 0) die("llama_decode failed on the prompt");
        llama_batch_free(b);
    }
}

// The two highest logits. Strict comparisons from id 1, so a tie goes to the lower id.
static void top_two(const float * l, int32_t n_vocab, int & best, int & second) {
    best = 0;
    second = -1;
    for (int i = 1; i < n_vocab; i++) {
        if (l[i] > l[best]) {
            second = best;
            best = i;
        } else if (second < 0 || l[i] > l[second]) {
            second = i;
        }
    }
}

static int run_greedy(const Args & a) {
    if (a.tokens.empty() || a.out.empty()) die("greedy needs --tokens and --out");
    const auto prompt = read_tokens(a.tokens);
    if (prompt.empty()) die("the prompt is empty");
    llama_model * m = load(a, false);
    const llama_vocab * v = llama_model_get_vocab(m);
    const int32_t n_vocab = llama_vocab_n_tokens(v);
    llama_context * c = context(m, a, (uint32_t) (prompt.size() + a.n + 16), 2048, 512, nullptr, nullptr);
    decode_prompt(c, prompt);
    std::string steps;
    bool eog = false;
    for (int step = 0; step < a.n; step++) {
        const float * l = llama_get_logits_ith(c, -1);
        int best, second;
        top_two(l, n_vocab, best, second);
        char e[192];
        snprintf(e, sizeof e, "%s{\"token\":%d,\"logit\":%.9g,\"second\":%d,\"second_logit\":%.9g}",
                 steps.empty() ? "" : ",\n", best, l[best], second, l[second]);
        steps += e;
        if (llama_vocab_is_eog(v, best)) {
            eog = true;
            break;
        }
        llama_token tok = best;
        if (llama_decode(c, llama_batch_get_one(&tok, 1)) != 0) die("llama_decode failed while generating");
    }
    const std::string json = "{\"source\":\"llama.cpp b11509\",\"backend\":\"" + a.backend + "\",\"prompt_tokens\":" +
                             std::to_string(prompt.size()) + ",\"stopped_at_eog\":" + (eog ? "true" : "false") +
                             ",\"steps\":[\n" + steps + "\n]}\n";
    write_file(a.out, json.data(), json.size());
    llama_free(c);
    llama_model_free(m);
    return 0;
}

// Teacher forcing (test 3's floor): the prompt as greedy reads it, then the tokens in --force, fed one at a time.
// Step k's logits follow the prompt and forced tokens 0..k-1. Each step records the token fed there (`token`), the
// run's own top two, and the fed token's logit. The last token isn't fed: nothing reads its logits.
static int run_forced(const Args & a) {
    if (a.tokens.empty() || a.force.empty() || a.out.empty()) die("forced needs --tokens, --force and --out");
    const auto prompt = read_tokens(a.tokens);
    const auto forced = read_tokens(a.force);
    if (prompt.empty()) die("the prompt is empty");
    if (forced.empty()) die(a.force + " holds no tokens");
    if (forced.size() > (size_t) a.n) {
        die(a.force + " holds " + std::to_string(forced.size()) + " tokens, more than --n " + std::to_string(a.n));
    }
    llama_model * m = load(a, false);
    const llama_vocab * v = llama_model_get_vocab(m);
    const int32_t n_vocab = llama_vocab_n_tokens(v);
    for (size_t k = 0; k < forced.size(); k++) {
        if (forced[k] < 0 || forced[k] >= n_vocab) {
            die("forced token " + std::to_string(k) + " is outside the vocabulary");
        }
        if (k + 1 < forced.size() && llama_vocab_is_eog(v, forced[k])) {
            die("forced token " + std::to_string(k) + " ends generation; only the last one may");
        }
    }
    // Greedy's context for --n steps, so the two modes compute the same logits until their tokens differ.
    llama_context * c = context(m, a, (uint32_t) (prompt.size() + a.n + 16), 2048, 512, nullptr, nullptr);
    decode_prompt(c, prompt);
    std::string steps;
    for (size_t k = 0; k < forced.size(); k++) {
        const float * l = llama_get_logits_ith(c, -1);
        int best, second;
        top_two(l, n_vocab, best, second);
        char e[256];
        snprintf(e, sizeof e,
                 "%s{\"token\":%d,\"top\":%d,\"logit\":%.9g,\"second\":%d,\"second_logit\":%.9g,\"token_logit\":%.9g}",
                 steps.empty() ? "" : ",\n", forced[k], best, l[best], second, l[second], l[forced[k]]);
        steps += e;
        if (k + 1 == forced.size()) break;
        llama_token tok = forced[k];
        if (llama_decode(c, llama_batch_get_one(&tok, 1)) != 0) die("llama_decode failed on a forced token");
    }
    const std::string json = "{\"source\":\"llama.cpp b11509\",\"backend\":\"" + a.backend + "\",\"prompt_tokens\":" +
                             std::to_string(prompt.size()) + ",\"steps\":[\n" + steps + "\n]}\n";
    write_file(a.out, json.data(), json.size());
    fprintf(stderr, "%zu forced steps written to %s\n", forced.size(), a.out.c_str());
    llama_free(c);
    llama_model_free(m);
    return 0;
}

// ---- tokenize ----

static int run_tokenize(const Args & a) {
    if (a.file.empty() || a.out.empty()) die("tokenize needs --file and --out");
    const auto text = read_file(a.file);  // binary: CRLF and every other byte reach the tokenizer unchanged
    llama_model * m = load(a, true);
    const llama_vocab * v = llama_model_get_vocab(m);
    std::vector<llama_token> ids(text.size() + 16);
    const int32_t n = llama_tokenize(v, (const char *) text.data(), (int32_t) text.size(), ids.data(),
                                     (int32_t) ids.size(), true, !a.no_parse_special);
    if (n < 0) die("the token buffer is too small");
    write_file(a.out, ids.data(), (size_t) n * 4);
    fprintf(stderr, "%d tokens\n", n);
    llama_model_free(m);
    return 0;
}

// ---- vocab ----

static int run_vocab(const Args & a) {
    if (a.out.empty()) die("vocab needs --out");
    llama_model * m = load(a, true);
    const llama_vocab * v = llama_model_get_vocab(m);
    std::string eog, control, user;
    auto add = [](std::string & s, int i) { s += (s.empty() ? "" : ",") + std::to_string(i); };
    const int32_t n = llama_vocab_n_tokens(v);
    for (llama_token i = 0; i < n; i++) {
        if (llama_vocab_is_eog(v, i)) add(eog, i);
        const auto attr = llama_vocab_get_attr(v, i);
        if (attr & LLAMA_TOKEN_ATTR_CONTROL) add(control, i);
        if (attr & LLAMA_TOKEN_ATTR_USER_DEFINED) add(user, i);
    }
    const std::string json = "{\"n_vocab\":" + std::to_string(n) + ",\"eog\":[" + eog + "],\"control\":[" + control +
                             "],\"user_defined\":[" + user + "]}\n";
    write_file(a.out, json.data(), json.size());
    llama_model_free(m);
    return 0;
}

int main(int argc, char ** argv) {
    const Args a = parse(argc, argv);
    if (a.mode == "taps" || a.mode == "mtp-taps") {
        // Keep RMS_NORM and MUL separate, so the unweighted norms exist as tensors.
#ifdef _WIN32
        _putenv_s("GGML_CPU_DISABLE_FUSION", "1");
#else
        setenv("GGML_CPU_DISABLE_FUSION", "1", 1);
#endif
    }
    llama_backend_init();
    ggml_backend_load_all();
    int rc = 1;
    if (a.mode == "taps") rc = run_taps(a);
    else if (a.mode == "mtp-taps") rc = run_mtp_taps(a);
    else if (a.mode == "greedy") rc = run_greedy(a);
    else if (a.mode == "forced") rc = run_forced(a);
    else if (a.mode == "tokenize") rc = run_tokenize(a);
    else if (a.mode == "vocab") rc = run_vocab(a);
    else die("unknown mode " + a.mode);
    llama_backend_free();
    return rc;
}
