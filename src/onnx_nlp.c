#include "onnx_nlp.h"
#include "app_log.h"
#include "utils.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>
#include <curl/curl.h>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef USE_ONNX
#include <onnxruntime_c_api.h>
#endif

/* ── Model URLs ──────────────────────────────────────────────── */

#define NER_URL       "https://huggingface.co/dslim/bert-base-NER/resolve/main/onnx/model.onnx"
#define EMBED_URL     "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
#define VOCAB_URL     "https://huggingface.co/dslim/bert-base-NER/resolve/main/vocab.txt"
/* NLI (distilbart-mnli) — tokenizer from HuggingFace, model from GitHub release */
#define NLI_MODEL_URL "https://github.com/aivrar/serp-models/releases/download/v1.0/nli.onnx"
#define NLI_VOCAB_URL "https://huggingface.co/valhalla/distilbart-mnli-12-3/resolve/main/vocab.json"
#define NLI_MERGE_URL "https://huggingface.co/valhalla/distilbart-mnli-12-3/resolve/main/merges.txt"

/* ── NER labels ──────────────────────────────────────────────── */
static const char *NER_LABELS[] = {
    "O", "B-MISC", "I-MISC", "B-PER", "I-PER",
    "B-ORG", "I-ORG", "B-LOC", "I-LOC"
};
#define NER_LABEL_COUNT 9

/* ── Context ─────────────────────────────────────────────────── */

struct OnnxNLP {
    int available;
    char model_dir[512];
    int use_gpu;
    char **vocab;
    int vocab_size;
#ifdef USE_ONNX
    const OrtApi *api;
    OrtEnv *env;
    OrtSession *ner_session;
    OrtSession *embed_session;
    OrtSessionOptions *opts;
#endif
};

/* ── Model checks ────────────────────────────────────────────── */

int onnx_nlp_models_present(const char *model_dir) {
    char path[512];
    static const char *required[] = {
        "ner.onnx", "embed.onnx", "vocab.txt",
        "nli.onnx", "nli_vocab.json", "nli_merges.txt"
    };
    for (int i = 0; i < 6; i++) {
        snprintf(path, sizeof(path), "%s/%s", model_dir, required[i]);
        if (!file_exists(path)) return 0;
    }
    return 1;
}

/* ── Download ────────────────────────────────────────────────── */

volatile int g_download_cancel = 0;

static size_t dl_write_cb(void *contents, size_t size, size_t nmemb, void *userp) {
    FILE *f = (FILE *)userp;
    return fwrite(contents, size, nmemb, f);
}

static int dl_progress_cb(void *clientp, double dltotal, double dlnow,
                          double ultotal, double ulnow) {
    (void)clientp; (void)dltotal; (void)dlnow; (void)ultotal; (void)ulnow;
    return g_download_cancel ? 1 : 0;  /* non-zero aborts transfer */
}

static int download_file(const char *url, const char *dest) {
    app_log(LOG_INFO, "Downloading: %s", url);
    app_log(LOG_INFO, "  -> %s", dest);

    /* Use curl directly with long timeout, streaming to file, no brotli */
    CURL *curl = curl_easy_init();
    if (!curl) { app_log(LOG_ERROR, "curl_easy_init failed"); return -1; }

    FILE *f = fopen(dest, "wb");
    if (!f) { app_log(LOG_ERROR, "Cannot create: %s", dest); curl_easy_cleanup(curl); return -1; }

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, dl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, f);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 600L);         /* 10 min for large files */
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, http_random_ua());
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "gzip, deflate"); /* no brotli */
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1000L);  /* abort if < 1KB/s */
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);     /* for 60 seconds */
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, dl_progress_cb);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    double dl_size = 0;
    curl_easy_getinfo(curl, CURLINFO_SIZE_DOWNLOAD, &dl_size);

    fclose(f);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        app_log(LOG_ERROR, "Download failed: %s - %s", url, curl_easy_strerror(res));
        remove(dest);  /* remove partial file */
        return -1;
    }

    if (http_code != 200) {
        app_log(LOG_ERROR, "Download HTTP %ld: %s", http_code, url);
        remove(dest);
        return -1;
    }

    app_log(LOG_INFO, "Downloaded %.1f MB -> %s", dl_size / (1024.0*1024.0), dest);
    return 0;
}

int onnx_nlp_download_models(const char *model_dir, onnx_download_cb cb, void *userdata) {
    (void)cb; (void)userdata;
    ensure_directory(model_dir);
    char path[512];
    int rc = 0;

    snprintf(path, sizeof(path), "%s/vocab.txt", model_dir);
    if (!file_exists(path)) { if (download_file(VOCAB_URL, path) != 0) rc = -1; }

    snprintf(path, sizeof(path), "%s/ner.onnx", model_dir);
    if (!file_exists(path)) { if (download_file(NER_URL, path) != 0) rc = -1; }

    snprintf(path, sizeof(path), "%s/embed.onnx", model_dir);
    if (!file_exists(path)) { if (download_file(EMBED_URL, path) != 0) rc = -1; }

    /* NLI model + tokenizer files */
    snprintf(path, sizeof(path), "%s/nli.onnx", model_dir);
    if (!file_exists(path)) { if (download_file(NLI_MODEL_URL, path) != 0) rc = -1; }

    snprintf(path, sizeof(path), "%s/nli_vocab.json", model_dir);
    if (!file_exists(path)) { if (download_file(NLI_VOCAB_URL, path) != 0) rc = -1; }

    snprintf(path, sizeof(path), "%s/nli_merges.txt", model_dir);
    if (!file_exists(path)) { if (download_file(NLI_MERGE_URL, path) != 0) rc = -1; }

    return rc;
}

/* ── Vocabulary with hash table for O(1) lookup ──────────────── */

#define VOCAB_HASH_SIZE 131071  /* prime, ~2.6x vocab size */

typedef struct VocabNode {
    char *token;
    int id;
    struct VocabNode *next;
} VocabNode;

static VocabNode *g_vocab_hash[VOCAB_HASH_SIZE];

static unsigned int vocab_hash(const char *s) {
    unsigned int h = 5381;
    while (*s) h = ((h << 5) + h) + (unsigned char)*s++;
    return h % VOCAB_HASH_SIZE;
}

static void vocab_hash_insert(const char *token, int id) {
    unsigned int h = vocab_hash(token);
    VocabNode *n = (VocabNode *)malloc(sizeof(VocabNode));
    if (!n) return;
    n->token = str_duplicate(token);
    if (!n->token) { free(n); return; }
    n->id = id;
    n->next = g_vocab_hash[h];
    g_vocab_hash[h] = n;
}

static void vocab_hash_free(void) {
    for (int i = 0; i < VOCAB_HASH_SIZE; i++) {
        VocabNode *n = g_vocab_hash[i];
        while (n) { VocabNode *next = n->next; free(n->token); free(n); n = next; }
        g_vocab_hash[i] = NULL;
    }
}

static int load_vocab(OnnxNLP *ctx) {
    char path[512];
    snprintf(path, sizeof(path), "%s/vocab.txt", ctx->model_dir);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    ctx->vocab = (char **)calloc(50000, sizeof(char *));
    if (!ctx->vocab) { fclose(f); return -1; }
    ctx->vocab_size = 0;
    memset(g_vocab_hash, 0, sizeof(g_vocab_hash));
    char line[256];
    while (fgets(line, sizeof(line), f) && ctx->vocab_size < 50000) {
        str_trim(line);
        if (line[0]) {
            ctx->vocab[ctx->vocab_size] = str_duplicate(line);
            vocab_hash_insert(line, ctx->vocab_size);
            ctx->vocab_size++;
        }
    }
    fclose(f);
    app_log(LOG_INFO, "ONNX vocab: %d tokens (hash table built)", ctx->vocab_size);
    return 0;
}

static void free_vocab(OnnxNLP *ctx) {
    if (ctx->vocab) {
        for (int i = 0; i < ctx->vocab_size; i++) free(ctx->vocab[i]);
        free(ctx->vocab);
        ctx->vocab = NULL;
    }
    vocab_hash_free();
}

static int vocab_lookup(OnnxNLP *ctx, const char *token) {
    (void)ctx;
    unsigned int h = vocab_hash(token);
    VocabNode *n = g_vocab_hash[h];
    while (n) {
        if (strcmp(n->token, token) == 0) return n->id;
        n = n->next;
    }
    return -1;
}

/* ── WordPiece tokenizer ─────────────────────────────────────── */

#define MAX_TOKENS 256  /* reduced from 512 to avoid stack overflow; truncates long text */

typedef struct {
    int ids[MAX_TOKENS];
    int attention[MAX_TOKENS];
    int type_ids[MAX_TOKENS];
    char pieces[MAX_TOKENS][64]; /* text per token for entity reconstruction */
    int count;
} TokenResult;

static void tokenize(OnnxNLP *ctx, const char *text, TokenResult *out) {
    memset(out, 0, sizeof(TokenResult));

    int cls_id = vocab_lookup(ctx, "[CLS]");
    int sep_id = vocab_lookup(ctx, "[SEP]");
    int unk_id = vocab_lookup(ctx, "[UNK]");
    if (cls_id < 0) cls_id = 101;
    if (sep_id < 0) sep_id = 102;
    if (unk_id < 0) unk_id = 100;

    /* [CLS] */
    out->ids[0] = cls_id;
    out->attention[0] = 1;
    snprintf(out->pieces[0], 64, "[CLS]");
    out->count = 1;

    /* Split text into words, tokenize each */
    const char *p = text;
    int word_idx = 0;

    while (*p && out->count < MAX_TOKENS - 1) {
        /* Skip whitespace and punctuation */
        while (*p && (isspace((unsigned char)*p) || ispunct((unsigned char)*p))) {
            /* Punctuation gets its own token if in vocab */
            if (ispunct((unsigned char)*p) && out->count < MAX_TOKENS - 1) {
                char punc[4] = {*p, '\0'};
                int pid = vocab_lookup(ctx, punc);
                if (pid >= 0) {
                    out->ids[out->count] = pid;
                    out->attention[out->count] = 1;
                    snprintf(out->pieces[out->count], 64, "%s", punc);
                    out->count++;
                }
            }
            p++;
        }
        if (!*p) break;

        /* Extract word — keep BOTH original case and lowercase.
           bert-base-NER is a CASED model: capitalization is the primary
           signal for entity recognition.  Try original case first in
           vocab lookup, fall back to lowercase only if not found.
           (Bug fix 2026-03-31: was always lowercasing, destroying the
           casing signal — model predicted O for everything.) */
        char orig[128], lower[128];
        int wlen = 0;
        while (*p && !isspace((unsigned char)*p) && !ispunct((unsigned char)*p) && wlen < 126) {
            orig[wlen] = *p;
            lower[wlen] = (char)tolower((unsigned char)*p);
            wlen++; p++;
        }
        orig[wlen] = '\0';
        lower[wlen] = '\0';

        /* Try full word: original case first, then lowercase */
        int id = vocab_lookup(ctx, orig);
        if (id < 0) id = vocab_lookup(ctx, lower);
        if (id >= 0) {
            out->ids[out->count] = id;
            out->attention[out->count] = 1;
            snprintf(out->pieces[out->count], 64, "%s", orig);
            out->count++;
        } else {
            /* WordPiece: greedy longest-match from left.
               Try original case first, fall back to lowercase. */
            int pos = 0;
            int is_first = 1;
            while (pos < wlen && out->count < MAX_TOKENS - 1) {
                int best_end = -1, best_id = -1;

                /* Try longest substring first — original case */
                for (int end = wlen; end > pos; end--) {
                    char sub[140];
                    int slen = end - pos;
                    if (is_first) {
                        memcpy(sub, orig + pos, slen);
                        sub[slen] = '\0';
                    } else {
                        sub[0] = '#'; sub[1] = '#';
                        memcpy(sub + 2, orig + pos, slen);
                        sub[slen + 2] = '\0';
                    }
                    int sid = vocab_lookup(ctx, sub);
                    if (sid >= 0) {
                        best_end = end;
                        best_id = sid;
                        break;
                    }
                }

                /* Fall back to lowercase if cased lookup failed */
                if (best_id < 0) {
                    for (int end = wlen; end > pos; end--) {
                        char sub[140];
                        int slen = end - pos;
                        if (is_first) {
                            memcpy(sub, lower + pos, slen);
                            sub[slen] = '\0';
                        } else {
                            sub[0] = '#'; sub[1] = '#';
                            memcpy(sub + 2, lower + pos, slen);
                            sub[slen + 2] = '\0';
                        }
                        int sid = vocab_lookup(ctx, sub);
                        if (sid >= 0) {
                            best_end = end;
                            best_id = sid;
                            break;
                        }
                    }
                }

                if (best_id >= 0) {
                    out->ids[out->count] = best_id;
                    out->attention[out->count] = 1;
                    int n = best_end - pos;
                    if (is_first) {
                        memcpy(out->pieces[out->count], orig + pos, n);
                        out->pieces[out->count][n] = '\0';
                    } else {
                        out->pieces[out->count][0] = '#';
                        out->pieces[out->count][1] = '#';
                        memcpy(out->pieces[out->count] + 2, orig + pos, n);
                        out->pieces[out->count][n + 2] = '\0';
                    }
                    out->count++;
                    pos = best_end;
                } else {
                    /* No match at all -- emit [UNK] for rest of word */
                    out->ids[out->count] = unk_id;
                    out->attention[out->count] = 1;
                    snprintf(out->pieces[out->count], 64, "%s", orig);
                    out->count++;
                    break;
                }
                is_first = 0;
            }
        }
        word_idx++;
    }

    /* [SEP] */
    out->ids[out->count] = sep_id;
    out->attention[out->count] = 1;
    snprintf(out->pieces[out->count], 64, "[SEP]");
    out->count++;
}

/* ── Lifecycle ───────────────────────────────────────────────── */

int onnx_nlp_init(OnnxNLP **ctx, const char *model_dir, int use_gpu) {
    *ctx = (OnnxNLP *)calloc(1, sizeof(OnnxNLP));
    if (!*ctx) return -1;
    snprintf((*ctx)->model_dir, sizeof((*ctx)->model_dir), "%s", model_dir);
    (*ctx)->use_gpu = use_gpu;

    if (!onnx_nlp_models_present(model_dir)) {
        app_log(LOG_INFO, "ONNX NLP: models not in %s -- download from Settings", model_dir);
        (*ctx)->available = 0;
        return 0;
    }

    if (load_vocab(*ctx) != 0) {
        (*ctx)->available = 0;
        return 0;
    }

#ifdef USE_ONNX
    const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    (*ctx)->api = api;

    OrtStatus *status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "serp_nlp", &(*ctx)->env);
    if (status) {
        app_log(LOG_ERROR, "ONNX: CreateEnv failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        (*ctx)->available = 0;
        return 0;
    }

    api->CreateSessionOptions(&(*ctx)->opts);
    api->SetIntraOpNumThreads((*ctx)->opts, 4);
    api->SetSessionGraphOptimizationLevel((*ctx)->opts, ORT_ENABLE_ALL);

    /* GPU provider -- dynamically try CUDA if onnxruntime-gpu is installed.
       OrtSessionOptionsAppendExecutionProvider_CUDA is only in the GPU build,
       so we load it at runtime to avoid link errors with the CPU build. */
    if (use_gpu) {
        typedef OrtStatus* (ORT_API_CALL *CudaProviderFn)(OrtSessionOptions*, int);
        HMODULE hort = GetModuleHandleA("onnxruntime.dll");
        CudaProviderFn cuda_fn = hort ?
            (CudaProviderFn)GetProcAddress(hort, "OrtSessionOptionsAppendExecutionProvider_CUDA") : NULL;
        if (cuda_fn) {
            OrtStatus *gpu_status = cuda_fn((*ctx)->opts, 0);
            if (gpu_status) {
                app_log(LOG_INFO, "ONNX: CUDA init failed (%s) -- using CPU",
                        api->GetErrorMessage(gpu_status));
                api->ReleaseStatus(gpu_status);
            } else {
                app_log(LOG_INFO, "ONNX: CUDA GPU enabled (device 0)");
            }
        } else {
            app_log(LOG_INFO, "ONNX: CPU build (install onnxruntime-gpu for GPU acceleration)");
        }
    }

    /* Load NER model */
    {char ner_path[512];
    snprintf(ner_path, sizeof(ner_path), "%s/ner.onnx", model_dir);
    wchar_t wpath[512];
    MultiByteToWideChar(CP_UTF8, 0, ner_path, -1, wpath, 512);
    status = api->CreateSession((*ctx)->env, wpath, (*ctx)->opts, &(*ctx)->ner_session);
    if (status) {
        app_log(LOG_ERROR, "ONNX: NER model load failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        (*ctx)->ner_session = NULL;
    } else {
        app_log(LOG_INFO, "ONNX: NER model loaded from %s", ner_path);
    }}

    /* Load embedding model */
    {char embed_path[512];
    snprintf(embed_path, sizeof(embed_path), "%s/embed.onnx", model_dir);
    wchar_t wpath[512];
    MultiByteToWideChar(CP_UTF8, 0, embed_path, -1, wpath, 512);
    status = api->CreateSession((*ctx)->env, wpath, (*ctx)->opts, &(*ctx)->embed_session);
    if (status) {
        app_log(LOG_WARN, "ONNX: Embed model load failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        (*ctx)->embed_session = NULL;
    } else {
        app_log(LOG_INFO, "ONNX: Embedding model loaded");
    }}

    (*ctx)->available = ((*ctx)->ner_session != NULL);
    app_log(LOG_INFO, "ONNX NLP: %s (NER=%s, Embed=%s)",
            (*ctx)->available ? "ready" : "partial",
            (*ctx)->ner_session ? "yes" : "no",
            (*ctx)->embed_session ? "yes" : "no");
#else
    app_log(LOG_INFO, "ONNX NLP: not compiled (USE_ONNX not defined)");
    (*ctx)->available = 0;
#endif

    return 0;
}

void onnx_nlp_shutdown(OnnxNLP *ctx) {
    if (!ctx) return;
#ifdef USE_ONNX
    if (ctx->ner_session) ctx->api->ReleaseSession(ctx->ner_session);
    if (ctx->embed_session) ctx->api->ReleaseSession(ctx->embed_session);
    if (ctx->opts) ctx->api->ReleaseSessionOptions(ctx->opts);
    if (ctx->env) ctx->api->ReleaseEnv(ctx->env);
#endif
    free_vocab(ctx);
    free(ctx);
}

int onnx_nlp_available(OnnxNLP *ctx) {
    return ctx ? ctx->available : 0;
}

void *onnx_nlp_get_api(OnnxNLP *ctx) {
#ifdef USE_ONNX
    return ctx ? (void *)ctx->api : NULL;
#else
    (void)ctx; return NULL;
#endif
}
void *onnx_nlp_get_env(OnnxNLP *ctx) {
#ifdef USE_ONNX
    return ctx ? (void *)ctx->env : NULL;
#else
    (void)ctx; return NULL;
#endif
}
void *onnx_nlp_get_opts(OnnxNLP *ctx) {
#ifdef USE_ONNX
    return ctx ? (void *)ctx->opts : NULL;
#else
    (void)ctx; return NULL;
#endif
}

/* ── NER inference ───────────────────────────────────────────── */

int onnx_nlp_extract_entities(OnnxNLP *ctx, const char *text,
                               NLPEntity *out, int max_entities) {
    if (!ctx || !ctx->available) return 0;

#ifdef USE_ONNX
    if (!ctx->ner_session) return 0;
    const OrtApi *api = ctx->api;

    /* Tokenize (heap-allocated to avoid stack overflow) */
    TokenResult *tok = (TokenResult *)calloc(1, sizeof(TokenResult));
    if (!tok) return 0;
    tokenize(ctx, text, tok);
    if (tok->count < 3) { free(tok); return 0; }

    /* Create input tensors (heap-allocated) */
    int64_t shape[] = {1, tok->count};
    int64_t *input_ids = (int64_t *)calloc(tok->count, sizeof(int64_t));
    int64_t *attn_mask = (int64_t *)calloc(tok->count, sizeof(int64_t));
    int64_t *type_ids = (int64_t *)calloc(tok->count, sizeof(int64_t));
    if (!input_ids || !attn_mask || !type_ids) {
        free(input_ids); free(attn_mask); free(type_ids); free(tok); return 0;
    }
    for (int i = 0; i < tok->count; i++) {
        input_ids[i] = tok->ids[i];
        attn_mask[i] = tok->attention[i];
        type_ids[i] = 0;
    }

    OrtMemoryInfo *mem_info;
    api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);

    OrtValue *input_tensors[3] = {NULL, NULL, NULL};
    api->CreateTensorWithDataAsOrtValue(mem_info, input_ids, tok->count * sizeof(int64_t),
        shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensors[0]);
    api->CreateTensorWithDataAsOrtValue(mem_info, attn_mask, tok->count * sizeof(int64_t),
        shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensors[1]);
    api->CreateTensorWithDataAsOrtValue(mem_info, type_ids, tok->count * sizeof(int64_t),
        shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensors[2]);

    const char *input_names[] = {"input_ids", "attention_mask", "token_type_ids"};
    const char *output_names[] = {"logits"};
    OrtValue *output = NULL;

    OrtStatus *status = api->Run(ctx->ner_session, NULL,
        input_names, (const OrtValue *const *)input_tensors, 3,
        output_names, 1, &output);

    api->ReleaseMemoryInfo(mem_info);

    if (status) {
        app_log(LOG_ERROR, "ONNX NER Run failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        for (int i = 0; i < 3; i++) if (input_tensors[i]) api->ReleaseValue(input_tensors[i]);
        free(input_ids); free(attn_mask); free(type_ids); free(tok);
        return 0;
    }

    /* Parse logits → entity labels */
    float *logits = NULL;
    api->GetTensorMutableData(output, (void **)&logits);

    int entity_count = 0;
    char current_entity[256] = "";
    char current_label[16] = "";
    int ent_len = 0;  /* tracks strlen(current_entity) to avoid repeated strlen calls */

    for (int t = 1; t < tok->count - 1 && entity_count < max_entities; t++) {
        /* Argmax over NER_LABEL_COUNT classes */
        int best = 0;
        float best_val = logits[t * NER_LABEL_COUNT];
        for (int c = 1; c < NER_LABEL_COUNT; c++) {
            if (logits[t * NER_LABEL_COUNT + c] > best_val) {
                best_val = logits[t * NER_LABEL_COUNT + c];
                best = c;
            }
        }

        const char *label = NER_LABELS[best];
        int is_subword = (tok->pieces[t][0] == '#' && tok->pieces[t][1] == '#');

        /* Helper: append piece to current entity (uses ent_len instead of strlen) */
        #define APPEND_PIECE(piece) do { \
            if ((piece)[0] == '#' && (piece)[1] == '#') { \
                int _space = (int)sizeof(current_entity) - ent_len - 1; \
                if (_space > 0) { strncat(current_entity + ent_len, (piece) + 2, _space); ent_len = (int)strlen(current_entity); } \
            } else if (ent_len > 0) { \
                int _space = (int)sizeof(current_entity) - ent_len - 1; \
                if (_space > 1) { current_entity[ent_len] = ' '; current_entity[ent_len+1] = '\0'; ent_len++; _space--; \
                    strncat(current_entity + ent_len, (piece), _space); ent_len = (int)strlen(current_entity); } \
            } else { \
                ent_len = snprintf(current_entity, sizeof(current_entity), "%s", (piece)); \
                if (ent_len >= (int)sizeof(current_entity)) ent_len = (int)sizeof(current_entity) - 1; \
            } \
        } while(0)

        /* Helper: flush current entity if valid (uses ent_len instead of strlen) */
        #define FLUSH_ENTITY() do { \
            if (ent_len >= 3 && entity_count < max_entities \
                && current_entity[0] != '#' /* reject ##subword fragments */ \
                && !(current_entity[0] >= 'a' && current_entity[0] <= 'z' && ent_len <= 3) /* reject short lowercase */ \
                ) { \
                snprintf(out[entity_count].text, sizeof(out[entity_count].text), "%s", current_entity); \
                snprintf(out[entity_count].label, sizeof(out[entity_count].label), "%s", current_label); \
                out[entity_count].frequency = 1; \
                out[entity_count].source_count = 1; \
                entity_count++; \
            } \
            current_entity[0] = '\0'; \
            ent_len = 0; \
        } while(0)

        if (label[0] == 'B') {
            /* If this is a subword (##...) and we have an active entity of the same type,
               treat it as continuation, not a new entity */
            if (is_subword && current_entity[0] && strcmp(current_label, label + 2) == 0) {
                APPEND_PIECE(tok->pieces[t]);
            } else {
                /* Flush previous entity, start new one */
                FLUSH_ENTITY();
                ent_len = snprintf(current_entity, sizeof(current_entity), "%s", tok->pieces[t]);
                if (ent_len >= (int)sizeof(current_entity)) ent_len = (int)sizeof(current_entity) - 1;
                snprintf(current_label, sizeof(current_label), "%s", label + 2);
            }
        } else if (label[0] == 'I') {
            if (current_entity[0]) {
                /* Continue current entity */
                APPEND_PIECE(tok->pieces[t]);
            } else {
                /* I without B -- start new entity (model error, but handle gracefully) */
                ent_len = snprintf(current_entity, sizeof(current_entity), "%s",
                         is_subword ? tok->pieces[t] + 2 : tok->pieces[t]);
                if (ent_len >= (int)sizeof(current_entity)) ent_len = (int)sizeof(current_entity) - 1;
                snprintf(current_label, sizeof(current_label), "%s", label + 2);
            }
        } else {
            /* O label -- flush */
            FLUSH_ENTITY();
        }

        #undef APPEND_PIECE
        #undef FLUSH_ENTITY
    }

    /* Flush last entity */
    if (ent_len >= 3 && entity_count < max_entities
        && current_entity[0] != '#') {
        snprintf(out[entity_count].text, sizeof(out[entity_count].text), "%s", current_entity);
        snprintf(out[entity_count].label, sizeof(out[entity_count].label), "%s", current_label);
        out[entity_count].frequency = 1;
        out[entity_count].source_count = 1;
        entity_count++;
    }

    /* Cleanup */
    if (output) api->ReleaseValue(output);
    for (int i = 0; i < 3; i++) if (input_tensors[i]) api->ReleaseValue(input_tensors[i]);
    free(input_ids); free(attn_mask); free(type_ids); free(tok);

    return entity_count;
#else
    (void)text; (void)out; (void)max_entities;
    return 0;
#endif
}

int onnx_nlp_extract_entities_batch(OnnxNLP *ctx,
                                     const char **texts, const char **domains,
                                     int text_count,
                                     NLPEntity *out, int max_entities) {
    if (!ctx || !ctx->available) return 0;

    /* Process each text, merge + dedup entities */
    int total = 0;
    for (int d = 0; d < text_count && total < max_entities; d++) {
        /* Truncate to ~500 words for BERT's 512 token limit */
        char chunk[4096];
        snprintf(chunk, sizeof(chunk), "%.4000s", texts[d]);

        NLPEntity page_entities[50];
        int found = 0;

        /* Protect against crashes in ONNX inference */
        __try {
            found = onnx_nlp_extract_entities(ctx, chunk, page_entities, 50);
        } __except(EXCEPTION_EXECUTE_HANDLER) {
            app_log(LOG_ERROR, "ONNX NER crashed on page %d (%s) -- skipping",
                    d, domains ? domains[d] : "unknown");
            found = 0;
        }
        if (found > 0)
            app_log(LOG_DEBUG, "NER page %d (%s): %d entities found", d, domains ? domains[d] : "?", found);
        else
            app_log(LOG_DEBUG, "NER page %d (%s): 0 entities (text: %.80s...)", d, domains ? domains[d] : "?", chunk);

        for (int e = 0; e < found && total < max_entities; e++) {
            /* Check if entity already seen */
            int existing = -1;
            for (int x = 0; x < total; x++) {
                if (strcmp(out[x].text, page_entities[e].text) == 0 &&
                    strcmp(out[x].label, page_entities[e].label) == 0) {
                    existing = x; break;
                }
            }
            if (existing >= 0) {
                out[existing].frequency++;
                out[existing].source_count++;
            } else {
                out[total] = page_entities[e];
                total++;
            }
        }
    }

    /* Sort by frequency descending */
    if (total > 1) {
        /* Simple insertion sort -- entity lists are small (<200) */
        for (int i = 1; i < total; i++) {
            NLPEntity tmp = out[i];
            int j = i - 1;
            while (j >= 0 && out[j].frequency < tmp.frequency) {
                out[j+1] = out[j]; j--;
            }
            out[j+1] = tmp;
        }
    }

    app_log(LOG_INFO, "ONNX NER batch: %d unique entities from %d pages", total, text_count);
    return total;
}

/* ── Embedding inference ─────────────────────────────────────── */

#define EMBED_DIM 384  /* all-MiniLM-L6-v2 output dimension */

int onnx_nlp_embed(OnnxNLP *ctx, const char *text,
                    float *out_embedding, int max_dim) {
    if (!ctx || !ctx->available) return 0;
    if (max_dim < EMBED_DIM) return 0;

#ifdef USE_ONNX
    if (!ctx->embed_session) return 0;
    const OrtApi *api = ctx->api;

    /* Tokenize (heap-allocated) */
    TokenResult *tok = (TokenResult *)calloc(1, sizeof(TokenResult));
    if (!tok) return 0;
    tokenize(ctx, text, tok);
    if (tok->count < 3) { free(tok); return 0; }

    /* Create input tensors (heap-allocated) */
    int64_t shape[] = {1, tok->count};
    int64_t *input_ids = (int64_t *)calloc(tok->count, sizeof(int64_t));
    int64_t *attn_mask = (int64_t *)calloc(tok->count, sizeof(int64_t));
    int64_t *type_ids = (int64_t *)calloc(tok->count, sizeof(int64_t));
    if (!input_ids || !attn_mask || !type_ids) {
        free(input_ids); free(attn_mask); free(type_ids); free(tok); return 0;
    }
    for (int i = 0; i < tok->count; i++) {
        input_ids[i] = tok->ids[i];
        attn_mask[i] = tok->attention[i];
        type_ids[i] = 0;
    }

    OrtMemoryInfo *mem_info;
    api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);

    OrtValue *input_tensors[3] = {NULL, NULL, NULL};
    api->CreateTensorWithDataAsOrtValue(mem_info, input_ids, tok->count * sizeof(int64_t),
        shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensors[0]);
    api->CreateTensorWithDataAsOrtValue(mem_info, attn_mask, tok->count * sizeof(int64_t),
        shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensors[1]);
    api->CreateTensorWithDataAsOrtValue(mem_info, type_ids, tok->count * sizeof(int64_t),
        shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensors[2]);

    /* MiniLM output name is "last_hidden_state" -- shape [1, seq_len, 384] */
    const char *input_names[] = {"input_ids", "attention_mask", "token_type_ids"};
    const char *output_names[] = {"last_hidden_state"};
    OrtValue *output = NULL;

    OrtStatus *status = api->Run(ctx->embed_session, NULL,
        input_names, (const OrtValue *const *)input_tensors, 3,
        output_names, 1, &output);

    api->ReleaseMemoryInfo(mem_info);

    if (status) {
        /* Some models use "output_0" instead of "last_hidden_state" -- try fallback */
        api->ReleaseStatus(status);
        const char *alt_names[] = {"output_0"};
        status = api->Run(ctx->embed_session, NULL,
            input_names, (const OrtValue *const *)input_tensors, 3,
            alt_names, 1, &output);
        if (status) {
            app_log(LOG_ERROR, "ONNX Embed Run failed: %s", api->GetErrorMessage(status));
            api->ReleaseStatus(status);
            for (int i = 0; i < 3; i++) if (input_tensors[i]) api->ReleaseValue(input_tensors[i]);
            free(input_ids); free(attn_mask); free(type_ids); free(tok);
            return 0;
        }
    }

    /* Get output tensor data -- shape [1, seq_len, embed_dim] */
    float *hidden = NULL;
    api->GetTensorMutableData(output, (void **)&hidden);

    /* Get actual dimensions */
    OrtTensorTypeAndShapeInfo *shape_info;
    api->GetTensorTypeAndShape(output, &shape_info);
    size_t dim_count;
    api->GetDimensionsCount(shape_info, &dim_count);
    int64_t dims[4] = {0};
    api->GetDimensions(shape_info, dims, dim_count);
    api->ReleaseTensorTypeAndShapeInfo(shape_info);

    int seq_len = (dim_count >= 2) ? (int)dims[1] : tok->count;
    int embed_dim = (dim_count >= 3) ? (int)dims[2] : EMBED_DIM;
    if (embed_dim > max_dim) embed_dim = max_dim;

    /* Mean pool: average hidden states across sequence dimension (attention-weighted) */
    memset(out_embedding, 0, embed_dim * sizeof(float));
    int valid_tokens = 0;
    for (int t = 0; t < seq_len; t++) {
        if (t < tok->count && tok->attention[t]) {
            for (int d = 0; d < embed_dim; d++)
                out_embedding[d] += hidden[t * embed_dim + d];
            valid_tokens++;
        }
    }
    if (valid_tokens > 0) {
        float inv = 1.0f / valid_tokens;
        for (int d = 0; d < embed_dim; d++)
            out_embedding[d] *= inv;
    }

    /* L2 normalize */
    float norm = 0;
    for (int d = 0; d < embed_dim; d++) norm += out_embedding[d] * out_embedding[d];
    if (norm > 0) {
        norm = sqrtf(norm);
        for (int d = 0; d < embed_dim; d++) out_embedding[d] /= norm;
    }

    /* Cleanup */
    if (output) api->ReleaseValue(output);
    for (int i = 0; i < 3; i++) if (input_tensors[i]) api->ReleaseValue(input_tensors[i]);
    free(input_ids); free(attn_mask); free(type_ids); free(tok);

    return embed_dim;
#else
    (void)text; (void)out_embedding; (void)max_dim;
    return 0;
#endif
}

/* ── Batch embedding inference ──────────────────────────────── */

#define BATCH_SIZE 32  /* max texts per ONNX call -- safe for CPU memory */

int onnx_nlp_embed_batch(OnnxNLP *ctx, const char **texts, int text_count,
                         float *out_embeddings, int max_dim) {
    if (!ctx || !ctx->available) return 0;
    if (max_dim < EMBED_DIM) return 0;
    if (text_count <= 0) return 0;

#ifdef USE_ONNX
    if (!ctx->embed_session) return 0;
    const OrtApi *api = ctx->api;

    /* Process in chunks of BATCH_SIZE */
    int any_succeeded = 0;
    for (int chunk_start = 0; chunk_start < text_count; chunk_start += BATCH_SIZE) {
        int chunk_size = text_count - chunk_start;
        if (chunk_size > BATCH_SIZE) chunk_size = BATCH_SIZE;

        /* 1. Tokenize all texts in this chunk, find max sequence length */
        TokenResult **toks = (TokenResult **)calloc(chunk_size, sizeof(TokenResult *));
        if (!toks) return 0;

        int max_seq = 0;
        int alloc_ok = 1;
        for (int i = 0; i < chunk_size; i++) {
            toks[i] = (TokenResult *)calloc(1, sizeof(TokenResult));
            if (!toks[i]) { alloc_ok = 0; break; }
            tokenize(ctx, texts[chunk_start + i], toks[i]);
            if (toks[i]->count < 3) {
                /* Very short/empty text -- give it minimal tokens so it doesn't break batch */
                toks[i]->count = 3;
            }
            if (toks[i]->count > max_seq) max_seq = toks[i]->count;
        }
        if (!alloc_ok) {
            for (int i = 0; i < chunk_size; i++) free(toks[i]);
            free(toks);
            memset(&out_embeddings[chunk_start * max_dim], 0,
                   chunk_size * max_dim * sizeof(float));
            continue;
        }

        /* 2. Allocate padded input arrays: [chunk_size * max_seq] */
        int64_t total_elements = (int64_t)chunk_size * max_seq;
        int64_t *input_ids = (int64_t *)calloc(total_elements, sizeof(int64_t));
        int64_t *attn_mask = (int64_t *)calloc(total_elements, sizeof(int64_t));
        int64_t *type_ids  = (int64_t *)calloc(total_elements, sizeof(int64_t));
        if (!input_ids || !attn_mask || !type_ids) {
            free(input_ids); free(attn_mask); free(type_ids);
            for (int i = 0; i < chunk_size; i++) free(toks[i]);
            free(toks);
            memset(&out_embeddings[chunk_start * max_dim], 0,
                   chunk_size * max_dim * sizeof(float));
            continue;
        }

        /* 3. Fill: actual tokens + attention=1 for real, zeros for padding */
        for (int i = 0; i < chunk_size; i++) {
            int64_t row_offset = (int64_t)i * max_seq;
            for (int j = 0; j < toks[i]->count; j++) {
                input_ids[row_offset + j] = toks[i]->ids[j];
                attn_mask[row_offset + j] = toks[i]->attention[j];
                type_ids[row_offset + j]  = 0;
            }
            /* Remaining positions are already 0 from calloc (padding) */
        }

        /* 4. Create tensors with shape [chunk_size, max_seq] */
        int64_t shape[] = {chunk_size, max_seq};
        OrtMemoryInfo *mem_info;
        api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);

        OrtValue *input_tensors[3] = {NULL, NULL, NULL};
        api->CreateTensorWithDataAsOrtValue(mem_info, input_ids,
            total_elements * sizeof(int64_t), shape, 2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensors[0]);
        api->CreateTensorWithDataAsOrtValue(mem_info, attn_mask,
            total_elements * sizeof(int64_t), shape, 2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensors[1]);
        api->CreateTensorWithDataAsOrtValue(mem_info, type_ids,
            total_elements * sizeof(int64_t), shape, 2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensors[2]);

        /* 5. Run ONNX once for the whole chunk */
        const char *input_names[] = {"input_ids", "attention_mask", "token_type_ids"};
        const char *output_names[] = {"last_hidden_state"};
        OrtValue *output = NULL;

        OrtStatus *status = api->Run(ctx->embed_session, NULL,
            input_names, (const OrtValue *const *)input_tensors, 3,
            output_names, 1, &output);

        if (status) {
            /* Fallback: some models use "output_0" */
            api->ReleaseStatus(status);
            const char *alt_names[] = {"output_0"};
            status = api->Run(ctx->embed_session, NULL,
                input_names, (const OrtValue *const *)input_tensors, 3,
                alt_names, 1, &output);
            if (status) {
                app_log(LOG_ERROR, "ONNX Embed batch Run failed (chunk %d): %s",
                        chunk_start, api->GetErrorMessage(status));
                api->ReleaseStatus(status);
                api->ReleaseMemoryInfo(mem_info);
                for (int i = 0; i < 3; i++)
                    if (input_tensors[i]) api->ReleaseValue(input_tensors[i]);
                free(input_ids); free(attn_mask); free(type_ids);
                for (int i = 0; i < chunk_size; i++) free(toks[i]);
                free(toks);
                /* Zero out this chunk's embeddings so cosine similarity returns 0 */
                memset(&out_embeddings[chunk_start * max_dim], 0,
                       chunk_size * max_dim * sizeof(float));
                continue;  /* try next chunk instead of discarding everything */
            }
        }

        api->ReleaseMemoryInfo(mem_info);

        /* 6. Get output tensor -- shape [chunk_size, max_seq, embed_dim] */
        float *hidden = NULL;
        api->GetTensorMutableData(output, (void **)&hidden);

        OrtTensorTypeAndShapeInfo *shape_info;
        api->GetTensorTypeAndShape(output, &shape_info);
        size_t dim_count;
        api->GetDimensionsCount(shape_info, &dim_count);
        int64_t dims[4] = {0};
        api->GetDimensions(shape_info, dims, dim_count);
        api->ReleaseTensorTypeAndShapeInfo(shape_info);

        int out_seq_len = (dim_count >= 2) ? (int)dims[1] : max_seq;
        int embed_dim = (dim_count >= 3) ? (int)dims[2] : EMBED_DIM;
        if (embed_dim > max_dim) embed_dim = max_dim;

        /* 7. For each text: mean-pool with attention mask, L2-normalize */
        for (int i = 0; i < chunk_size; i++) {
            float *emb = &out_embeddings[(chunk_start + i) * max_dim];
            float *text_hidden = &hidden[(int64_t)i * out_seq_len * embed_dim];
            memset(emb, 0, embed_dim * sizeof(float));

            int valid_tokens = 0;
            for (int t = 0; t < out_seq_len; t++) {
                if (t < toks[i]->count && toks[i]->attention[t]) {
                    for (int d = 0; d < embed_dim; d++)
                        emb[d] += text_hidden[t * embed_dim + d];
                    valid_tokens++;
                }
            }
            if (valid_tokens > 0) {
                float inv = 1.0f / valid_tokens;
                for (int d = 0; d < embed_dim; d++)
                    emb[d] *= inv;
            }

            /* L2 normalize */
            float norm = 0;
            for (int d = 0; d < embed_dim; d++) norm += emb[d] * emb[d];
            if (norm > 0) {
                norm = sqrtf(norm);
                for (int d = 0; d < embed_dim; d++) emb[d] /= norm;
            }
        }

        any_succeeded = 1;

        /* 8. Free chunk resources */
        if (output) api->ReleaseValue(output);
        for (int i = 0; i < 3; i++)
            if (input_tensors[i]) api->ReleaseValue(input_tensors[i]);
        free(input_ids); free(attn_mask); free(type_ids);
        for (int i = 0; i < chunk_size; i++) free(toks[i]);
        free(toks);
    }

    return any_succeeded ? EMBED_DIM : 0;
#else
    (void)texts; (void)text_count; (void)out_embeddings; (void)max_dim;
    return 0;
#endif
}

float onnx_nlp_similarity(const float *a, const float *b, int dim) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = sqrtf(na) * sqrtf(nb);
    return denom > 0 ? dot / denom : 0;
}
