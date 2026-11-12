#pragma once
// Solar-Open-100B's own generation format, enforced the way Upstage's vLLM serves it: `SolarOpenTemplateLogitsProcessor`
// in their vLLM fork (`vllm/model_executor/models/solar_open_logits_processor.py`) masks every step to what the chat
// template allows.  Without it the model is free to leave the format - measured here: Qwen's
// `<tool_call>name key="value"</tool_call>`, a call with `<|tool_call:name|>` left out, and
// `<tool_call>[execute]</tool_call>` with the payload in a fence.
//
// The rules, in their order (non-reasoning: the template's empty think block is already written by the prompt):
//
//   <|content|> ... <|flush|>                                  content, and it cannot open another message
//   <|tool_calls|> (<|tool_call:begin|> ID(<=10) <|tool_call:name|> NAME <|tool_call:args|> JSON
//                   <|tool_call:end|>)+ <|calls|>              one or more calls, then the turn's end
//
// Host only, no CUDA: a unit test (src/glm4/tmpl_mask_test.cpp) runs it without a GPU.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <vector>

namespace strata::glm4 {

// The family's markers, as its tokenizer numbers them (checked against the pack: <|begin|> 20, <|end|> 21,
// <|think|> 22, <|content|> 23, <|flush|> 24, <|calls|> 25, <|tool_calls|> 30, <|tool_call:begin|> 31, :end 32,
// :name 33, :args 34, and <|tool_response|> 35 .. 39).  20..39 are the message markers: the model may write one
// only where the template puts it.
inline constexpr int kBegin = 20, kEnd = 21, kThink = 22, kContent = 23, kFlush = 24, kCalls = 25;
inline constexpr int kToolCalls = 30, kCallBegin = 31, kCallEnd = 32, kCallName = 33, kCallArgs = 34;
inline constexpr int kFirstMarker = 20, kLastMarker = 39;

enum MaskMode { kOff = 0, kContentOnly = 1, kContentOrTools = 2, kToolsOnly = 3 };
enum MaskState {
    sInit,          // nothing written: a content reply or a tool call
    sContent,       // writing the answer
    sCallBegin,     // <|tool_calls|> written: <|tool_call:begin|> must follow
    sCallId,        // writing the call id
    sCallName,      // after <|tool_call:name|>
    sCallArgs,      // after <|tool_call:args|>
    sCallEnd,       // after <|tool_call:end|>: another call or <|calls|>
    sDone,          // the turn ended; nothing more is masked
};

struct Mask {
    int mode = kOff;
    int state = sInit;
    int id_tokens = 0;              // id tokens written after <|tool_call:begin|> (the template allows 10)
    int id_limit = 10;

    bool on() const { return mode != kOff && state != sDone; }
};

namespace tmpl_detail {
constexpr float kNever = -std::numeric_limits<float>::max();   // nvcc rejects -INFINITY here

inline bool has(std::initializer_list<int> ids, int id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

/// Leave only `allow` with a finite logit (the step may not be a regular token at all).
inline void only(std::vector<float>& lg, std::initializer_list<int> allow) {
    float keep[4] = {0.f, 0.f, 0.f, 0.f};
    int n = 0;
    for (int id : allow) keep[n++] = (size_t) id < lg.size() ? lg[(size_t) id] : kNever;
    std::fill(lg.begin(), lg.end(), kNever);
    n = 0;
    for (int id : allow)
        if ((size_t) id < lg.size()) lg[(size_t) id] = keep[n++];
}

/// Regular tokens stay allowed; the markers do not, except those in `allow`.
inline void except(std::vector<float>& lg, std::initializer_list<int> allow) {
    for (int id = kFirstMarker; id <= kLastMarker; ++id)
        if ((size_t) id < lg.size() && !has(allow, id)) lg[(size_t) id] = kNever;
}
}  // namespace tmpl_detail

/// The next step's logits with everything the template does not allow at -inf.
inline void mask(std::vector<float>& lg, const Mask& m) {
    using tmpl_detail::except;
    using tmpl_detail::only;
    if (!m.on()) return;
    switch (m.state) {
        case sInit:
            if (m.mode == kContentOnly) only(lg, {kContent});
            else if (m.mode == kToolsOnly) only(lg, {kToolCalls});
            else only(lg, {kContent, kToolCalls});
            break;
        case sContent:
            // The template writes a content turn as <|content|> ... <|flush|>, and its processor turns <|end|> into
            // <|flush|> so that a content turn cannot open another message.  Same here, then only <|flush|> survives.
            if ((size_t) kFlush < lg.size() && (size_t) kEnd < lg.size())
                lg[(size_t) kFlush] = std::max(lg[(size_t) kFlush], lg[(size_t) kEnd]);
            except(lg, {kFlush});
            break;
        case sCallBegin:
            only(lg, {kCallBegin});
            break;
        case sCallId:
            if (m.id_tokens >= m.id_limit) only(lg, {kCallName});
            else except(lg, {kCallName});
            break;
        case sCallName:
            except(lg, {kCallArgs});
            break;
        case sCallArgs:
            except(lg, {kCallEnd});
            break;
        case sCallEnd:
            only(lg, {kCallBegin, kCalls});
            break;
        default:
            break;
    }
}

/// The state after `token` was sampled.
inline void advance(Mask& m, int token) {
    if (!m.on()) return;
    switch (m.state) {
        case sInit:
            if (token == kContent) m.state = sContent;
            else if (token == kToolCalls && m.mode != kContentOnly) m.state = sCallBegin;
            else m.state = sDone;
            break;
        case sContent:
            if (token == kFlush || token == kEnd) m.state = sDone;
            break;
        case sCallBegin:
            m.state = sCallId;
            m.id_tokens = 0;
            break;
        case sCallId:
            if (token == kCallName) m.state = sCallName;
            else ++m.id_tokens;
            break;
        case sCallName:
            if (token == kCallArgs) m.state = sCallArgs;
            break;
        case sCallArgs:
            if (token == kCallEnd) m.state = sCallEnd;
            break;
        case sCallEnd:
            if (token == kCallBegin) {
                m.state = sCallId;
                m.id_tokens = 0;
            } else {
                m.state = sDone;            // <|calls|> ends the turn
            }
            break;
        default:
            break;
    }
}

}  // namespace strata::glm4
