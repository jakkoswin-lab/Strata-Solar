// Solar's template mask (include/strata/glm4/tmpl_mask.hpp), CPU only, no model: which tokens a step may emit.
#include "strata/glm4/tmpl_mask.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::glm4;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

/// A logit row where every token ties, so "allowed" is exactly "finite".
std::vector<float> flat(int n = 256) { return std::vector<float>((size_t) n, 1.f); }

bool allowed(const std::vector<float>& lg, int id) {
    return (size_t) id < lg.size() && lg[(size_t) id] != tmpl_detail::kNever;
}

int allowed_count(const std::vector<float>& lg) {
    int n = 0;
    for (float v : lg) n += v != tmpl_detail::kNever;
    return n;
}
}  // namespace

int main() {
    {   // content only: the first step can only be <|content|>
        Mask m;
        m.mode = kContentOnly;
        auto lg = flat();
        mask(lg, m);
        check(allowed_count(lg) == 1 && allowed(lg, kContent), "content-only start");
    }
    {   // a tool turn: <|content|> or <|tool_calls|>, nothing else (a regular token is not allowed either)
        Mask m;
        m.mode = kContentOrTools;
        auto lg = flat();
        mask(lg, m);
        check(allowed_count(lg) == 2 && allowed(lg, kContent) && allowed(lg, kToolCalls), "tool turn start");
    }
    {   // a forced call: only <|tool_calls|>
        Mask m;
        m.mode = kToolsOnly;
        auto lg = flat();
        mask(lg, m);
        check(allowed_count(lg) == 1 && allowed(lg, kToolCalls), "forced call start");
    }
    {   // content: regular tokens stay, <|end|> is folded into <|flush|>, the other markers go
        Mask m;
        m.mode = kContentOnly;
        m.state = sContent;
        auto lg = flat();
        lg[kEnd] = 0.5f;                       // the model prefers to end with <|end|>
        lg[kBegin] = 9.f;                      // and might open the next message
        mask(lg, m);
        check(allowed(lg, 7), "content keeps regular tokens");
        check(allowed(lg, kFlush) && lg[kFlush] >= 0.5f, "content folds <|end|> into <|flush|>");
        check(!allowed(lg, kEnd) && !allowed(lg, kBegin) && !allowed(lg, kToolCalls), "content masks markers");
    }
    {   // <|tool_calls|> forces <|tool_call:begin|>
        Mask m;
        m.mode = kContentOrTools;
        m.state = sCallBegin;
        auto lg = flat();
        mask(lg, m);
        check(allowed_count(lg) == 1 && allowed(lg, kCallBegin), "call begin");
    }
    {   // the id: regular tokens, then the name marker after ten of them
        Mask m;
        m.mode = kContentOrTools;
        m.state = sCallId;
        m.id_tokens = 3;
        auto lg = flat();
        mask(lg, m);
        check(allowed(lg, 12) && allowed(lg, kCallName), "id keeps regular tokens and the name marker");
        check(!allowed(lg, kCallArgs) && !allowed(lg, kCallEnd), "id masks the later markers");
        for (int i = 0; i < 7; ++i) advance(m, 65);          // ten id tokens in all
        auto lg2 = flat();
        mask(lg2, m);
        check(allowed_count(lg2) == 1 && allowed(lg2, kCallName), "id limit forces the name marker");
    }
    {   // name -> args -> end
        Mask m;
        m.mode = kContentOrTools;
        m.state = sCallName;
        auto lg = flat();
        mask(lg, m);
        check(allowed(lg, 5) && allowed(lg, kCallArgs) && !allowed(lg, kCallEnd), "name state");
        advance(m, 65);
        advance(m, kCallArgs);
        check(m.state == sCallArgs, "args state reached");
        auto lg2 = flat();
        mask(lg2, m);
        check(allowed(lg2, kCallEnd) && !allowed(lg2, kCallName), "args state");
        advance(m, 65);
        advance(m, kCallEnd);
        check(m.state == sCallEnd, "end state reached");
        auto lg3 = flat();
        mask(lg3, m);
        check(allowed_count(lg3) == 2 && allowed(lg3, kCallBegin) && allowed(lg3, kCalls), "end: another call or <|calls|>");
    }
    {   // a full call, driven token by token: the state machine reaches sDone only at <|calls|>
        Mask m;
        m.mode = kContentOrTools;
        const int ids[] = {kToolCalls, kCallBegin, 'a', '1', kCallName, 's', 'h', kCallArgs, '{', '}', kCallEnd, kCalls};
        for (int t : ids) {
            auto lg = flat();
            mask(lg, m);
            check(allowed(lg, t), "the next token of a well-formed call is allowed");
            advance(m, t);
        }
        check(m.state == sDone, "the turn ends at <|calls|>");
        auto lg = flat();
        mask(lg, m);
        check(allowed_count(lg) == (int) lg.size(), "nothing is masked after the turn");
    }
    {   // a second call in the same turn
        Mask m;
        m.mode = kContentOrTools;
        const int ids[] = {kToolCalls, kCallBegin, 'x', kCallName, 'f', kCallArgs, '{', '}', kCallEnd, kCallBegin};
        for (int t : ids) advance(m, t);
        check(m.state == sCallId && m.id_tokens == 0, "a second call starts over");
    }
    {   // an unexpected token ends the masking (defensive: the mask should not have let it through)
        Mask m;
        m.mode = kContentOnly;
        advance(m, kToolCalls);
        check(m.state == sDone, "an unexpected token ends the masking");
    }
    {   // kOff changes nothing, byte for byte
        Mask m;
        auto lg = flat();
        auto before = lg;
        mask(lg, m);
        check(lg == before, "off is a no-op");
    }
    std::printf("tmpl_mask: %d checks ok\n", checks);
    return 0;
}
